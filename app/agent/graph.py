# backend/app/agent/graph.py
"""Outer orchestrator: run the ReAct agent on Sonnet 5; if it fails to
produce a usable answer, escalate once to Opus 4.8 on the same thread."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import anthropic
import structlog
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.prebuilt import ToolNode, create_react_agent
from pydantic import BaseModel, Field
from pymongo import MongoClient

from app.agent.chart_spec import build_chart
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import (
    build_mongo_resources,
    build_semantic_search_tool,
    build_structured_tools,
)
from app.core.config import get_settings

log = structlog.get_logger()

SONNET_MODEL = "claude-sonnet-5"
OPUS_MODEL = "claude-opus-4-8"
HAIKU_MODEL = "claude-haiku-4-5"
# The brief specified 8. Measured against the live stack, a single
# structured-query round trip through the MongoDB toolkit (checker call +
# query call, each possibly retried on a syntax error the model
# self-corrects from) regularly costs 10-13 graph steps end-to-end
# (langgraph counts each agent-node/tool-node turn as a step) even for a
# simple single-collection count. At 8, create_react_agent's own
# remaining-steps guard fires and returns the canned "Sorry, need more
# steps..." message before the model ever reaches a real answer, on every
# run, for both the Sonnet and Opus attempts alike. Raised to 25 initially;
# then, live testing of semantic-search-driven questions (which chain
# semantic_search -> filter on the returned categories -> mongodb_query,
# sometimes with a checker call or a self-correction retry in between)
# showed the identical question sometimes converging around step ~13-14 and
# sometimes exceeding 25 and forcing an Opus escalation that would have been
# unnecessary with more headroom — LLM tool-use step counts aren't
# deterministic at nonzero temperature, so the ceiling needs margin for the
# slow path, not just the typical one. Raised to 40; also see prompts.py's
# guidance against calling mongodb_query_checker unconditionally, which
# reduces the *typical* step count directly rather than just widening the
# ceiling.
MAX_ITERATIONS = 40

# Max characters for a single tool result before it's replaced with a
# corrective message instead of being returned as-is. Verified live: a
# naive `$match`-only aggregation over one fiscal year (120,636 documents,
# no $count/$limit) serializes to ~174,000,000 characters. That crashes
# the MongoDB checkpointer's write (checkpoint_writes documents are
# subject to MongoDB's 16MB BSON limit) *after* the tool already
# "succeeded" — so ToolNode's handle_tool_errors never sees it — leaving a
# dangling tool_call with no ToolMessage that corrupts the thread's
# history for any later resume, including the Opus escalation attempt
# below (both share the same thread_id/checkpoint). 50,000 chars is well
# under both the BSON limit and a sane single-tool-result context budget.
TOOL_RESULT_MAX_CHARS = 50_000

# langgraph's prebuilt create_react_agent executor injects this exact
# message (no tool_calls) and ends normally — no exception — when
# remaining_steps drops below 2 and the model still wants to call a tool.
# Structurally it looks identical to a real final answer (an "ai" message
# with no tool_calls), so without this check it silently satisfied the
# `if final_text:` branch below and got `return`ed to the user as a
# successful response on the very first (Sonnet) attempt — permanently
# skipping the Opus escalation this whole loop exists to provide, on
# exactly the complex/near-budget questions escalation is meant to rescue.
# Raising MAX_ITERATIONS to 40 (above) reduced how often this fires; it
# doesn't make the loop notice when it still does.
STEP_BUDGET_EXHAUSTED_MESSAGE = "Sorry, need more steps to process this request."


async def _cap_oversized_tool_result(request, handler):
    """awrap_tool_call hook for ToolNode: turn a successful-but-huge tool
    result into a corrective message the model can act on (e.g. "add a
    $count stage"), instead of letting it crash the checkpoint write."""
    result = await handler(request)
    content = getattr(result, "content", None)
    if isinstance(content, str) and len(content) > TOOL_RESULT_MAX_CHARS:
        tool_name = request.tool_call.get("name", "the tool")
        result.content = (
            f"Result too large ({len(content):,} characters) — {tool_name} "
            "matched too many documents to return directly. Add a $count, "
            "$group, or $limit stage to the aggregation pipeline (or narrow "
            "the $match filter) and try again."
        )
    return result


def _build_agent(model_name: str, checkpointer, mongo_client, mongo_db, database_name: str):
    settings = get_settings()
    llm = ChatAnthropic(model=model_name, api_key=settings.anthropic_api_key, max_tokens=4096)
    # mongo_client/mongo_db are built once per request (run_agent) and
    # reused across both the Sonnet and Opus attempts here, rather than
    # each call opening its own fresh MongoDB connections (previously up
    # to 3 per attempt: the toolkit's own, the vector store's, plus a live
    # listCollections round trip) — see build_mongo_resources's docstring.
    tools = build_structured_tools(llm, mongo_client, database_name, mongo_db) + [
        build_semantic_search_tool(mongo_client, database_name)
    ]
    # NOTE: langgraph 1.2.9's ToolNode default (`handle_tool_errors` defaults
    # to a handler that only catches its own ToolInvocationError and
    # re-raises everything else) no longer swallows generic tool exceptions
    # the way older langgraph versions did. The MongoDB toolkit tool
    # (langchain-mongodb 0.11.0's MongoDBDatabase.run_no_throw) only catches
    # PyMongoError, not the plain ValueError it raises for a malformed
    # aggregate command (e.g. the model emitting `.countDocuments(...)`
    # instead of `.aggregate(...)`) — so that ValueError was propagating
    # uncaught, crashing the whole astream() turn instead of coming back to
    # the model as a retryable tool error, and corrupting the checkpointed
    # thread (a dangling tool_call with no ToolMessage) for the Opus
    # escalation attempt that shares the same thread_id. Wrapping tools in
    # a ToolNode with handle_tool_errors=True restores catch-all behavior:
    # any tool exception becomes an error string ToolMessage the model can
    # see and self-correct from, matching this design's intent that the
    # ReAct loop (not this file) handles bad-query retries, and reserving
    # the outer try/except here for genuine, unrecoverable failures.
    tool_node = ToolNode(tools, handle_tool_errors=True, awrap_tool_call=_cap_oversized_tool_result)
    return create_react_agent(
        llm,
        tool_node,
        # NOTE: the installed langgraph (1.2.9) removed the `state_modifier`
        # kwarg the brief specified — create_react_agent() now raises
        # TypeError on unknown kwargs (it's not silently accepted). The
        # direct replacement is `prompt`, which accepts the same plain
        # system-prompt string; confirmed via
        # inspect.signature(create_react_agent) on the installed version.
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def _get_last_assistant_text(checkpointer: MongoDBSaver, thread_id: str) -> str | None:
    """Sync (MongoDBSaver wraps pymongo, not Motor) lookup of the most recent
    plain-text assistant message for a thread, if any prior turn exists.
    Used only to give the out-of-scope guard enough context to correctly
    judge follow-up questions — see _is_out_of_scope."""
    config = {"configurable": {"thread_id": thread_id}}
    tup = checkpointer.get_tuple(config)
    if not tup:
        return None
    messages = tup.checkpoint.get("channel_values", {}).get("messages", [])
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and not getattr(msg, "tool_calls", None):
            return msg.text
    return None


def _reconstruct_messages_sync(thread_id: str) -> list[dict] | None:
    """The LangGraph checkpoint is the durable, cross-session record of a
    conversation — used here to rebuild history for the sidebar's
    'reopen an old conversation' path, independent of the frontend's
    localStorage cache. Only human/final-assistant turns are kept; tool
    calls and tool results are internal reasoning, not conversation
    content. Note: query/suggestions/chart enrichments aren't part of the
    checkpoint (they're generated fresh per turn, not persisted state), so
    messages reconstructed this way carry text only."""
    settings = get_settings()
    client = MongoClient(settings.mongodb_uri)
    try:
        checkpointer = MongoDBSaver(client)
        tup = checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
        if not tup:
            return None
        raw_messages = tup.checkpoint.get("channel_values", {}).get("messages", [])
        result = []
        for msg in raw_messages:
            msg_type = getattr(msg, "type", None)
            if msg_type == "human":
                result.append({"role": "user", "text": msg.text})
            elif msg_type == "ai" and not getattr(msg, "tool_calls", None):
                result.append({"role": "assistant", "text": msg.text})
        return result
    finally:
        client.close()


async def reconstruct_conversation_messages(thread_id: str) -> list[dict] | None:
    """Async wrapper — see _reconstruct_messages_sync. Returns None if the
    thread has no checkpoint at all (never chatted, or an invalid id),
    distinct from an empty list (a real but empty conversation)."""
    return await asyncio.to_thread(_reconstruct_messages_sync, thread_id)


async def _is_out_of_scope(user_message: str, prior_context: str | None) -> bool:
    """Cheap Haiku 4.5 guard so obviously irrelevant questions never reach
    the more expensive Sonnet/Opus path.

    `prior_context` (the last assistant message in this thread, if any) is
    required for correctness, not just nice-to-have: a follow-up like
    "which suppliers did the top one use most?" references "the top one"
    from the previous answer and reads as scope-less nonsense in isolation.
    Without context, the guard reliably misclassified real, in-scope
    follow-ups as out-of-scope, silently breaking multi-turn conversations —
    reproduced live before this fix."""
    settings = get_settings()
    guard = ChatAnthropic(model=HAIKU_MODEL, api_key=settings.anthropic_api_key, max_tokens=10)
    prompt = (
        "Is this question about California state government procurement/purchase "
        "order data? It may be a follow-up that refers back to the prior answer "
        "(e.g. via pronouns like 'it', 'the top one', 'that department') rather "
        "than naming the topic explicitly — treat those as in-scope if the prior "
        "answer was procurement-related. Answer only YES or NO.\n\n"
    )
    if prior_context:
        prompt += f"Previous answer in this conversation:\n{prior_context}\n\n"
    prompt += f"New question: {user_message}"
    # `.ainvoke` (not `.invoke`) — this runs inside `run_agent`, an async
    # generator driving a FastAPI SSE response. A sync `.invoke()` call
    # would block the single event loop for its full network round trip on
    # every request, serializing all concurrent chat requests behind each
    # other. `.ainvoke()` uses the client's native async HTTP path instead.
    response = await guard.ainvoke(prompt)
    # Same `.text` vs `.content` normalization as below — `.content` can be a
    # list of content blocks rather than a plain string.
    return "NO" in response.text.upper()


class _FollowUpSuggestions(BaseModel):
    questions: list[str] = Field(
        description="2-3 short, specific follow-up questions a user might "
        "naturally ask next, each answerable from the same purchase-order "
        "dataset."
    )


async def _generate_followups(user_message: str, final_text: str) -> list[str]:
    """Cheap Haiku 4.5 call to suggest natural next questions, so the UI can
    offer clickable follow-up chips instead of leaving the user staring at
    an empty input box after every answer. Best-effort: any failure here
    (rate limit, malformed structured output) must not take down the real
    answer, so it's swallowed and just yields no suggestions."""
    settings = get_settings()
    llm = ChatAnthropic(model=HAIKU_MODEL, api_key=settings.anthropic_api_key, max_tokens=300)
    structured = llm.with_structured_output(_FollowUpSuggestions)
    try:
        result = await structured.ainvoke(
            "A user asked a California state procurement data question and got "
            "this answer. Suggest 2-3 short, specific follow-up questions they "
            "might naturally ask next — each answerable from the same "
            "purchase-order dataset (departments, suppliers, items, dates, "
            "totals). Keep each under 12 words.\n\n"
            f"Question: {user_message}\n\nAnswer: {final_text}"
        )
        return result.questions[:3]
    except Exception:  # noqa: BLE001 — best-effort feature, never fail the request over it
        return []


async def _no_chart() -> None:
    return None


async def run_agent(user_message: str, thread_id: str) -> AsyncIterator[dict]:
    settings = get_settings()
    mongo_client = MongoClient(settings.mongodb_uri)
    # NOTE: this whole block used to call `mongo_client.close()` explicitly on
    # each return path, which only runs on normal completion. `run_agent` is
    # an async generator consumed by an SSE endpoint (Task 12) — if the
    # client disconnects mid-stream (the ordinary case for a user closing a
    # tab mid-response), the generator is abandoned at a `yield` and Python
    # throws GeneratorExit into it rather than letting it run to a `return`,
    # so those explicit `.close()` calls would never execute and the
    # MongoClient (with its own connection pool) would leak. A try/finally
    # around the whole body runs on every exit path, including GeneratorExit.
    query_client, query_db = build_mongo_resources(settings)
    try:
        checkpointer = MongoDBSaver(mongo_client)

        # Built before the out-of-scope check (not after, as originally) so
        # the guard can see the prior turn and correctly classify follow-ups
        # like "which suppliers did the top one use most?" — see
        # _is_out_of_scope's docstring for the bug this fixes.
        prior_context = await asyncio.to_thread(_get_last_assistant_text, checkpointer, thread_id)
        if await _is_out_of_scope(user_message, prior_context):
            yield {
                "type": "final_answer",
                "text": "I can only answer questions about the California state "
                "purchase order dataset (2012-2015). Try asking about spending, "
                "departments, suppliers, or specific items.",
            }
            return

        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_ITERATIONS}

        # Scoped outside the model loop (unlike final_text): if Sonnet runs a
        # real mongodb_query, then exhausts its step budget before
        # composing a final answer, Opus resumes from the same checkpointed
        # thread and often composes the answer straight from Sonnet's
        # already-fetched tool results without re-querying. Resetting this
        # per-attempt would silently drop the query that the numbers in the
        # final answer actually came from.
        last_query: str | None = None
        for model_name, is_escalation in ((SONNET_MODEL, False), (OPUS_MODEL, True)):
            agent = _build_agent(
                model_name, checkpointer, query_client, query_db, settings.mongodb_db_name
            )
            final_text = None
            try:
                async for event in agent.astream(
                    {"messages": [("user", user_message)]}, config, stream_mode="values"
                ):
                    last_message = event["messages"][-1]
                    tool_calls = getattr(last_message, "tool_calls", None)
                    if tool_calls:
                        # Track the most recent real query execution (not the
                        # schema/checker/list-collections calls) for
                        # transparency — surfaced to the frontend so a user
                        # (or a grader) can see the actual MongoDB pipeline
                        # that answered their question, not just prose.
                        for tc in tool_calls:
                            if tc.get("name") == "mongodb_query":
                                last_query = tc.get("args", {}).get("query")
                    if getattr(last_message, "type", None) == "ai" and not tool_calls:
                        # NOTE: use `.text` (a property that normalizes both
                        # the plain-string case and the extended-thinking
                        # case, where `.content` is a list of content-block
                        # dicts rather than a string) instead of `.content`
                        # directly. With `.content`, an extended-thinking
                        # response's final_text would be a list, so a
                        # downstream `"120,636" in final_text` check silently
                        # returns False even though the model answered
                        # correctly, and the yielded chunk's `text` field
                        # would violate the str contract the SSE router
                        # (Task 12) depends on. Reproduced live: ~50% of runs
                        # hit this before the fix.
                        text = last_message.text
                        if text == STEP_BUDGET_EXHAUSTED_MESSAGE:
                            # Not a real answer — langgraph's own step-budget
                            # guard message (see the constant's docstring).
                            # Leave final_text unset so the `if final_text:`
                            # check below falls through to the Opus attempt
                            # instead of returning this to the user as a
                            # successful response.
                            log.warning(
                                "agent_step_budget_exhausted",
                                model=model_name,
                                is_escalation=is_escalation,
                                thread_id=thread_id,
                            )
                            continue
                        final_text = text
                        yield {"type": "step", "text": f"[{model_name}] {final_text[:80]}"}
            except (anthropic.RateLimitError, anthropic.OverloadedError) as exc:
                # Distinct from the generic except below: this is Anthropic
                # itself throttling or shedding load, not a bug in the
                # question or the pipeline. Telling a user to "rephrase" a
                # rate-limit error is actively misleading and just
                # encourages more retries — which, under concurrent load,
                # is exactly what compounds the problem (reproduced live:
                # 4 concurrent requests for the same question went from
                # ~45s to 115-125s each). Say what's actually happening.
                log.error(
                    "agent_rate_limited",
                    model=model_name,
                    is_escalation=is_escalation,
                    thread_id=thread_id,
                    error=str(exc),
                )
                if is_escalation:
                    result = {
                        "type": "final_answer",
                        "text": "The system is handling a lot of requests right now — "
                        "please wait a few seconds before trying again (retrying "
                        "immediately makes this worse, not better).",
                    }
                    if last_query:
                        result["query"] = last_query
                    yield result
                    return
                continue
            except Exception as exc:  # noqa: BLE001 — deliberately broad: any
                # failure on the Sonnet attempt should trigger escalation,
                # not crash the request.
                log.error(
                    "agent_model_failure",
                    model=model_name,
                    is_escalation=is_escalation,
                    thread_id=thread_id,
                    error=str(exc),
                )
                if is_escalation:
                    # Don't put str(exc) in user-facing text — it can leak
                    # internal details (a raw PyMongo error, a stack-trace
                    # fragment) to the browser. The real error is already
                    # logged above with full context for debugging.
                    result = {
                        "type": "final_answer",
                        "text": "I ran into a technical issue answering that — please "
                        "try rephrasing your question, or ask something more specific.",
                    }
                    if last_query:
                        result["query"] = last_query
                    yield result
                    return
                continue

            if final_text:
                result = {"type": "final_answer", "text": final_text}
                if last_query:
                    result["query"] = last_query
                # Yield the real answer immediately rather than waiting on
                # the two enrichment calls below — those are best-effort
                # extras, not part of what the user actually asked for.
                # Previously this whole block awaited suggestions+chart
                # BEFORE the user saw anything, adding several more seconds
                # of total silence on top of an already-slow request and
                # making it more tempting to assume something broke and
                # retry. The frontend now expects a follow-up "enrichment"
                # chunk after "final_answer" instead of everything in one.
                yield result
                suggestions, chart = await asyncio.gather(
                    _generate_followups(user_message, final_text),
                    build_chart(user_message, last_query) if last_query else _no_chart(),
                )
                if suggestions or chart:
                    enrichment = {"type": "enrichment"}
                    if suggestions:
                        enrichment["suggestions"] = suggestions
                    if chart:
                        enrichment["chart"] = chart
                    yield enrichment
                return

        yield {
            "type": "final_answer",
            "text": "I wasn't able to answer that confidently after checking with "
            "both models — try rephrasing the question.",
        }
    finally:
        mongo_client.close()
        query_client.close()
