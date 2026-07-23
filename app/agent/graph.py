# backend/app/agent/graph.py
"""Outer orchestrator: run the ReAct agent on Sonnet 5; if it fails to
produce a usable answer, escalate once to Opus 4.8 on the same thread."""
from __future__ import annotations

from typing import AsyncIterator

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.prebuilt import ToolNode, create_react_agent
from pymongo import MongoClient

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import build_semantic_search_tool, build_structured_tools
from app.core.config import get_settings

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
# run, for both the Sonnet and Opus attempts alike (verified: with 25 the
# same question reliably converges by step ~13-14). Raised to 25 to give
# the ReAct loop realistic headroom while still bounding runaway loops.
MAX_ITERATIONS = 25

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


def _build_agent(model_name: str, checkpointer):
    settings = get_settings()
    llm = ChatAnthropic(model=model_name, api_key=settings.anthropic_api_key, max_tokens=4096)
    tools = build_structured_tools(llm) + [build_semantic_search_tool()]
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
    tool_node = ToolNode(
        tools, handle_tool_errors=True, awrap_tool_call=_cap_oversized_tool_result
    )
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


def _is_out_of_scope(user_message: str) -> bool:
    """Cheap Haiku 4.5 guard so obviously irrelevant questions never reach
    the more expensive Sonnet/Opus path."""
    settings = get_settings()
    guard = ChatAnthropic(model=HAIKU_MODEL, api_key=settings.anthropic_api_key, max_tokens=10)
    response = guard.invoke(
        "Is this question about California state government procurement/purchase "
        f"order data? Answer only YES or NO.\n\nQuestion: {user_message}"
    )
    # Same `.text` vs `.content` normalization as below — `.content` can be a
    # list of content blocks rather than a plain string.
    return "NO" in response.text.upper()


async def run_agent(user_message: str, thread_id: str) -> AsyncIterator[dict]:
    if _is_out_of_scope(user_message):
        yield {
            "type": "final_answer",
            "text": "I can only answer questions about the California state "
            "purchase order dataset (2012-2015). Try asking about spending, "
            "departments, suppliers, or specific items.",
        }
        return

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
    try:
        checkpointer = MongoDBSaver(mongo_client)
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_ITERATIONS}

        for model_name, is_escalation in ((SONNET_MODEL, False), (OPUS_MODEL, True)):
            agent = _build_agent(model_name, checkpointer)
            final_text = None
            try:
                async for event in agent.astream(
                    {"messages": [("user", user_message)]}, config, stream_mode="values"
                ):
                    last_message = event["messages"][-1]
                    if getattr(last_message, "type", None) == "ai" and not getattr(
                        last_message, "tool_calls", None
                    ):
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
                        final_text = last_message.text
                        yield {"type": "step", "text": f"[{model_name}] {final_text[:80]}"}
            except Exception as exc:  # noqa: BLE001 — deliberately broad: any
                # failure on the Sonnet attempt should trigger escalation,
                # not crash the request.
                if is_escalation:
                    yield {
                        "type": "final_answer",
                        "text": f"I wasn't able to answer that confidently. ({exc})",
                    }
                    return
                continue

            if final_text:
                yield {"type": "final_answer", "text": final_text}
                return

        yield {
            "type": "final_answer",
            "text": "I wasn't able to answer that confidently after checking with "
            "both models — try rephrasing the question.",
        }
    finally:
        mongo_client.close()
