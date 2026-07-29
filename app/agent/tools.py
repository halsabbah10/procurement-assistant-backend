# backend/app/agent/tools.py
from typing import Any

from bson.json_util import dumps
from langchain_core.tools import BaseTool, create_retriever_tool
from langchain_mongodb.agent_toolkit import MongoDBDatabase, MongoDBDatabaseToolkit
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import BaseModel, ConfigDict, Field
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from app.core.config import get_settings
from app.db.safe_pipeline import MAX_QUERY_TIME_MS, InvalidQueryError, parse_agent_pipeline

VECTOR_INDEX_NAME = "item_vector_index"
# voyage-4 is Matryoshka-trained (supports 256/512/1024/2048 via
# output_dimension without re-embedding). Verified empirically on this
# dataset's 13,294 UNSPSC categories: 512-dim vs 1024-dim top-5 retrieval
# overlap averaged 4.4/5 across 8 representative queries, with the same
# top-1 hit in 6/8 and a semantically equivalent category in the other 2 —
# negligible quality loss for a filter-then-aggregate downstream use. In
# exchange, per-document size drops from ~13.6KB to ~7.0KB, taking the full
# collection from ~135MB to ~88MB, which is the difference between fitting
# in the Atlas M0 free tier's 512MB dataSize+indexSize quota (alongside the
# fixed ~380MB purchase_orders collection) and exceeding it.
EMBEDDING_DIMENSION = 512


class _SafeQueryInput(BaseModel):
    query: str = Field(..., description="A detailed and correct MongoDB query.")


class SafeMongoDBQueryTool(BaseTool):
    """Drop-in replacement for langchain_mongodb's QueryMongoDBDatabaseTool
    (same name/description, so this app's SYSTEM_PROMPT and the model's
    existing tool-use behavior need no changes). Executes through
    app/db/safe_pipeline.py's eval()-free parser instead of
    MongoDBDatabase.run_no_throw, which parses the pipeline with a bare
    eval() that (see that module's docstring) is not actually sandboxed.
    The `query` argument here is LLM output steered by untrusted chat text
    — ordinary prompt injection can reach it — not a trusted internal
    value, so eval() on it is a real RCE surface, not a theoretical one."""

    name: str = "mongodb_query"
    description: str = (
        "Execute a MongoDB query against the database and get back the result. "
        "If the query is not correct, an error message will be returned. "
        "If an error is returned, rewrite the query, check the query, and try again."
    )
    args_schema: type[BaseModel] = _SafeQueryInput
    client: MongoClient = Field(exclude=True)
    database_name: str = Field(exclude=True)
    allowed_collections: set[str] = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, query: str, **kwargs: Any) -> str:
        try:
            collection, pipeline = parse_agent_pipeline(query, self.allowed_collections)
        except InvalidQueryError as exc:
            return f"Error: {exc}"
        try:
            db = self.client[self.database_name]
            result = list(db[collection].aggregate(pipeline, maxTimeMS=MAX_QUERY_TIME_MS))
            return dumps(result, indent=2)
        except PyMongoError as exc:
            return f"Error: {exc}"


def build_mongo_resources(settings) -> tuple[MongoClient, MongoDBDatabase]:
    """One MongoClient per request, shared by the checkpointer, the toolkit's
    schema/list/checker tools, the safe query tool, and the vector store —
    replaces what used to be up to 3 separate fresh connections opened per
    model attempt (Sonnet, then again on Opus escalation), each paying its
    own handshake/topology-discovery cost and its own connection pool
    against an Atlas M0 cluster with a real, low connection cap."""
    client: MongoClient = MongoClient(settings.mongodb_uri)
    db = MongoDBDatabase(client, database=settings.mongodb_db_name)
    return client, db


def build_structured_tools(
    llm, client: MongoClient, database_name: str, db: MongoDBDatabase
) -> list:
    toolkit = MongoDBDatabaseToolkit(db=db, llm=llm)
    # Keep the toolkit's schema/list/checker tools (mongodb_schema,
    # mongodb_list_collections, mongodb_query_checker) — none of them call
    # MongoDBDatabase._parse_command/eval(). Drop its query tool only.
    tools = [t for t in toolkit.get_tools() if t.name != "mongodb_query"]
    safe_query_tool = SafeMongoDBQueryTool(
        client=client,
        database_name=database_name,
        allowed_collections=set(db.get_usable_collection_names()),
    )
    return [safe_query_tool, *tools]


def build_semantic_search_tool(client: MongoClient, database_name: str):
    settings = get_settings()
    embeddings = VoyageAIEmbeddings(
        model="voyage-4",
        voyage_api_key=settings.voyage_api_key,
        output_dimension=EMBEDDING_DIMENSION,
    )
    collection = client[database_name]["item_embeddings"]
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=VECTOR_INDEX_NAME,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    return create_retriever_tool(
        retriever,
        "semantic_search",
        "Find UNSPSC purchasing categories (segment/family/class/commodity "
        "titles) that are semantically related to a natural-language "
        "description, when the exact wording won't match the data verbatim "
        "(e.g. 'cybersecurity' -> 'Network security software'). Returns "
        "matching commodity_title/class_title/family_title/segment_title "
        "values you can then filter a structured aggregation on.",
    )
