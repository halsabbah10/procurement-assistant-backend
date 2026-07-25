# backend/app/agent/tools.py
from langchain_core.tools import create_retriever_tool
from langchain_mongodb.agent_toolkit import MongoDBDatabase, MongoDBDatabaseToolkit
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_voyageai import VoyageAIEmbeddings

from app.core.config import get_settings

VECTOR_INDEX_NAME = "item_vector_index"


def build_structured_tools(llm) -> list:
    settings = get_settings()
    db = MongoDBDatabase.from_connection_string(settings.mongodb_uri, database=settings.mongodb_db_name)
    toolkit = MongoDBDatabaseToolkit(db=db, llm=llm)
    return toolkit.get_tools()


def build_semantic_search_tool():
    settings = get_settings()
    embeddings = VoyageAIEmbeddings(model="voyage-4", voyage_api_key=settings.voyage_api_key)
    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=settings.mongodb_uri,
        namespace=f"{settings.mongodb_db_name}.item_embeddings",
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
