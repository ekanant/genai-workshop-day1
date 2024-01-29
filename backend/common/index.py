import logging
import os
from llama_index import (
    StorageContext,
    load_index_from_storage,
    ServiceContext
)
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine

from common.constants import STORAGE_DIR_DRINKS, STORAGE_DIR_FOODS
from common.context_engine import create_service_context

from common.constants import (
    STORAGE_DIR_DRINKS,
    STORAGE_DIR_FOODS,
)

logger = logging.getLogger("uvicorn")

service_context = create_service_context()

food_storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR_FOODS)
food_index = load_index_from_storage(storage_context=food_storage_context, service_context=service_context)

# build index and query engine
food_vector_query_engine = food_index.as_query_engine()

drink_storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR_DRINKS)
drink_index = load_index_from_storage(storage_context=drink_storage_context, service_context=service_context)

# build index and query engine
drink_vector_query_engine = drink_index.as_query_engine()

query_engine_tools = [
    QueryEngineTool(
        query_engine=food_vector_query_engine,
        metadata=ToolMetadata(
            name="food_engine",
            description="Provide data about food",
        ),
    ),
    QueryEngineTool(
        query_engine=drink_vector_query_engine,
        metadata=ToolMetadata(
            name="drink_engine",
            description="Provide data about drink",
        ),
    ),
]

s_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools = query_engine_tools,
    service_context=service_context,
    verbose=True,
    use_async=False)

def get_sub_question_query_engine():
    # TODO: 3 - Check if storage already exists, load the existing index that's created from Checkpoint 1
    # Two separate VectorStoreIndex should be created from food-recipes and drink-recipes respectively
    logger.info("Loading multiple indices..")
    return s_engine
