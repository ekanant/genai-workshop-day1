import logging
import os
from llama_index import (
    StorageContext,
    load_index_from_storage,
)

from common.constants import STORAGE_DIR
from context_engine import create_service_context


def get_chat_engine():
    service_context = create_service_context()
    # check if storage already exists
    if not os.path.exists(STORAGE_DIR):
        raise Exception(
            "StorageContext is empty - call 'python app/engine/generate.py' to generate the storage first"
        )
    logger = logging.getLogger("uvicorn")
    # load the existing index
    logger.info(f"Loading index from {STORAGE_DIR}...")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context, service_context=service_context)
    logger.info(f"Finished loading index from {STORAGE_DIR}")
    return index.as_chat_engine()


def get_sub_question_query_engine():
    # TODO: 3 - Check if storage already exists, load the existing index that's created from Checkpoint 1
    # Two separate VectorStoreIndex should be created from food-recipes and drink-recipes respectively
    logger.info("Loading multiple indices..")
