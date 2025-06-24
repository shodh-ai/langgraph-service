# graph/utils.py

import logging
import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Dict

logger = logging.getLogger(__name__)

# --- Configuration ---
DB_DIRECTORY = "chroma_db"
COLLECTION_NAME = "tutor_knowledge_base" # Use the new, unified collection name
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_RESULTS = 3

# --- ChromaDB Client Initialization (Singleton Pattern) ---
# We initialize it once and reuse it across all RAG nodes that import this utility.
client = None
collection = None
try:
    # Construct the path relative to this file's location
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, DB_DIRECTORY)
    
    client = chromadb.PersistentClient(path=db_path)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
    logger.info(f"Shared ChromaDB client successfully connected to collection '{COLLECTION_NAME}'.")
except Exception as e:
    logger.error(f"Failed to initialize shared ChromaDB client. RAG nodes will fail. Error: {e}", exc_info=True)
    client = None
    collection = None

async def query_knowledge_base(query_string: str, category: str) -> List[Dict]:
    """
    A shared utility function to query the ChromaDB vector store.

    Args:
        query_string: The text to search for.
        category: The specific category to filter by in the metadata.

    Returns:
        A list of retrieved document metadatas.
    """
    if not collection:
        error_msg = "ChromaDB connection not available. Cannot perform RAG."
        logger.error(error_msg)
        return []

    try:
        logger.info(f"Querying KB for category '{category}' with query: '{query_string[:100]}...'")
        
        query_results = collection.query(
            query_texts=[query_string],
            n_results=TOP_K_RESULTS,
            where={"category": category} # The dynamic filter
        )

        # The full original data is stored in the metadata.
        retrieved_documents = query_results.get('metadatas', [[]])[0]
        
        logger.info(f"Query successful. Retrieved {len(retrieved_documents)} documents for category '{category}'.")
        return retrieved_documents

    except Exception as e:
        logger.error(f"An error occurred during knowledge base query for category '{category}': {e}", exc_info=True)
        return []