# FINAL, ROBUST graph/utils.py

import logging
import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# --- Configuration ---
DB_DIRECTORY = "chroma_db"
COLLECTION_NAME = "tutor_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_RESULTS = 3

# --- Global variables to hold the client and collection ---
# We still want to reuse the connection if it's already established.
_client: Optional[chromadb.Client] = None
_collection: Optional[chromadb.Collection] = None

def get_chroma_collection() -> Optional[chromadb.Collection]:
    """
    Initializes and returns the ChromaDB collection.
    Uses a singleton pattern to avoid reconnecting on every call.
    This is more resilient than initializing at the top level.
    """
    global _client, _collection

    # If the connection already exists, just return it.
    if _collection is not None:
        return _collection

    # If it doesn't exist, try to create it.
    try:
        logger.info("Attempting to initialize ChromaDB connection...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_root, DB_DIRECTORY)
        
        if not os.path.exists(db_path):
            logger.error(f"ChromaDB directory not found at: {db_path}. Please run the ingestion script.")
            return None

        _client = chromadb.PersistentClient(path=db_path)
        
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        _collection = _client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=sentence_transformer_ef
        )
        logger.info(f"ChromaDB connection successful. Collection '{COLLECTION_NAME}' loaded.")
        return _collection

    except Exception as e:
        logger.error(f"Failed to initialize or get ChromaDB collection: {e}", exc_info=True)
        # Ensure they are reset to None on failure
        _client = None
        _collection = None
        return None

async def query_knowledge_base(query_string: str, category: str) -> List[Dict]:
    """
    A shared utility function to query the ChromaDB vector store.
    """
    # Get the collection using our new, resilient function.
    collection = get_chroma_collection()
    
    if not collection:
        logger.error("Cannot perform RAG because ChromaDB collection is not available.")
        return []

    try:
        logger.info(f"Querying KB for category '{category}' with query: '{query_string[:100]}...'")
        
        query_results = collection.query(
            query_texts=[query_string],
            n_results=TOP_K_RESULTS,
            where={"category": category}
        )
        
        retrieved_documents = query_results.get('metadatas', [[]])[0]
        
        logger.info(f"Query successful. Retrieved {len(retrieved_documents)} documents for category '{category}'.")
        return retrieved_documents

    except Exception as e:
        logger.error(f"An error occurred during knowledge base query for category '{category}': {e}", exc_info=True)
        return []