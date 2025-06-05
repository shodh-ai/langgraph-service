import json
from state import AgentGraphState
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def semantic_search_by_diagnose(
    data_entries: List[Dict[str, Any]], query: str, top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform semantic search on the data entries using Google Generative AI embeddings via LangChain.

    Args:
        data_entries: List of data entries (dictionaries)
        query: Search query string
        top_k: Number of top results to return (default: 10)

    Returns:
        List of top_k data entries that best match the query semantically
    """
    # Initialize the Google Generative AI embeddings model
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Generate embedding for the query
        query_embedding = embedding_model.embed_query(query)
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []

    # Generate embeddings for each entry's diagnose field
    entry_embeddings = []
    for idx, entry in enumerate(data_entries):
        try:
            if "Diagnose" in entry and entry["Diagnose"]:
                diagnose_embedding = embedding_model.embed_query(entry["Diagnose"])
                entry_embeddings.append((idx, diagnose_embedding))
            else:
                print(f"Entry {idx} missing 'Diagnose' field or it's empty")
        except Exception as e:
            print(f"Error generating embedding for entry {idx}: {e}")

    # Calculate similarity scores
    similarities = []
    for idx, entry_embedding in entry_embeddings:
        # Cosine similarity calculation
        similarity = np.dot(query_embedding, entry_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(entry_embedding)
        )
        similarities.append((idx, similarity))

    # Sort by similarity (highest first) and get top_k results
    top_indices = [
        idx for idx, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    ]

    # Return the top_k entries with all fields intact
    results = [data_entries[idx] for idx in top_indices]
    return results


async def RAG_document_node(state: AgentGraphState) -> dict:
    logger.info(
        f"RAGDocumentNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )

    document_data = state.get("document_data", [])
    query = state.get("explanation", "")
    logger.info(f"RAGDocumentNode: Query: {query}")

    results = semantic_search_by_diagnose(document_data, query)
    logger.info(f"RAGDocumentNode: Found Results")

    return {"document_data": results}
