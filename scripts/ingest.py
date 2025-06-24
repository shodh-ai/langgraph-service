# scripts/ingest.py

import chromadb
from chromadb.utils import embedding_functions
import json
import os
import logging
import time

# --- Configuration ---
DB_DIRECTORY = "chroma_db"  # Directory to store the persistent database
COLLECTION_NAME = "tutor_knowledge_base" # A more descriptive name
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # This MUST match the model you'll use in the RAG node
BATCH_SIZE = 500  # Process 500 documents at a time to manage memory and network traffic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest_data_to_chroma():
    """
    Reads the unified knowledge base, generates embeddings, and ingests
    the data into a persistent ChromaDB collection in batches.
    """
    # --- 1. Define Paths ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jsonl_path = os.path.join(project_root, 'data', 'unified', 'unified_knowledge_base.jsonl')
    db_path = os.path.join(project_root, DB_DIRECTORY)

    if not os.path.exists(jsonl_path):
        logging.error(f"Unified knowledge base file not found at: {jsonl_path}")
        return

    # --- 2. Initialize ChromaDB Client and Embedding Function ---
    logging.info(f"Initializing ChromaDB client at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)

    # The embedding function will run locally on your machine to generate the vectors.
    # It will download the model the first time it's used.
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # --- 3. Create or Get the Collection ---
    # Using get_or_create is safer than just create. It won't error if the collection
    # already exists, allowing you to re-run the script to add new data.
    logging.info(f"Accessing collection '{COLLECTION_NAME}'...")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"} # Using cosine distance is standard for semantic search
    )
    
    # --- 4. Read the Data from the JSONL File ---
    try:
        with open(jsonl_path, 'r') as f:
            all_records = [json.loads(line) for line in f]
        logging.info(f"Successfully loaded {len(all_records)} records from {jsonl_path}")
    except Exception as e:
        logging.error(f"Failed to read or parse JSONL file. Error: {e}", exc_info=True)
        return

    # --- 5. Process and Ingest in Batches ---
    total_records = len(all_records)
    for i in range(0, total_records, BATCH_SIZE):
        batch_start_time = time.time()
        # Get the slice of records for the current batch
        batch_records = all_records[i:i + BATCH_SIZE]
        
        start_index = i
        end_index = i + len(batch_records)
        
        logging.info(f"Processing batch {start_index+1}-{end_index} of {total_records}...")

        # Prepare data for this specific batch
        documents_to_embed = [rec.get('document_for_embedding', '') for rec in batch_records]
        metadatas_to_store = [rec.get('metadata', {}) for rec in batch_records]
        # Create unique IDs for each record
        ids_for_records = [f"record_{start_index + j}" for j in range(len(batch_records))]

        try:
            # The .add method will automatically use the collection's embedding function
            # to convert the 'documents' into vectors. This is the slow part.
            collection.add(
                documents=documents_to_embed,
                metadatas=metadatas_to_store,
                ids=ids_for_records
            )
            
            batch_end_time = time.time()
            duration = batch_end_time - batch_start_time
            logging.info(f"Successfully ingested batch {start_index+1}-{end_index}. Time taken: {duration:.2f} seconds.")

        except Exception as e:
            logging.error(f"Failed to ingest batch {start_index+1}-{end_index}. Error: {e}", exc_info=True)
            # You might want to add logic here to retry the batch or save failed batches to a file.
            # For now, we'll just log the error and continue to the next batch.
            continue
            
    logging.info("--- Ingestion process complete. ---")
    # You can verify the number of items in the collection
    count = collection.count()
    logging.info(f"Total items in collection '{COLLECTION_NAME}': {count}")


if __name__ == "__main__":
    ingest_data_to_chroma()