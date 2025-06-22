import pandas as pd
import chromadb
import json
import os
from tqdm import tqdm

# --- Configuration ---
CSV_FILE_PATH = "modelling_data.csv"
DB_DIRECTORY = "chroma_db"
COLLECTION_NAME = "modeling_examples"
# Using a smaller, efficient model for local embedding generation
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
BATCH_SIZE = 100 # Process N rows at a time

# Columns to combine for the document that will be embedded
TEXT_COLUMNS_TO_EMBED = [
    "Example_Prompt_Text",
    "Student_Goal_Context",
    "Student_Confidence_Context",
    "English_Comfort_Level",
    "Teacher_Initial_Impression",
    "Student_Struggle_Context"
]

def main():
    """
    Main function to read data from a CSV, generate embeddings, and store them in ChromaDB.
    """
    # --- 1. Setup ChromaDB Client ---
    print(f"Setting up ChromaDB client in directory: {DB_DIRECTORY}")
    # Create the DB directory if it doesn't exist
    if not os.path.exists(DB_DIRECTORY):
        os.makedirs(DB_DIRECTORY)
        
    client = chromadb.PersistentClient(path=DB_DIRECTORY)
    
    # Using the SentenceTransformerEmbeddingFunction to handle embedding generation automatically
    from chromadb.utils import embedding_functions
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    print(f"Getting or creating collection: {COLLECTION_NAME}")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"} # Using cosine distance for similarity is often good for text
    )

    # --- 2. Read and Process CSV data in chunks ---
    print(f"Reading data from {CSV_FILE_PATH}...")
    
    try:
        # Use chunksize to process the large file in batches
        chunk_iterator = pd.read_csv(CSV_FILE_PATH, chunksize=BATCH_SIZE, on_bad_lines='warn')
    except FileNotFoundError:
        print(f"Error: The file {CSV_FILE_PATH} was not found.")
        return

    total_rows_processed = 0
    
    for i, chunk in enumerate(tqdm(chunk_iterator, desc="Processing and ingesting chunks")):
        # --- 3. Prepare Documents, Metadata, and IDs for the current chunk ---
        
        # Fill NaN values with empty strings to avoid errors during concatenation
        chunk.fillna('', inplace=True)
        
        # Combine relevant text columns into a single document for embedding
        documents = chunk[TEXT_COLUMNS_TO_EMBED].apply(lambda row: "\n".join(row.values.astype(str)), axis=1).tolist()
        
        # Create metadata for each document. ChromaDB metadata values must be str, int, float, or bool.
        # We will convert the entire row to a dictionary and ensure all values are storable.
        metadatas = []
        for _, row in chunk.iterrows():
            # Convert row to dict, then convert all values to string to be safe for ChromaDB
            row_dict = {k: str(v) for k, v in row.to_dict().items()}
            metadatas.append(row_dict)
            
        # Generate unique IDs for each document in the chunk
        # Using chunk index and row index within the chunk to ensure uniqueness
        ids = [f"row_{i * BATCH_SIZE + j}" for j in range(len(chunk))]

        # --- 4. Upsert the data into ChromaDB ---
        # The embedding function handles the embedding generation automatically.
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        total_rows_processed += len(chunk)

    print("\n--- Ingestion Complete ---")
    print(f"Total documents processed: {total_rows_processed}")
    print(f"Total documents in collection '{COLLECTION_NAME}': {collection.count()}")
    print(f"ChromaDB database is stored in the '{DB_DIRECTORY}' directory.")

if __name__ == "__main__":
    main()
