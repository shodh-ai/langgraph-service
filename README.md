# Rox Backend AI Service (LangGraph)

This service contains the core agentic logic for Rox, built using LangGraph.

## Running the Application

To run the main web server, use the following command:

```bash
python app.py
```

The server will be available at `http://localhost:8080`.

## Initial Setup: Data Ingestion

Before running the application for the first time, you need to populate the local vector database with the modeling examples. This is a one-time setup process.

Run the following command from the root of the `backend_ai_service_langgraph` directory:

```bash
python ingest_modelling_data.py
```

This script will:
1. Read the data from `modelling_data.csv`.
2. Generate embeddings for the relevant text fields.
3. Store the embeddings and associated metadata in a local ChromaDB instance located at `./chroma_db`.

Once the script completes, you can start the main application.
