import logging
import os
import chromadb
from chromadb.utils import embedding_functions
from state import AgentGraphState

logger = logging.getLogger(__name__)

# --- Configuration ---
DB_DIRECTORY = "chroma_db"
COLLECTION_NAME = "tutor_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Must match the model used for ingestion
TOP_K_RESULTS = 3 # Number of similar examples to retrieve

# Columns from the state to construct the query for embedding
QUERY_CONTEXT_COLUMNS = [
    "example_prompt_text",
    "student_goal_context",
    "student_confidence_context",
    "teacher_initial_impression",
    "student_struggle_context"
]

# --- ChromaDB Client Initialization ---
# Initialize the client once and reuse it.
# This assumes the DB is in the root of the `backend_ai_service_langgraph` directory.
client = None
try:
    client = chromadb.PersistentClient(path=DB_DIRECTORY)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
    logger.info(f"Successfully connected to ChromaDB collection '{COLLECTION_NAME}'.")
except Exception as e:
    logger.error(f"Failed to connect to ChromaDB. Ensure the database has been created by running the ingestion script. Error: {e}", exc_info=True)
    client = None # Ensure client is None if connection fails

async def modelling_RAG_document_node(state: AgentGraphState) -> dict:
    """
    Queries the ChromaDB vector store to find relevant modeling examples based on student context.
    """
    logger.info("---Executing RAG Node (Vector DB Version)---")

    if not client or not collection:
        error_msg = "ChromaDB client is not available. Cannot perform RAG."
        logger.error(error_msg)
        return {"modelling_document_data": [], "error": error_msg}

    # 1. Construct the query string from the state
    query_parts = [str(state.get(key, "")) for key in QUERY_CONTEXT_COLUMNS]
    query_string = " \n\n ".join(filter(None, query_parts)).strip()

    if not query_string:
        logger.warning("RAG Node: Query string is empty. Skipping vector search.")
        return {"modelling_document_data": [], "info": "Query for RAG was empty."}

    logger.info(f"RAG Node: Constructed query for vector search: '{query_string[:200]}...'")

    try:
        # 2. Query the ChromaDB collection
        # The embedding function handles embedding the query_string automatically.
        query_results = collection.query(
            query_texts=[query_string],
            n_results=TOP_K_RESULTS,
            # include=['metadatas', 'documents', 'distances'] # For debugging
        )

        # 3. Extract and format the results
        # The full original data is stored in the metadata.
        retrieved_documents = query_results.get('metadatas', [[]])[0]
        
        if not retrieved_documents:
            logger.info("RAG Node: No documents found in ChromaDB for the given query.")
        else:
            logger.info(f"RAG Node: Retrieved {len(retrieved_documents)} documents from ChromaDB.")

        return {"modelling_document_data": retrieved_documents}

    except Exception as e:
        logger.error(f"RAG Node: An error occurred during ChromaDB query: {e}", exc_info=True)
        return {"modelling_document_data": [], "error": f"Failed to query vector database: {e}"}

# Example usage (for local testing if needed)
async def main_test():
    # Mock state for testing
    class MockAgentGraphState(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    # Dummy data similar to what modelling_query_document_node would provide
    sample_docs = [
        {
            'Example_Prompt_Text': 'Describe your favorite city.', 
            'Student_Goal_Context': 'Improve speaking fluency for TOEFL.', 
            'Student_Confidence_Context': 'Somewhat confident, but nervous about complex sentences.', 
            'English_Comfort_Level': 'Conversational', 
            'Teacher_Initial_Impression': 'Good vocabulary, but struggles with structure.', 
            'Student_Struggle_Context': 'Organizing ideas coherently under pressure.',
            'pre_modeling_setup_script': 'Setup for city description.', 
            'modeling_and_think_aloud_sequence_json': '[{"type":"think_aloud", "content":"Let us start..."}]',
            # ... other fields
        },
        {
            'Example_Prompt_Text': 'Explain the importance of renewable energy.', 
            'Student_Goal_Context': 'Write a strong argumentative essay.', 
            'Student_Confidence_Context': 'Confident in writing, but wants to improve academic tone.', 
            'English_Comfort_Level': 'Advanced', 
            'Teacher_Initial_Impression': 'Excellent grammar and vocabulary.', 
            'Student_Struggle_Context': 'Using sophisticated transition words.',
            'pre_modeling_setup_script': 'Setup for renewable energy essay.', 
            'modeling_and_think_aloud_sequence_json': '[{"type":"think_aloud", "content":"The first point is..."}]',
            # ... other fields
        },
        {
            'Example_Prompt_Text': 'Describe your favorite book and explain why you like it.', 
            'Student_Goal_Context': 'Score 105+ for grad school.', 
            'Student_Confidence_Context': 'Fairly confident, polish writing.', 
            'English_Comfort_Level': 'Beginner', 
            'Teacher_Initial_Impression': 'Needs foundational support.', 
            'Student_Struggle_Context': 'Student has difficulty starting tasks.',
            'pre_modeling_setup_script': 'PREP structure for favorite book.', 
            'modeling_and_think_aloud_sequence_json': '[{"type":"think_aloud", "content":"My favorite book is..."}]',
            # ... other fields
        }
    ]

    # Test case 1: Query matching the third document
    state1 = MockAgentGraphState({
        "user_id": "test_user_rag_1",
        "modelling_document_data": sample_docs,
        "example_prompt_text": "Describe your favorite book and explain why you like it so much.",
        "student_goal_context": "My main goal is to score 105+ for grad school, focusing on speaking and writing.",
        "student_confidence_context": "I'm fairly confident, but want to polish writing and make speaking more natural under pressure.",
        "teacher_initial_impression": "Needs Foundational Support in this Skill Area",
        "student_struggle_context": "Student has difficulty starting tasks."
    })
    
    print("\n--- Test Case 1: Matching 'favorite book' query ---")
    result1 = await modelling_RAG_document_node(state1)
    print(f"RAG Result 1: Found {len(result1.get('modelling_document_data', []))} documents.")
    for doc in result1.get('modelling_document_data', []):
        print(f"  - Prompt: {doc.get('Example_Prompt_Text')}")
    if result1.get('error'): print(f"Error: {result1.get('error')}")
    if result1.get('warning'): print(f"Warning: {result1.get('warning')}")

    # Test case 2: Query matching the first document
    state2 = MockAgentGraphState({
        "user_id": "test_user_rag_2",
        "modelling_document_data": sample_docs,
        "example_prompt_text": "Tell me about your favorite city.", # Slightly different phrasing
        "student_goal_context": "Improve speaking fluency.",
        "student_confidence_context": "Nervous about complex sentences.",
        "teacher_initial_impression": "Good vocabulary, struggles with structure.",
        "student_struggle_context": "Organizing ideas coherently."
    })
    print("\n--- Test Case 2: Matching 'favorite city' query ---")
    result2 = await modelling_RAG_document_node(state2)
    print(f"RAG Result 2: Found {len(result2.get('modelling_document_data', []))} documents.")
    for doc in result2.get('modelling_document_data', []):
        print(f"  - Prompt: {doc.get('Example_Prompt_Text')}")
    if result2.get('error'): print(f"Error: {result2.get('error')}")
    if result2.get('warning'): print(f"Warning: {result2.get('warning')}")

    # Test case 3: Empty document list
    state3 = MockAgentGraphState({
        "user_id": "test_user_rag_3",
        "modelling_document_data": [], # Empty list
        "example_prompt_text": "Anything"
    })
    print("\n--- Test Case 3: Empty document list ---")
    result3 = await modelling_RAG_document_node(state3)
    print(f"RAG Result 3: Found {len(result3.get('modelling_document_data', []))} documents.")
    if result3.get('error'): print(f"Error: {result3.get('error')}")

    # Test case 4: Empty query string (all context fields empty)
    state4 = MockAgentGraphState({
        "user_id": "test_user_rag_4",
        "modelling_document_data": sample_docs,
        "example_prompt_text": "",
        "student_goal_context": "",
        "student_confidence_context": "",
        "teacher_initial_impression": "",
        "student_struggle_context": ""
    })
    print("\n--- Test Case 4: Empty query string ---")
    result4 = await modelling_RAG_document_node(state4)
    print(f"RAG Result 4: Found {len(result4.get('modelling_document_data', []))} documents.") # Should return all docs due to empty query
    if result4.get('warning'): print(f"Warning: {result4.get('warning')}")

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    # IMPORTANT: Set GOOGLE_API_KEY environment variable before running this test.
    # e.g., export GOOGLE_API_KEY='your_api_key_here'
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable is not set. RAG tests will fail.")
        print("Please set it before running: export GOOGLE_API_KEY='your_api_key'")
    else:
        asyncio.run(main_test())
