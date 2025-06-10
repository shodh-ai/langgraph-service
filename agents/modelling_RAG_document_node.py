import logging
import os
import google.generativeai as genai
import numpy as np
import pandas as pd # Used for DataFrame conversion for easy embedding
from state import AgentGraphState

logger = logging.getLogger(__name__)

# Columns to be used for constructing the query and the searchable documents
RAG_INPUT_COLUMNS = [
    'Example_Prompt_Text', 
    'Student_Goal_Context', 
    'Student_Confidence_Context', 
    'Teacher_Initial_Impression', 
    'Student_Struggle_Context'
]

async def modelling_RAG_document_node(state: AgentGraphState) -> dict:
    """
    Performs RAG on the filtered modelling_document_data using student context as a query.
    """
    logger.info(
        f"ModellingRAGDocumentNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )

    # For debugging state:
    logger.info(f"ModellingRAGDocumentNode: State keys available: {list(state.keys())}")
    raw_modelling_data = state.get("modelling_document_data") # Get without default first
    logger.info(f"ModellingRAGDocumentNode: Raw value of 'modelling_document_data' from state.get (no default): {raw_modelling_data}")
    logger.info(f"ModellingRAGDocumentNode: Type of raw 'modelling_document_data': {type(raw_modelling_data)}")

    if raw_modelling_data is None:
        if "modelling_document_data" not in state:
            logger.warning("ModellingRAGDocumentNode: Key 'modelling_document_data' NOT FOUND in state keys. Skipping RAG.")
        else:
            # Key is present, but its value is None
            logger.warning("ModellingRAGDocumentNode: Key 'modelling_document_data' IS PRESENT in state but its value is None. Skipping RAG.")
        return {"modelling_document_data": [], "error": "No documents provided for RAG (key missing or None)"}
    
    document_data_list = raw_modelling_data 
    
    if not document_data_list: # This means raw_modelling_data was an empty list []
        logger.info("ModellingRAGDocumentNode: Key 'modelling_document_data' found, but the list of documents is empty. Skipping RAG as no documents to process.")
        # Return an empty list and an informational message. The generator node should handle empty RAG results.
        return {"modelling_document_data": [], "info": "No documents after filtering, RAG skipped."}

    # Retrieve student context fields from state to form the query
    # These fields would be populated by an earlier node in the modelling flow (e.g., an input node)
    query_example_prompt = state.get("example_prompt_text", "")
    query_student_goal = state.get("student_goal_context", "")
    query_student_confidence = state.get("student_confidence_context", "")
    query_teacher_impression = state.get("teacher_initial_impression", "")
    query_student_struggle = state.get("student_struggle_context", "")

    # Construct the query string
    query_parts = [
        str(query_example_prompt),
        str(query_student_goal),
        str(query_student_confidence),
        str(query_teacher_impression),
        str(query_student_struggle)
    ]
    query_string = " \n\n ".join(filter(None, query_parts)).strip()

    if not query_string:
        logger.warning("ModellingRAGDocumentNode: Query string is empty. Skipping RAG.")
        # Return all documents if query is empty, or handle as an error
        return {"modelling_document_data": document_data_list, "warning": "Query for RAG was empty"}

    logger.info(f"ModellingRAGDocumentNode: Constructed query for RAG: '{query_string[:200]}...'" )

    # Configure Google Generative AI
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("ModellingRAGDocumentNode: GOOGLE_API_KEY environment variable is not set.")
        return {"modelling_document_data": [], "error": "GOOGLE_API_KEY not set"}
    genai.configure(api_key=api_key)

    try:
        # Prepare texts for embedding from the document_data_list
        texts_to_embed = []
        for doc in document_data_list:
            doc_parts = [str(doc.get(col, "")) for col in RAG_INPUT_COLUMNS]
            texts_to_embed.append(" \n\n ".join(filter(None, doc_parts)).strip())
        
        if not texts_to_embed:
            logger.warning("ModellingRAGDocumentNode: No text content found in documents to embed.")
            return {"modelling_document_data": [], "error": "No text content in documents for RAG"}

        # Generate embeddings for the documents and the query
        # Using 'text-embedding-004' or a similar model suitable for retrieval
        logger.info(f"ModellingRAGDocumentNode: Generating embeddings for {len(texts_to_embed)} documents and 1 query.")
        result_query = genai.embed_content(model="models/text-embedding-004", content=query_string, task_type="RETRIEVAL_QUERY")
        result_docs = genai.embed_content(model="models/text-embedding-004", content=texts_to_embed, task_type="RETRIEVAL_DOCUMENT")
        
        query_embedding = np.array(result_query['embedding'])
        doc_embeddings = np.array(result_docs['embedding']) # result_docs['embedding'] is already the list of embedding vectors
        logger.info(f"ModellingRAGDocumentNode: Embeddings generated. Query shape: {query_embedding.shape}, Docs shape: {doc_embeddings.shape}")

        # Calculate cosine similarity
        # Similarities: dot product of normalized vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        similarities = np.dot(doc_norms, query_norm)

        # Get top N results (e.g., top 3)
        top_n = min(3, len(document_data_list))
        top_indices = np.argsort(similarities)[-top_n:][::-1] # Sort descending, get top N indices

        ranked_documents = [document_data_list[i] for i in top_indices]
        
        # Log the similarity scores for the top documents for debugging/evaluation
        for i, idx in enumerate(top_indices):
            logger.info(f"ModellingRAGDocumentNode: Top {i+1} match (Index: {idx}) - Similarity: {similarities[idx]:.4f} - Prompt: {document_data_list[idx].get('Example_Prompt_Text', '')[:100]}...")

        logger.info(f"ModellingRAGDocumentNode: RAG complete. Returning {len(ranked_documents)} documents.")
        return {"modelling_document_data": ranked_documents}

    except Exception as e:
        logger.error(f"ModellingRAGDocumentNode: Error during RAG processing: {e}", exc_info=True)
        # Fallback to returning all documents or an empty list in case of error
        return {"modelling_document_data": document_data_list, "error": f"Error in RAG: {str(e)}"}


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
