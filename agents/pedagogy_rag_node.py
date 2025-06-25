# langgraph-service/agents/pedagogy_rag_node.py
import logging
from state import AgentGraphState
from graph.utils import query_knowledge_base

logger = logging.getLogger(__name__)

async def pedagogy_rag_node(state: AgentGraphState) -> dict:
    """
    Queries the knowledge base for pedagogical strategies based on the student's
    current learning context and recent performance.
    """
    logger.info("---PEDAGOGY RAG NODE---")

    # Construct a query based on the student's state
    query_parts = [
        f"Current learning objective: {state.get('Learning_Objective_Focus', 'Not specified')}",
        f"Recent student transcript: {state.get('transcript', '')}",
        f"Current student model summary: {state.get('student_model', {}).get('summary', 'No summary available.')}"
    ]
    query_string = " \n ".join(filter(None, query_parts)).strip()

    if not query_string or query_string.isspace():
        logger.warning("Pedagogy RAG Node: Query string is empty. Skipping vector search.")
        return {"pedagogy_rag_results": []}

    logger.info(f"Querying knowledge base for pedagogy with: {query_string[:100]}...")

    # Query the knowledge base, filtering by the 'pedagogy' category
    retrieved_documents = await query_knowledge_base(
        query_string=query_string,
        category="pedagogy"
    )

    logger.info(f"Retrieved {len(retrieved_documents)} documents from pedagogy knowledge base.")

    return {"rag_document_data": retrieved_documents}
