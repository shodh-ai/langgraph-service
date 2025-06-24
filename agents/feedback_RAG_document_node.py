# graph/feedback_RAG_document_node.py

import logging
from state import AgentGraphState
from graph.utils import query_knowledge_base

logger = logging.getLogger(__name__)

async def feedback_RAG_document_node(state: AgentGraphState) -> dict:
    """
    Queries the unified knowledge base to find the most relevant 'feedback'
    strategies based on the student's error and affective state.
    """
    logger.info("---Executing Feedback RAG Node---")

    # Query for feedback should focus on the error and the student's emotional state
    query_parts = [
        f"Student Error Type: {state.get('diagnosed_error_type', '')}",
        f"Student Affective State: {state.get('Student_Affective_State', '')}",
        f"Student Proficiency: {state.get('Student_Comfort_Level', '')}",
        f"Learning Objective: {state.get('Learning_Objective_Focus', '')}",
    ]
    query_string = " \n ".join(filter(None, query_parts)).strip()

    if not query_string:
        logger.warning("Feedback RAG Node: Query string is empty. Skipping vector search.")
        return {"rag_document_data": []}

    retrieved_documents = await query_knowledge_base(
        query_string=query_string,
        category="feedback"
    )

    return {"rag_document_data": retrieved_documents}