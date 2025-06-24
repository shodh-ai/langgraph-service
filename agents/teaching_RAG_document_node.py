# graph/teaching_RAG_document_node.py

import logging
from state import AgentGraphState
from graph.utils import query_knowledge_base # We will create a shared utility for the DB query

logger = logging.getLogger(__name__)

async def teaching_RAG_document_node(state: AgentGraphState) -> dict:
    """
    Queries the unified knowledge base to find the most relevant 'teaching'
    documents based on the current student context and learning objective.
    """
    logger.info("---Executing Teaching RAG Node---")

    # Define which parts of the state form the query for 'teaching'
    # This might be different from other flows. For teaching, the objective is key.
    query_parts = [
        f"Learning Objective: {state.get('Learning_Objective_Focus', '')}",
        f"Student Proficiency: {state.get('STUDENT_PROFICIENCY', '')}",
        f"Student Affective State: {state.get('STUDENT_AFFECTIVE_STATE', '')}",
        f"Key concepts to explain: {state.get('LESSON_FOR_STUDENT', '')}",
    ]
    query_string = " \n ".join(filter(None, query_parts)).strip()

    if not query_string:
        logger.warning("Teaching RAG Node: Query string is empty. Skipping vector search.")
        return {"rag_document_data": []}

    # Use the shared utility to query the database with the specific category
    retrieved_documents = await query_knowledge_base(
        query_string=query_string,
        category="teaching"
    )

    return {"rag_document_data": retrieved_documents}