# graph/scaffolding_RAG_document_node.py

import logging
from state import AgentGraphState
from graph.utils import query_knowledge_base

logger = logging.getLogger(__name__)

async def scaffolding_RAG_document_node(state: AgentGraphState) -> dict:
    """
    Queries the unified knowledge base to find the most relevant 'scaffolding'
    techniques based on the student's specific struggle point.
    """
    logger.info("---Executing Scaffolding RAG Node---")

    # The query for scaffolding should be highly focused on the specific problem
    query_parts = [
        f"Learning Task: {state.get('Learning_Objective_Task', '')}",
        f"Specific Struggle Point: {state.get('Specific_Struggle_Point', '')}",
        f"Student English Level: {state.get('English_Comfort_Level', '')}",
        f"Student Attitude: {state.get('Student_Attitude_Context', '')}",
    ]
    query_string = " \n ".join(filter(None, query_parts)).strip()

    if not query_string:
        logger.warning("Scaffolding RAG Node: Query string is empty. Skipping vector search.")
        return {"rag_document_data": []}

    retrieved_documents = await query_knowledge_base(
        query_string=query_string,
        category="scaffolding"
    )

    return {"rag_document_data": retrieved_documents}