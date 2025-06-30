# graph/cowriting_RAG_document_node.py

import logging
from state import AgentGraphState
from graph.utils import query_knowledge_base

logger = logging.getLogger(__name__)

async def cowriting_RAG_document_node(state: AgentGraphState) -> dict:
    """
    Queries the unified knowledge base to find the most relevant 'cowriting'
    interventions based on the student's written text and diagnosed issue.
    """
    logger.info("---Executing Co-writing RAG Node---")

    query_parts = [
        f"Learning Objective: {state.get('Learning_Objective_Focus', '')}",
        f"Student's written text: {state.get('Student_Written_Input_Chunk', '')}",
        f"Assessed issue with the text: {state.get('Immediate_Assessment_of_Input', '')}",
        f"Student's stated thought process: {state.get('Student_Articulated_Thought', '')}",
    ]
    query_string = " \n ".join(filter(None, query_parts)).strip()

    if not query_string:
        logger.warning("Co-writing RAG Node: Query string is empty. Skipping vector search.")
        return {"rag_document_data": []}

    retrieved_documents = await query_knowledge_base(
        query_string=query_string,
        category="cowriting"
    )

    return {"rag_document_data": retrieved_documents}