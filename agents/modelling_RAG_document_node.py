# agents/modelling_RAG_document_node.py

import logging
from state import AgentGraphState
from graph.utils import query_knowledge_base

logger = logging.getLogger(__name__)

async def modelling_RAG_document_node(state: AgentGraphState) -> dict:
    """
    Queries the knowledge base for the modelling flow, intelligently finding the lesson_id
    from either the top-level state or the initial nested context.
    """
    logger.info("---Executing Modelling RAG Node---")

    plan = state.get("pedagogical_plan")
    current_index = state.get("current_plan_step_index", 0)

    # Intelligently find the lesson_id from its two possible locations.
    lesson_id = state.get('lesson_id')
    if not lesson_id:
        lesson_id = state.get('current_context', {}).get('lesson_id')
    
    # Also get other context from the most reliable source (top-level if available)
    learning_objective = state.get('Learning_Objective_Focus') or state.get('current_context', {}).get('Learning_Objective_Focus', '')
    student_proficiency = state.get('STUDENT_PROFICIENCY') or state.get('current_context', {}).get('STUDENT_PROFICIENCY', '')

    current_step_focus = ""
    if plan and isinstance(plan, list) and 0 <= current_index < len(plan):
        current_step_focus = plan[current_index].get('focus', '')

    query_parts = [
        f"Topic/Focus for this step: {current_step_focus}",
        f"Overall Learning Objective: {learning_objective}",
        f"Student Proficiency: {student_proficiency}",
    ]
    query_string = " ".join(filter(None, query_parts)).strip()

    # The category for filtering IS the lesson_id
    category = lesson_id
    if not category:
        logger.warning("No lesson_id found in state or context. RAG query will be less specific.")
    else:
        logger.info(f"Querying knowledge base with category: {category}")

    retrieved_documents = await query_knowledge_base(
        query_string=query_string,
        category=category
    )

    # Return ONLY the new documents.
    return {"rag_document_data": retrieved_documents}
