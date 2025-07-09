# agents/teaching_RAG_document_node.py (The Final, State-Preserving Version)

import logging
from state import AgentGraphState
from graph.utils import query_knowledge_base


logger = logging.getLogger(__name__)

async def teaching_RAG_document_node(state: AgentGraphState) -> dict:
    """
    Queries the knowledge base, intelligently finding the lesson_id from either
    the top-level state OR the initial nested context, and preserves all state.
    """
    logger.info("---Executing Robust Teaching RAG Node---")

    plan = state.get("pedagogical_plan")
    current_index = state.get("current_plan_step_index", 0)

    # --- THIS IS THE FINAL FIX ---
    # Intelligently find the lesson_id from its two possible locations.
    # 1. Try the top-level state first (for turns after the planner has run).
    lesson_id = state.get('lesson_id')
    # 2. If not found, check the initial nested context (for the first turn before the planner).
    if not lesson_id:
        lesson_id = state.get('current_context', {}).get('lesson_id')
    # --- END FINAL FIX ---
    
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
        # This warning will now only appear if lesson_id is truly missing from the payload
        logger.warning("No lesson_id found in state or context. RAG query will be less specific.")
    else:
        logger.info(f"Querying knowledge base with category: {category}")

    retrieved_documents = await query_knowledge_base(
        query_string=query_string,
        category=category # Pass the found lesson_id as the category
    )

    # Return ONLY the new documents.
    return {"rag_document_data": retrieved_documents}