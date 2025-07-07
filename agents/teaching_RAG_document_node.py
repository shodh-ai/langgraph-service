# agents/teaching_RAG_document_node.py (The Final, State-Preserving Version)

import logging
from state import AgentGraphState
from graph.utils import query_knowledge_base


logger = logging.getLogger(__name__)

async def teaching_RAG_document_node(state: AgentGraphState) -> dict:
    """
    Queries the knowledge base for teaching documents AND preserves the critical
    plan and context state for all subsequent nodes in the graph.
    """
    logger.info("---Executing Teaching RAG Node (State-Preserving)---")

    # --- THIS IS THE FIX: Read all the state we need to preserve ---
    plan = state.get("pedagogical_plan")
    current_index = state.get("current_plan_step_index", 0)
    
    # --- Read from the reliable top-level state keys, not the nested context. ---
    learning_objective = state.get('Learning_Objective_Focus', '')
    student_proficiency = state.get('STUDENT_PROFICIENCY', '')
    lesson_id = state.get('lesson_id') # Read lesson_id from top level
    # --- END FIX ---

    current_step_focus = ""
    if plan and isinstance(plan, list) and 0 <= current_index < len(plan):
        current_step = plan[current_index]
        if isinstance(current_step, dict):
            current_step_focus = current_step.get('focus', '')
        logger.info(f"Current lesson step focus: {current_step_focus}")

    query_parts = [
        f"Topic/Focus for this step: {current_step_focus}",
        f"Overall Learning Objective: {learning_objective}",
        f"Student Proficiency: {student_proficiency}",
    ]
    query_string = " ".join(filter(None, query_parts)).strip()

    category = lesson_id
    if not category:
        logger.warning("No lesson_id found in state. Cannot perform a targeted RAG query.")
    else:
        logger.info(f"Querying knowledge base with category: {category}")

    retrieved_documents = await query_knowledge_base(
        query_string=query_string,
        category=category
    )

    return {
        "rag_document_data": retrieved_documents,
        "pedagogical_plan": state.get("pedagogical_plan"),
        "current_plan_step_index": state.get("current_plan_step_index"),
        "lesson_id": state.get("lesson_id"),
        "Learning_Objective_Focus": state.get("Learning_Objective_Focus"),
        "STUDENT_PROFICIENCY": state.get("STUDENT_PROFICIENCY"),
        "STUDENT_AFFECTIVE_STATE": state.get("STUDENT_AFFECTIVE_STATE"),
    }