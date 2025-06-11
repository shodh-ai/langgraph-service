import logging
from state import AgentGraphState
from memory import memory_stub # Imports the instance from memory/__init__.py

logger = logging.getLogger(__name__)

async def load_student_data_node(state: AgentGraphState) -> dict:
    """
    Loads student data from Mem0, extracts 'next_task_details' from the most recent
    interaction, and updates the state.
    """
    user_id = state["user_id"]
    logger.info(f"StudentModelNode: Loading student data for user_id: '{user_id}' from Mem0")

    # Get student data from memory stub
    student_data = memory_stub.get_student_data(user_id)
    logger.info(f"StudentModelNode: Retrieved student data from Mem0: {student_data}")

    # Initialize updates with the full student memory context
    updates = {"student_memory_context": student_data}

    # Extract 'next_task_details' from the last interaction, if available
    interaction_history = (student_data or {}).get("interaction_history", [])
    if interaction_history:
        last_interaction = interaction_history[-1]
        next_task_details = last_interaction.get("task_details")
        if next_task_details:
            updates["next_task_details"] = next_task_details
            logger.info(f"StudentModelNode: Loaded 'next_task_details' from last interaction: {next_task_details}")
        else:
            logger.info("StudentModelNode: Last interaction found, but it has no 'task_details'.")
    else:
        logger.info("StudentModelNode: No interaction history found for user, so no 'next_task_details' to load.")

    return updates

async def save_interaction_node(state: AgentGraphState) -> dict:
    """Saves the current interaction to Mem0."""
    user_id = state["user_id"]
    
    # Get output content (in Phase 1, this might be using feedback_content for backward compatibility)
    output_content = state.get("output_content") or state.get("feedback_content", {})
    
    # Prepare interaction summary
    interaction_data = {
        "transcript": state.get("transcript"),
        "full_submitted_transcript": state.get("full_submitted_transcript"),
        "diagnosis": state.get("diagnosis_result"),
        "feedback": output_content.get("text_for_tts") if isinstance(output_content, dict) else None,
        "task_details": state.get("next_task_details")
    }
    
    logger.info(f"StudentModelNode: Saving interaction for user_id: '{user_id}' to Mem0")
    logger.debug(f"StudentModelNode: Interaction data: {interaction_data}")
    
    # Save to memory stub
    memory_stub.add_interaction(user_id, interaction_data)
    
    return {} # No direct state update needed
