import logging
from state import AgentGraphState
from memory import memory_stub # Imports the instance from memory/__init__.py

logger = logging.getLogger(__name__)

async def load_student_data_node(state: AgentGraphState) -> dict:
    """Loads student data from Mem0 and updates the state."""
    user_id = state["user_id"]
    logger.info(f"StudentModelNode: Loading student data for user_id: '{user_id}' from Mem0")
    
    # Get student data from memory stub
    student_data = memory_stub.get_student_data(user_id)
    logger.info(f"StudentModelNode: Retrieved student data from Mem0: {student_data}")
    
    return {"student_memory_context": student_data}

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
