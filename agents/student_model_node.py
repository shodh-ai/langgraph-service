import logging
from state import AgentGraphState
from memory import mem0_memory # Imports the instance from memory/__init__.py

logger = logging.getLogger(__name__)

async def load_student_context_node(state: AgentGraphState) -> dict:
    user_id = state["user_id"]
    logger.info(f"StudentModelNode: Loading context for user_id: '{user_id}'")
    # In a real async setup, memory_stub.get_student_data would be async
    # For the stub, we call its synchronous method.
    student_data = mem0_memory.get_student_data(user_id) 
    return {"student_memory_context": student_data}

async def save_interaction_summary_node(state: AgentGraphState) -> dict:
    user_id = state["user_id"]
    feedback_content_value = state.get("feedback_content")
    summary_data = {
        "transcript": state.get("transcript"),
        "diagnosis": state.get("diagnosis_result"),
        "feedback": feedback_content_value.get("text") if isinstance(feedback_content_value, dict) else None
    }
    logger.info(f"StudentModelNode: Saving interaction for user_id: '{user_id}', Data: {summary_data}")
    # In a real async setup, memory_stub.add_interaction_to_history would be async
    mem0_memory.add_interaction_to_history(user_id, summary_data)
    return {} # No direct state update needed from this, or could return a success flag
