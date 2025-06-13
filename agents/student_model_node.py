import logging
import json
from typing import Any, Dict

from state import AgentGraphState
from memory.mem0_client import shared_mem0_client

logger = logging.getLogger(__name__)

async def load_student_data_node(state: AgentGraphState) -> dict:
    """
    Loads student data from Mem0, extracts 'next_task_details' from the most recent
    interaction, and updates the state.
    """
    user_id = state["user_id"]
    logger.info(f"StudentModelNode: Loading student data for user_id: '{user_id}' from Mem0")

    # Get all memories from Mem0 for the user
    try:
        all_memories = shared_mem0_client.get_all(user_id=user_id)
        student_data: Dict[str, Any] = {"profile": {}, "interaction_history": []}

        for mem in all_memories:
            meta = mem.metadata or {}
            if meta.get('type') == 'profile':
                if isinstance(mem.data, dict):
                    student_data["profile"].update(mem.data)
            elif meta.get('type') == 'interaction':
                try:
                    interaction_dict = json.loads(mem.data)
                    student_data["interaction_history"].append(interaction_dict)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Could not parse interaction memory for user {user_id}: {mem.data}")

        # Sort history just in case, though mem0 usually returns latest first
        student_data["interaction_history"].sort(key=lambda x: x.get('timestamp', 0), reverse=True)

    except Exception as e:
        logger.error(f"Failed to retrieve or process data from Mem0 for user {user_id}: {e}", exc_info=True)
        student_data = {"profile": {}, "interaction_history": []}
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
    
    # Get key data from state
    transcript = state.get("transcript", "")
    output_content = state.get("output_content") or state.get("feedback_content", "")

    # Prepare interaction summary for logging
    interaction_data = {
        "transcript": transcript,
        "full_submitted_transcript": state.get("full_submitted_transcript"),
        "diagnosis": state.get("diagnosis_result"),
        "feedback": output_content.get("text_for_tts") if isinstance(output_content, dict) else output_content,
        "task_details": state.get("next_task_details")
    }
    
    logger.info(f"StudentModelNode: Saving interaction for user_id: '{user_id}' to Mem0")
    logger.debug(f"StudentModelNode: Interaction data: {interaction_data}")
    
    # Format data for Mem0, ensuring content is a string
    assistant_content = output_content.get("text_for_tts") if isinstance(output_content, dict) else str(output_content)
    messages_to_save = [
        {"role": "user", "content": transcript},
        {"role": "assistant", "content": assistant_content}
    ]

    # Save to Mem0
    try:
        shared_mem0_client.add(
            messages=messages_to_save,
            user_id=user_id,
            metadata={'type': 'interaction'}
        )
        logger.info(f"Successfully saved interaction for user_id: '{user_id}' to Mem0.")
    except Exception as e:
        logger.error(f"Failed to save interaction to Mem0 for user_id: '{user_id}': {e}", exc_info=True)
    
    return {} # No direct state update needed
