import logging
from typing import Dict, Any
from state import AgentGraphState
from memory import memory_stub # Import the shared instance # Assuming Mem0Memory is accessible
import json

logger = logging.getLogger(__name__)

# Use the shared memory_stub instance directly.
# If memory_stub failed to initialize in memory/__init__.py,
# the application would likely have issues before this module is heavily used.
mem0_memory_client = memory_stub

async def finalize_session_in_mem0_node(state: AgentGraphState) -> Dict[str, Any]:
    user_id = state.get("user_id")
    session_is_ending = state.get("session_is_ending", False)
    final_data = state.get("final_session_data_to_save")

    logger.info(f"Finalize Session in Mem0 Node activated for user {user_id}. Session ending: {session_is_ending}")

    if not session_is_ending:
        logger.info(f"Session not marked as ending for user {user_id}. Skipping finalization.")
        return {}

    if not user_id:
        logger.error("User ID is missing in state. Cannot finalize session in Mem0.")
        return {} # Or handle error appropriately

    if not final_data or not isinstance(final_data, dict):
        logger.warning(f"'final_session_data_to_save' is missing or invalid for user {user_id}. Cannot finalize.")
        return {}

    if not mem0_memory_client:
        logger.error("Mem0Memory client not initialized. Cannot save session summary to Mem0.")
        # Potentially store this data in a fallback or log for manual processing
        return {"error_saving_session_summary": "Mem0 client unavailable"}

    try:
        logger.info(f"Attempting to save session summary to Mem0 for user {user_id}: {final_data}")
        # Use the add method, similar to add_interaction_to_history, but with a different type
        # Serialize the final_data dictionary to a JSON string
        final_data_json_string = json.dumps(final_data)

        # Create the messages payload as List[Dict[str, str]]
        # Each dictionary should contain "role" and "content" keys for Mem0 processing.
        messages_payload = [{
            "role": "user",  # Storing session summary, attributing to the user context
            "content": final_data_json_string
        }]

        mem0_memory_client.mem0_instance.add(
            messages_payload,  # Pass the structured payload as the first argument
            user_id=user_id,
            metadata={'type': 'session_summary', mem0_memory_client.user_id_field: user_id}
        )
        logger.info(f"Successfully saved session summary to Mem0 for user {user_id}.")
        
        # Optionally, clear the final_session_data_to_save from state after saving
        return {"final_session_data_to_save": None} 

    except Exception as e:
        logger.error(f"Error saving session summary to Mem0 for user {user_id}: {e}", exc_info=True)
        # Decide on error handling: re-raise, return error state, or log and continue
        return {"error_saving_session_summary": str(e)}

    # This node primarily performs a backend operation and doesn't directly alter output_content for the client.
    # It passes through other state elements.
    return {}
