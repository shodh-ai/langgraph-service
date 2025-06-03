import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def format_final_output_node(state: AgentGraphState) -> dict:
    """
    Combines greeting TTS and task suggestion TTS into a final response.
    Consolidates UI actions from previous nodes.
    
    Args:
        state: The current agent graph state
        
    Returns:
        Dict with the final 'output_content' for the client.
    """
    logger.info(f"OutputFormatterNode: Formatting final output for the client.")

    # Retrieve Greeting TTS (from handle_home_greeting_node via greeting_data)
    greeting_data = state.get("greeting_data", {})
    greeting_tts = greeting_data.get("greeting_tts", "")
    if greeting_tts:
        logger.info(f"OutputFormatterNode: Retrieved greeting_tts: '{greeting_tts[:100]}...')")

    # Retrieve Task Suggestion TTS (from determine_next_pedagogical_step_stub_node via task_suggestion_llm_output)
    task_suggestion_data = state.get("task_suggestion_llm_output", {})
    task_suggestion_tts = task_suggestion_data.get("task_suggestion_tts", "")
    if task_suggestion_tts:
        logger.info(f"OutputFormatterNode: Retrieved task_suggestion_tts: '{task_suggestion_tts[:100]}...')")

    # Combine TTS strings
    final_combined_tts = ""
    if greeting_tts:
        final_combined_tts += greeting_tts
    if task_suggestion_tts:
        if final_combined_tts: # Add a space if greeting_tts was also present
            final_combined_tts += " " 
        final_combined_tts += task_suggestion_tts
    logger.info(f"OutputFormatterNode: Final combined TTS: '{final_combined_tts[:100]}...')")
    
    if not final_combined_tts:
        logger.warning("OutputFormatterNode: No TTS strings found from previous nodes. Using default response.")
        final_combined_tts = "I'm ready to help with your practice!"

    # Retrieve UI Actions (set by determine_next_pedagogical_step_stub_node in output_content.ui_actions)
    # The output_content in the state at this point is the one from the last node that modified it (curriculum_navigator)
    current_output_content_from_state = state.get("output_content", {})
    if not current_output_content_from_state:
        logger.warning("OutputFormatterNode: No output_content found in state. Using default UI actions.")
        ui_actions = []
    else:
        ui_actions = current_output_content_from_state.get("ui_actions", [])
    if not isinstance(ui_actions, list):
        logger.warning(f"OutputFormatterNode: ui_actions was not a list, re-initializing. Value: {ui_actions}")
        ui_actions = []
    
    logger.info(f"OutputFormatterNode: Consolidating UI actions. Initial actions: {ui_actions}")
    # The test SHOW_ALERT action is removed by not adding it here.

    final_output_for_client = {
        "response": final_combined_tts,
        "ui_actions": ui_actions
    }
    
    logger.info(f"OutputFormatterNode: Final combined response: '{final_combined_tts[:200]}...')")
    logger.info(f"OutputFormatterNode: Final UI actions: {ui_actions}")

    # For backward compatibility, also update feedback_content
    # If other parts of the system rely on feedback_content, ensure it's consistent.
    # For this flow, output_content is the primary carrier of the final response.
    update_payload = {"output_content": final_output_for_client}
    if "feedback_content" not in state or state.get("feedback_content") is None:
        logger.info("OutputFormatterNode: Also setting feedback_content for backward compatibility.")
        update_payload["feedback_content"] = final_output_for_client
    
    return update_payload
