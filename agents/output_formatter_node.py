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
    # greeting_tts = greeting_data.get("greeting_tts", "") if greeting_data else ""
    # logger.info(f"OutputFormatterNode: Retrieved greeting_tts: '{greeting_tts[:100]}...')")

    # Retrieve Task Suggestion TTS (from determine_next_pedagogical_step_stub_node via task_suggestion_llm_output)
    # task_suggestion_data = state.get("task_suggestion_llm_output", {})
    # task_suggestion_tts = task_suggestion_data.get("task_suggestion_tts", "") if task_suggestion_data else ""
    # logger.info(f"OutputFormatterNode: Retrieved task_suggestion_tts: '{task_suggestion_tts[:100]}...')")

    # Retrieve Navigation TTS (from prepare_navigation_node)
    # navigation_tts = state.get("navigation_tts", "")
    # if navigation_tts:
    #     logger.info(f"OutputFormatterNode: Retrieved navigation_tts: '{navigation_tts[:100]}...')")

    # Retrieve Conversational TTS (from conversation_handler_node, etc.)
    raw_conversational_tts = state.get("conversational_tts")
    logger.info(f"OutputFormatterNode: Raw 'conversational_tts' from state: '{raw_conversational_tts}' (Type: {type(raw_conversational_tts)}) ")

    # Attempt to get TTS from direct key, then from output_content
    if raw_conversational_tts and str(raw_conversational_tts).strip():
        conversational_tts = str(raw_conversational_tts)
        logger.info(f"OutputFormatterNode: Using 'conversational_tts' directly: '{conversational_tts[:100]}...' ")
    else:
        output_content_from_state = state.get("output_content", {})
        if isinstance(output_content_from_state, dict):
            raw_tts_from_output_content = output_content_from_state.get("text_for_tts")
            if raw_tts_from_output_content and str(raw_tts_from_output_content).strip():
                conversational_tts = str(raw_tts_from_output_content)
                logger.info(f"OutputFormatterNode: Using 'text_for_tts' from 'output_content': '{conversational_tts[:100]}...' ")
            else:
                conversational_tts = ""
                logger.info("OutputFormatterNode: 'text_for_tts' from 'output_content' is also empty or None.")
        else:
            conversational_tts = ""
            logger.info("OutputFormatterNode: 'output_content' is not a dict or not found, 'conversational_tts' remains empty.")

    # Combine TTS strings using a list
    tts_parts = []
    # if greeting_tts:
    #     tts_parts.append(greeting_tts)
    # if task_suggestion_tts:
    #     tts_parts.append(task_suggestion_tts)
    # if navigation_tts:
    #     tts_parts.append(navigation_tts)
    if conversational_tts: # This is the key we are debugging
        tts_parts.append(conversational_tts)
    # ---- END DEBUGGING SIMPLIFICATION ----

    # Filter out any genuinely empty strings that might have been added (e.g. if state.get defaulted to "")
    # and then join them with spaces.
    final_combined_tts = " ".join(filter(None, tts_parts))

    logger.info(f"OutputFormatterNode: TTS parts collected: {tts_parts}")
    logger.info(f"OutputFormatterNode: Final combined TTS after join: '{final_combined_tts[:200]}...' (Length: {len(final_combined_tts)})")
    
    if not final_combined_tts.strip(): # Check if empty or just whitespace
        logger.warning("OutputFormatterNode: No effective TTS strings found (or only whitespace). Using default response.")
        final_combined_tts = "I'm ready to help with your practice!"
    else:
        logger.info("OutputFormatterNode: Effective TTS string found.")

    # Retrieve UI Actions
    ui_actions = state.get("ui_actions_for_formatter", [])
    logger.info(f"OutputFormatterNode: Initial 'ui_actions_for_formatter': {ui_actions}")

    if not ui_actions: # If primary source is empty, try output_content
        output_content_from_state = state.get("output_content", {})
        if isinstance(output_content_from_state, dict):
            ui_actions_from_output_content = output_content_from_state.get("ui_actions", [])
            if ui_actions_from_output_content:
                ui_actions = ui_actions_from_output_content
                logger.info(f"OutputFormatterNode: Using 'ui_actions' from 'output_content': {ui_actions}")
            else:
                logger.info("OutputFormatterNode: 'ui_actions' from 'output_content' is also empty.")
        else:
            logger.info("OutputFormatterNode: 'output_content' is not a dict or not found, 'ui_actions' remains as initially fetched.")

    if not isinstance(ui_actions, list):
        logger.warning(f"OutputFormatterNode: Final 'ui_actions' is not a list, re-initializing. Value: {ui_actions}")
        ui_actions = []
    
    logger.info(f"OutputFormatterNode: Consolidating UI actions. Final effective ui_actions: {ui_actions}")
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
