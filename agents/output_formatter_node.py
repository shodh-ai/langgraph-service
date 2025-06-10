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

    # Retrieve Conversational TTS
    # Priority for conversational_tts:
    # 1. state.teaching_output_content.text_for_tts
    # 2. state.modelling_output_content.text_for_tts
    # 3. state.conversational_tts (direct key)
    # 4. state.output_content.text_for_tts

    conversational_tts = ""

    # 1. Check teaching_output_content
    teaching_output = state.get("teaching_output_content", {})
    logger.info(f"OutputFormatterNode: Checking 'teaching_output_content' for TTS: {teaching_output.get('text_for_tts') is not None}")
    if isinstance(teaching_output, dict) and teaching_output.get("text_for_tts"):
        tts_candidate = str(teaching_output["text_for_tts"])
        if tts_candidate.strip():
            conversational_tts = tts_candidate
            logger.info(f"OutputFormatterNode: Using 'text_for_tts' from 'teaching_output_content': '{conversational_tts[:100]}...' ")

    # 2. Check modelling_output_content
    modelling_output = state.get("modelling_output_content", {}) # Initialize modelling_output here
    logger.info(f"OutputFormatterNode: Checking 'modelling_output_content' for TTS: {modelling_output.get('text_for_tts') is not None}") # Safe to log now

    if not conversational_tts: # Only use modelling_output for TTS if not already found
        if isinstance(modelling_output, dict) and modelling_output.get("text_for_tts"):
            tts_candidate = str(modelling_output["text_for_tts"])
            if tts_candidate.strip():
                conversational_tts = tts_candidate
                logger.info(f"OutputFormatterNode: Using 'text_for_tts' from 'modelling_output_content': '{conversational_tts[:100]}...' ")
    
    if not conversational_tts:
        raw_conversational_tts_direct = state.get("conversational_tts")
        logger.info(f"OutputFormatterNode: Checking 'conversational_tts' directly: {raw_conversational_tts_direct is not None}")
        if raw_conversational_tts_direct and str(raw_conversational_tts_direct).strip():
            conversational_tts = str(raw_conversational_tts_direct)
            logger.info(f"OutputFormatterNode: Using 'conversational_tts' directly: '{conversational_tts[:100]}...' ")
        else:
            output_content_from_state = state.get("output_content", {})
            logger.info(f"OutputFormatterNode: Checking 'output_content' for TTS: {output_content_from_state.get('text_for_tts') is not None}")
            if isinstance(output_content_from_state, dict):
                raw_tts_from_output_content = output_content_from_state.get("text_for_tts")
                if raw_tts_from_output_content and str(raw_tts_from_output_content).strip():
                    conversational_tts = str(raw_tts_from_output_content)
                    logger.info(f"OutputFormatterNode: Using 'text_for_tts' from 'output_content': '{conversational_tts[:100]}...' ")
                else:
                    logger.info("OutputFormatterNode: 'text_for_tts' from 'output_content' is empty or None.")
            else:
                logger.info("OutputFormatterNode: 'output_content' is not a dict or not found.")
    
    if not conversational_tts.strip():
        logger.info("OutputFormatterNode: No effective conversational TTS found from any source. Will be empty string for tts_parts.")
        conversational_tts = "" # Ensure it's an empty string if nothing found or only whitespace

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
    # Priority for ui_actions:
    # 1. state.teaching_output_content.ui_actions
    # 2. state.modelling_output_content.ui_actions
    # 3. state.ui_actions_for_formatter
    # 4. state.output_content.ui_actions
    
    ui_actions = [] # Default to empty list

    # 1. Check teaching_output_content for UI actions
    teaching_output_for_ui = state.get("teaching_output_content", {})
    logger.info(f"OutputFormatterNode: Checking 'teaching_output_content' for UI actions: {teaching_output_for_ui.get('ui_actions') is not None}")
    teaching_actions = teaching_output_for_ui.get("ui_actions")
    if teaching_actions is not None:
        logger.info(f"OutputFormatterNode: Found 'ui_actions' in teaching_output_content. Type: {type(teaching_actions)}, Content (first 200 chars): {str(teaching_actions)[:200]}")
        if isinstance(teaching_actions, list) and teaching_actions:
            try:
                ui_actions.extend(teaching_actions)
                logger.info(f"OutputFormatterNode: Successfully extended ui_actions from teaching_output_content. Count: {len(ui_actions)}")
            except TypeError as e:
                logger.error(f"OutputFormatterNode: TypeError extending ui_actions from teaching_output_content. Error: {e}", exc_info=True)
        elif teaching_actions: # Truthy but not a list or empty list
            logger.warning(f"OutputFormatterNode: 'ui_actions' in teaching_output_content was not a non-empty list. Ignored.")
    else:
        logger.info("OutputFormatterNode: 'ui_actions' key not found in teaching_output_content or was None.")

    # 2. Check modelling_output_content
    modelling_output_for_ui = state.get("modelling_output_content", {}) # Initialize unconditionally
    logger.info(f"OutputFormatterNode: Checking 'modelling_output_content' for UI actions: {modelling_output_for_ui.get('ui_actions') is not None}")
    
    # Only try to extend ui_actions from modelling_output if ui_actions is still empty (i.e., not populated by teaching_output)
    if not ui_actions: 
        modelling_actions = modelling_output_for_ui.get("ui_actions")
        if modelling_actions is not None:
            logger.info(f"OutputFormatterNode: Found 'ui_actions' in modelling_output_content. Type: {type(modelling_actions)}, Content (first 200 chars): {str(modelling_actions)[:200]}")
            if isinstance(modelling_actions, list) and modelling_actions:
                try:
                    ui_actions.extend(modelling_actions)
                    logger.info(f"OutputFormatterNode: Successfully extended ui_actions from modelling_output_content. Count: {len(ui_actions)}")
                except TypeError as e:
                    logger.error(f"OutputFormatterNode: TypeError extending ui_actions from modelling_output_content. Error: {e}", exc_info=True)
            elif modelling_actions: # Truthy but not a list or empty list
                logger.warning(f"OutputFormatterNode: 'ui_actions' in modelling_output_content was not a non-empty list. Ignored.")
        else:
            logger.info("OutputFormatterNode: 'ui_actions' key not found in modelling_output_content or was None.")

    # Fallback to ui_actions_for_formatter if no UI actions from modelling_output_content
    if not ui_actions:
        formatter_actions = state.get("ui_actions_for_formatter")
        if formatter_actions is not None:
            logger.info(f"OutputFormatterNode: Checking 'ui_actions_for_formatter'. Type: {type(formatter_actions)}, Content: {str(formatter_actions)[:200]}")
            if isinstance(formatter_actions, list) and formatter_actions:
                try:
                    ui_actions.extend(formatter_actions)
                    logger.info(f"OutputFormatterNode: Successfully extended ui_actions from ui_actions_for_formatter. Count: {len(ui_actions)}")
                except TypeError as e:
                    logger.error(f"OutputFormatterNode: TypeError extending ui_actions from ui_actions_for_formatter. Error: {e}", exc_info=True)
            elif formatter_actions:
                logger.warning("OutputFormatterNode: 'ui_actions_for_formatter' was not a non-empty list. Ignored.")
        else:
            logger.info("OutputFormatterNode: 'ui_actions_for_formatter' not found or was None.")

    # Fallback to output_content's ui_actions if still no UI actions
    if not ui_actions:
        output_content_data = state.get("output_content") # Retrieve without default first
        if isinstance(output_content_data, dict):
            oc_actions = output_content_data.get("ui_actions")
            if oc_actions is not None:
                logger.info(f"OutputFormatterNode: Checking 'output_content.ui_actions'. Type: {type(oc_actions)}, Content: {str(oc_actions)[:200]}")
                if isinstance(oc_actions, list) and oc_actions:
                    try:
                        ui_actions.extend(oc_actions)
                        logger.info(f"OutputFormatterNode: Successfully extended ui_actions from output_content. Count: {len(ui_actions)}")
                    except TypeError as e:
                        logger.error(f"OutputFormatterNode: TypeError extending ui_actions from output_content. Error: {e}", exc_info=True)
                elif oc_actions:
                    logger.warning("OutputFormatterNode: 'ui_actions' in output_content was not a non-empty list. Ignored.")
            else:
                logger.info("OutputFormatterNode: 'ui_actions' key not found in 'output_content' dictionary or was None.")
        elif output_content_data is not None: # It exists in state, but it's not a dict
            logger.warning(f"OutputFormatterNode: 'output_content' in state is not a dictionary (Type: {type(output_content_data)}). UI actions from it will be ignored.")
        else: # It's not in state at all
            logger.info("OutputFormatterNode: 'output_content' not found in state. UI actions from it will be ignored.")

    # Final safeguard: ensure ui_actions is a list. If it became None or some other non-list type
    # after all prior processing, default it to an empty list.
    if not isinstance(ui_actions, list):
        logger.warning(f"OutputFormatterNode: 'ui_actions' was not a list before final response creation (Type: {type(ui_actions)}, Value: {str(ui_actions)[:200]}). Resetting to empty list.")
        ui_actions = []
    
    # Log the definitive state of ui_actions just before it's added to the client response dictionary.
    logger.info(f"OutputFormatterNode: Final 'ui_actions' for client response: {ui_actions} (Type: {type(ui_actions)})")
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
