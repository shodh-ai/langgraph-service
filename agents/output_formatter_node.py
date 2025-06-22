import logging
from typing import Dict, Any, List, Optional
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def format_final_output_for_client_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Consolidates all intended outputs for the client (text for TTS, UI actions, 
    navigation instructions, next task information, raw system outputs) from 
    various fields of the AgentGraphState populated by preceding agent nodes.
    Structures this information into a definitive dictionary for the FastAPI endpoint.
    
    Args:
        state: The current agent graph state
        
    Returns:
        A dictionary containing the final structured output for the client.
    """
    node_name = "OutputFormatterNode" # Consistent logging prefix
    logger.info(f"{node_name}: Formatting final output for the client.")

    client_response_data: Dict[str, Any] = {}

    # --- Standardized Output Formatting for TTS and UI Actions ---
    final_text_for_tts = ""
    final_ui_actions: List[Dict[str, Any]] = []

    # Check for output from the modelling flow first, as it's a self-contained unit.
    modelling_output = state.get("modelling_output_content")
    if modelling_output and isinstance(modelling_output, dict):
        logger.info(f"{node_name}: Found 'modelling_output_content'. Prioritizing it for final output.")
        final_text_for_tts = modelling_output.get("text_for_tts", "")
        final_ui_actions = modelling_output.get("ui_actions", [])
        logger.info(f"{node_name}: Extracted from modelling output - TTS: '{final_text_for_tts[:100]}...', UI Actions: {len(final_ui_actions)} actions.")
    else:
        # --- Fallback to Consolidating from various other sources if no modelling output ---
        logger.info(f"{node_name}: 'modelling_output_content' not found. Consolidating from other state fields.")
        
        # Consolidate TTS from multiple potential sources in order of priority
        teaching_output = state.get("teaching_output_content") or {}
        pedagogy_output = state.get("task_suggestion_llm_output") or {}
        llm_output = state.get("llm_output") or {}

        tts_sources = [
            teaching_output.get("text_for_tts"),
            pedagogy_output.get("task_suggestion_tts"),
            state.get("conversational_tts"),
            llm_output.get("text_for_tts")
        ]
        
        for source_tts in tts_sources:
            if source_tts and isinstance(source_tts, str) and source_tts.strip():
                final_text_for_tts = source_tts
                logger.info(f"{node_name}: Found TTS from a fallback source: '{final_text_for_tts[:100]}...'")
                break
        
        # Consolidate UI actions from the accumulated list
        accumulated_ui_actions = state.get("accumulated_ui_actions", [])
        if isinstance(accumulated_ui_actions, list):
            final_ui_actions.extend(accumulated_ui_actions)
        else:
            logger.warning(f"{node_name}: 'accumulated_ui_actions' was not a list. Resetting to empty list.")


    # Final assignment to the response dictionary
    client_response_data["final_text_for_tts"] = final_text_for_tts
    client_response_data["final_ui_actions"] = final_ui_actions
    logger.info(f"{node_name}: Final consolidated TTS: '{final_text_for_tts[:100]}...'")
    logger.info(f"{node_name}: Final consolidated UI Actions: {final_ui_actions}")

    # --- Next Task Information ---
    next_task_details = state.get("next_task_details")
    if next_task_details is not None:
        client_response_data["final_next_task_info"] = next_task_details
        logger.info(f"{node_name}: Included 'final_next_task_info': {next_task_details}")
    else:
        client_response_data["final_next_task_info"] = None
        logger.info(f"{node_name}: 'next_task_details' not found. 'final_next_task_info' is None.")

    # --- Navigation Instruction ---
    nav_target = state.get("navigation_instruction_target") # Key from design document
    nav_data = state.get("data_for_target_page")       # Key from design document

    if nav_target:
        navigation_instruction = {"page_name": nav_target}
        if nav_data is not None:
            navigation_instruction["data"] = nav_data
        client_response_data["final_navigation_instruction"] = navigation_instruction
        logger.info(f"{node_name}: Included 'final_navigation_instruction': {navigation_instruction}")
    else:
        client_response_data["final_navigation_instruction"] = None
        logger.info(f"{node_name}: 'navigation_instruction_target' not found. 'final_navigation_instruction' is None.")

    # --- Raw System Outputs ---
    # This section is for debugging and providing the frontend with all context if needed.
    raw_teaching = state.get("teaching_output_content")
    if raw_teaching:
        client_response_data["raw_teaching_output"] = raw_teaching
        logger.info(f"{node_name}: Included 'raw_teaching_output'.")

    # Use the 'modelling_output' variable we defined earlier
    if modelling_output:
        client_response_data["raw_modelling_output"] = modelling_output
        logger.info(f"{node_name}: Included 'raw_modelling_output'.")
        # Flatten other keys from modelling_output into the main response for easier access
        if isinstance(modelling_output, dict):
            for key, value in modelling_output.items():
                if key not in ["text_for_tts", "ui_actions"]:
                    if key not in client_response_data:
                        client_response_data[key] = value
                        logger.info(f"{node_name}: Flattened '{key}' from modelling_output_content into response.")
                    else:
                        logger.warning(f"{node_name}: Key '{key}' from modelling_output_content already exists in response ('{client_response_data.get(key)}'). Not overwriting with '{value}'.")

    raw_feedback = state.get("feedback_output_content")
    if raw_feedback:
        client_response_data["raw_feedback_output"] = raw_feedback
        logger.info(f"{node_name}: Included 'raw_feedback_output'.")

    raw_pedagogy = state.get("task_suggestion_llm_output")
    if raw_pedagogy:
        client_response_data["raw_pedagogy_output"] = raw_pedagogy
        logger.info(f"{node_name}: Included 'raw_pedagogy_output'.")

    raw_initial_report = state.get("initial_report_content")
    if raw_initial_report:
        client_response_data["raw_initial_report_output"] = raw_initial_report
        logger.info(f"{node_name}: Included 'raw_initial_report_output'.")

    logger.info(f"{node_name}: Final client response data keys: {list(client_response_data.keys())}")
    
    return client_response_data
