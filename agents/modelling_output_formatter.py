# agents/modelling_output_formatter.py
import logging
from state import AgentGraphState
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# A mapping from the generator's simple action types to the frontend's specific action types
ACTION_TYPE_MAP = {
    "update_prompt_display": "UPDATE_TEXT_CONTENT",
    "think_aloud": "UPDATE_TEXT_CONTENT",
    "ai_writing_chunk": "APPEND_TEXT_TO_EDITOR_REALTIME",
    "highlight_writing": "HIGHLIGHT_TEXT_RANGES",
    "display_remark": "DISPLAY_REMARKS_LIST",
    "self_correction": "REPLACE_TEXT_RANGE",
}

# A mapping for the targetElementId for each action
TARGET_ELEMENT_MAP = {
    "update_prompt_display": "p8ModelTaskPromptDisplay",
    "think_aloud": "p8AiThinkAloudPanelOrSubtitle",
    "ai_writing_chunk": "p8ModelEssayDisplayArea", # Or p8TiptapEditorForAiSpeech
    "highlight_writing": "p8ModelEssayDisplayArea",
    "display_remark": "p8ModelExplanationPanel",
    "self_correction": "p8ModelEssayDisplayArea",
}

async def modelling_output_formatter_node(state: AgentGraphState) -> dict:
    """
    Translates the generator's creative script into a sequence of precise UI actions.
    """
    logger.info("---Executing Modelling Output Formatter (Advanced UI Version)---")

    payload = state.get("intermediate_modelling_payload")
    if not payload or payload.get("error"):
        error_msg = payload.get("error_message", "An error occurred preparing the model.")
        logger.warning(f"Modelling Formatter received an error: {error_msg}")
        return {"final_text_for_tts": error_msg, "final_ui_actions": []}

    sequence = payload.get("sequence", [])
    if not sequence:
        return {"final_text_for_tts": "I had an idea, but couldn't quite structure it. Let's try again.", "final_ui_actions": []}

    final_tts_parts = []
    final_ui_actions = []

    for step in sequence:
        step_type = step.get("type")
        step_payload = step.get("payload", {})
        
        # --- 1. Assemble the TTS Script ---
        # We'll speak the "think_aloud" parts and the written chunks.
        if step_type in ["think_aloud", "ai_writing_chunk"]:
            final_tts_parts.append(step_payload.get("text") or step_payload.get("text_chunk"))

        # --- 2. Translate to Frontend UI Actions ---
        action_type = ACTION_TYPE_MAP.get(step_type)
        target_id = TARGET_ELEMENT_MAP.get(step_type)

        if not action_type or not target_id:
            logger.warning(f"No UI action mapping found for step type: {step_type}")
            continue

        # Prepare parameters based on the specific action
        params = {}
        if step_type == "highlight_writing":
            params = {"ranges": [step_payload]}
        elif step_type == "display_remark":
            params = {"remarks": [step_payload]}
        else:
            params = step_payload # For most actions, the payload maps directly

        final_ui_actions.append({
            "action_type": action_type,
            "parameters": {
                "targetElementId": target_id,
                **params # Unpack the prepared parameters
            }
        })

    # This is a complex interaction, so we package it into a single sequence command
    # for the livekit-service to orchestrate.
    final_orchestration_action = {
        "action_type": "EXECUTE_INTERACTIVE_SEQUENCE",
        "parameters": {
            "tts_script": " ".join(filter(None, final_tts_parts)),
            "ui_actions_with_timing": final_ui_actions # A real implementation might add timing info
        }
    }
    
    return {
        "final_text_for_tts": " ".join(filter(None, final_tts_parts)), # Still return the full TTS
        "final_ui_actions": final_ui_actions # And the list of actions
    }