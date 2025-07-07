# agents/modelling_output_formatter.py (Simplified)
import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

ACTION_TYPE_MAP = {
    "update_prompt_display": "UPDATE_TEXT_CONTENT",
    "think_aloud": "UPDATE_TEXT_CONTENT",
    "ai_writing_chunk": "APPEND_TEXT_TO_EDITOR_REALTIME",
    "highlight_writing": "HIGHLIGHT_TEXT_RANGES",
    "display_remark": "DISPLAY_REMARKS_LIST",
    "self_correction": "REPLACE_TEXT_RANGE",
}

TARGET_ELEMENT_MAP = {
    "update_prompt_display": "p8ModelTaskPromptDisplay",
    "think_aloud": "p8AiThinkAloudPanelOrSubtitle",
    # All other actions target the main editor or panel on the modelling page
    "default": "p8ModelEssayDisplayArea", 
}

async def modelling_output_formatter_node(state: AgentGraphState) -> dict:
    """
    Translates the generator's script into a simple list of UI action dictionaries
    for the livekit-service to process AND preserves the full session state.
    """
    logger.info("---Executing Fully State-Preserving Modelling Output Formatter ---")

    # --- Start of State Preservation ---
    modeling_plan = state.get("modeling_plan")
    current_plan_step_index = state.get("current_plan_step_index")
    activity_id = state.get("activity_id")
    learning_objective = state.get("Learning_Objective_Focus")
    student_proficiency = state.get("STUDENT_PROFICIENCY")
    student_affective_state = state.get("STUDENT_AFFECTIVE_STATE")
    # --- End of State Preservation ---

    payload = state.get("intermediate_modelling_payload")
    if not payload or "sequence" not in payload:
        logger.warning("No payload or sequence found in state. Returning empty actions.")
        # Still return the preserved state even on failure
        return {
            "final_text_for_tts": "I seem to have lost my train of thought. Let's try that again.",
            "final_ui_actions": [],
            "modeling_plan": modeling_plan,
            "current_plan_step_index": current_plan_step_index,
            "activity_id": activity_id,
            "Learning_Objective_Focus": learning_objective,
            "STUDENT_PROFICIENCY": student_proficiency,
            "STUDENT_AFFECTIVE_STATE": student_affective_state,
        }

    sequence = payload.get("sequence", [])
    final_tts_parts = []
    final_ui_actions = []

    for step in sequence:
        step_type = step.get("type")
        step_payload = step.get("payload", {})
        action_type_str = ACTION_TYPE_MAP.get(step_type)

        if not action_type_str:
            continue

        if step_type in ["think_aloud", "ai_writing_chunk"]:
            final_tts_parts.append(step_payload.get("text") or step_payload.get("text_chunk"))

        params = {}
        if action_type_str == "UPDATE_TEXT_CONTENT":
            params["target_element_id"] = TARGET_ELEMENT_MAP.get(step_type, TARGET_ELEMENT_MAP["default"])
            params["text"] = step_payload.get("text")
        else:
            params = step_payload

        final_ui_actions.append({"action_type": action_type_str, "parameters": params})

    # Return the formatted output AND the full, preserved session state.
    return {
        "final_text_for_tts": " ".join(filter(None, final_tts_parts)),
        "final_ui_actions": final_ui_actions,

        # Pass through the entire session state
        "modeling_plan": modeling_plan,
        "current_plan_step_index": current_plan_step_index,
        "activity_id": activity_id,
        "Learning_Objective_Focus": learning_objective,
        "STUDENT_PROFICIENCY": student_proficiency,
        "STUDENT_AFFECTIVE_STATE": student_affective_state,
        "intermediate_modelling_payload": None, # Clear payload after use
    }