# In your LangGraph project's teaching_output_formatter.py

import logging
import uuid
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

async def teaching_output_formatter_node(state: dict) -> dict:
    """
    Translates the generator's rich payload into final UI actions AND
    preserves the entire session state for the checkpointer.
    """
    logger.info("---Executing Comprehensive and State-Preserving Output Formatter---")

    payload = state.get("intermediate_teaching_payload", {})
    final_actions = []
    full_text_to_speak = ""

    if not payload:
        logger.error("Formatter received an empty teaching payload.")
        # Even on error, we must preserve the state
    else:
        # --- 1. Process the Conversational Sequence ---
        sequence = payload.get("sequence", [])
        if sequence:
            tts_parts = [step.get("content", "") for step in sequence if step.get("type") == "tts"]
            full_text_to_speak = " ".join(filter(None, tts_parts)).strip()
            listen_step = next((step for step in reversed(sequence) if step.get("type") == "listen"), None)
            
            if listen_step:
                final_actions.append({
                    "action_type": "SPEAK_THEN_LISTEN",
                    "parameters": {
                        "speech_id": f"teach-{uuid.uuid4().hex[:8]}",
                        "text_to_speak": full_text_to_speak,
                        "listen_config": {
                            "timeout_ms": listen_step.get("timeout_ms", 8000),
                            "prompt_if_silent": listen_step.get("prompt_if_silent", "Are you still there?"),
                            "expected_intent": listen_step.get("expected_intent", "user_response")
                        }
                    }
                })
            else:
                final_actions.append({"action_type": "SPEAK_TEXT", "parameters": {"text": full_text_to_speak}})
        else:
            full_text_to_speak = payload.get("core_explanation", "Let me explain.")
            final_actions.append({"action_type": "SPEAK_TEXT", "parameters": {"text": full_text_to_speak}})

        # --- 2. Process the Visual Aid Suggestion ---
        visual_suggestion = payload.get("visual_aid_suggestion")
        if visual_suggestion and isinstance(visual_suggestion, dict) and visual_suggestion.get("steps"):
            final_actions.insert(0, {
                "action_type": "DISPLAY_VISUAL_AID",
                "parameters": {"visual_description": visual_suggestion}
            })

    logger.info(f"Formatted teaching payload into {len(final_actions)} final UI actions.")

    # --- THIS IS THE CRITICAL FIX ---
    # Return the final output AND all the state keys that need to be saved.
    return {
        "final_text_for_tts": full_text_to_speak,
        "final_ui_actions": final_actions,

        # Preserve the session state
        "pedagogical_plan": state.get("pedagogical_plan"),
        "current_plan_step_index": state.get("current_plan_step_index"),
        "lesson_id": state.get("lesson_id"),
        "Learning_Objective_Focus": state.get("Learning_Objective_Focus"),
        "STUDENT_PROFICIENCY": state.get("STUDENT_PROFICIENCY"),
        "STUDENT_AFFECTIVE_STATE": state.get("STUDENT_AFFECTIVE_STATE"),
    }