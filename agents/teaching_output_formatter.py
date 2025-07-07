# In your LangGraph project's teaching_output_formatter.py

import logging
import uuid
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

async def teaching_output_formatter_node(state: dict) -> dict:
    """
    Translates the generator's rich payload into a set of executable
    UI actions for the livekit-service, including both a conversational
    sequence and any visual aids.
    """
    logger.info("---Executing Comprehensive Teaching Output Formatter---")
    
    payload = state.get("intermediate_teaching_payload", {})
    if not payload:
        logger.error("Formatter received an empty teaching payload.")
        return {"final_ui_actions": []}

    final_actions = []
    full_text_to_speak = ""

    # --- 1. Process the Conversational Sequence ---
    sequence = payload.get("sequence", [])
    if sequence:
        tts_parts = [step.get("content", "") for step in sequence if step.get("type") == "tts"]
        full_text_to_speak = " ".join(filter(None, tts_parts)).strip()
        
        listen_step = next((step for step in reversed(sequence) if step.get("type") == "listen"), None)
        
        if listen_step:
            speak_then_listen_action = {
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
            }
            final_actions.append(speak_then_listen_action)
        else:
            # If no listen step, just create a simple speak action
            final_actions.append({
                "action_type": "SPEAK_TEXT",
                "parameters": {"text": full_text_to_speak}
            })
    else:
        # Fallback if no sequence is provided
        full_text_to_speak = payload.get("core_explanation", "Let me explain.")
        final_actions.append({
                "action_type": "SPEAK_TEXT",
                "parameters": {"text": full_text_to_speak}
            })

    # --- 2. Process the Visual Aid Suggestion ---
    visual_suggestion = payload.get("visual_aid_suggestion")
    if visual_suggestion and isinstance(visual_suggestion, dict) and visual_suggestion.get("steps"):
        logger.info("Found a visual aid suggestion to format.")
        visual_aid_action = {
            "action_type": "DISPLAY_VISUAL_AID",
            "parameters": {
                # The frontend expects the entire object in a 'visual_description' parameter
                "visual_description": visual_suggestion 
            }
        }
        # Prepend the visual aid action so the canvas draws *while* the AI is speaking
        final_actions.insert(0, visual_aid_action)

    logger.info(f"Formatted teaching payload into {len(final_actions)} final UI actions.")

    # Return in the standard format
    return {
        "final_text_for_tts": full_text_to_speak,
        "final_ui_actions": final_actions,
    }