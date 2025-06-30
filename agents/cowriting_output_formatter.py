# graph/cowriting_output_formatter.py

import logging
from state import AgentGraphState
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def create_ui_action(action_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """A simple helper to ensure consistent UI action structure."""
    return {"action_type": action_type, "parameters": parameters}

async def cowriting_output_formatter_node(state: AgentGraphState) -> dict:
    """
    Translates the generator's conversational plan into specific,
    rich UI actions for the co-writing frontend.
    """
    logger.info("---Executing Co-Writing Output Formatter---")

    payload = state.get("intermediate_cowriting_payload")
    if not payload or payload.get("error"):
        error_msg = payload.get("error_message", "An unknown error occurred in the co-writing generator.")
        logger.warning(f"Co-writing formatter received an error or empty payload: {error_msg}")
        # Return a graceful error message to the user
        return {
            "final_text_for_tts": error_msg,
            "final_ui_actions": []
        }

    # --- This is the core translation logic ---
    final_ui_actions: List[Dict[str, Any]] = []
    
    # Get the conversational sequence from the generator
    sequence = payload.get("sequence", [])

    # The TTS for the user is the combination of all spoken parts in the sequence.
    tts_parts = [step.get("content", "") for step in sequence if step.get("type") == "tts"]
    final_tts = " ".join(filter(None, tts_parts)).strip()

    # Now, let's create the UI actions you want. We can infer them from the generator's output.
    # This is a conceptual example. In a real system, the generator's output might be richer,
    # or you might add more logic here to decide which UI action to show.

    # For now, let's assume we want to show the AI's suggestions in a separate panel.
    main_explanation = payload.get("main_explanation")
    if main_explanation:
        final_ui_actions.append(
            create_ui_action(
                action_type="UPDATE_TEXT_CONTENT",
                parameters={
                    "targetElementId": "liveCoachingFeedbackPanel",
                    "text": f"AI Suggestion: {main_explanation}"
                }
            )
        )

    # Let's also add one of the more complex UI actions as an example.
    # Imagine the generator's output also included a specific suggestion.
    # We will simulate this for now.
    simulated_suggestion = {
        "start_pos": 50,
        "end_pos": 75,
        "suggestion_text": "Consider rephrasing for a stronger argument."
    }
    final_ui_actions.append(
        create_ui_action(
            action_type="SHOW_INLINE_SUGGESTION",
            parameters={
                "targetElementId": "liveWritingEditor",
                **simulated_suggestion # Unpack the suggestion dictionary
            }
        )
    )

    # The final step of the sequence is always a 'listen'.
    # This translates directly to the 'PROMPT_FOR_USER_INPUT' action we designed.
    last_step = sequence[-1] if sequence else {}
    if last_step.get("type") == "listen":
        final_ui_actions.append(
            create_ui_action(
                action_type="PROMPT_FOR_USER_INPUT",
                parameters={
                    "prompt_text": last_step.get("prompt_if_silent", "Your turn!"),
                    "expected_intent": last_step.get("expected_intent")
                }
            )
        )
    
    logger.info(f"Co-writing formatter created {len(final_ui_actions)} UI actions.")

    # Return the final, client-ready output
    return {
        "final_text_for_tts": final_tts,
        "final_ui_actions": final_ui_actions
    }