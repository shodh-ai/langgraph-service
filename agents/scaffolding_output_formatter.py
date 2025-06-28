# agents/scaffolding_output_formatter.py
import logging
from state import AgentGraphState
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def create_ui_action(action_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to create a UI action dictionary."""
    return {"action_type": action_type, "parameters": parameters}

async def scaffolding_output_formatter_node(state: AgentGraphState) -> dict:
    """
    Translates the generator's scaffolding plan into specific UI actions.
    """
    logger.info("---Executing Scaffolding Output Formatter---")

    payload = state.get("intermediate_scaffolding_payload")
    if not payload or payload.get("error"):
        error_msg = payload.get("error_message", "An error occurred preparing the activity.")
        logger.warning(f"Scaffolding Formatter received an error: {error_msg}")
        return {"final_text_for_tts": error_msg, "final_ui_actions": []}

    final_ui_actions: List[Dict[str, Any]] = []

    # 1. Update the main prompt display
    if prompt_text := payload.get("prompt_display_text"):
        final_ui_actions.append(create_ui_action(
            "UPDATE_TEXT_CONTENT",
            {"targetElementId": "p8dEssayPromptDisplay", "text": prompt_text}
        ))

    # 2. Set the initial content of the student's editor
    if editor_content := payload.get("initial_editor_content"):
        final_ui_actions.append(create_ui_action(
            "SET_EDITOR_CONTENT",
            {"targetElementId": "p8dStudentWritingEditor", "html_content": editor_content}
        ))

    # The guidance script from the generator becomes the TTS and bubble text
    guidance_script = payload.get("ai_guidance_script", [])
    
    # 3. The first line of the script is the main instruction for TTS and the chat bubble
    if guidance_script:
        main_instruction = guidance_script[0]
        final_text_for_tts = main_instruction
        final_ui_actions.append(create_ui_action(
            "UPDATE_TEXT_CONTENT",
            {"targetElementId": "p8dCurrentInstruction", "text": main_instruction}
        ))
        final_ui_actions.append(create_ui_action(
            "UPDATE_TEXT_CONTENT",
            {"targetElementId": "p8dAiChatBubble", "text": main_instruction}
        ))
    else:
        final_text_for_tts = "Okay, please begin the writing task."

    # Note: Subsequent feedback like "Good topic sentence!" would be handled
    # by a separate "handle_student_writing_chunk" flow that triggers after
    # the student types something. This setup is for the INITIAL scaffolding.
    
    logger.info(f"Scaffolding formatter created TTS and {len(final_ui_actions)} UI actions.")

    return {
        "final_text_for_tts": final_text_for_tts,
        "final_ui_actions": final_ui_actions
    }