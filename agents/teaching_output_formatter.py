# new graph/teaching_output_formatter.py
import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def teaching_output_formatter_node(state: AgentGraphState) -> dict:
    """
    Standardizes the output from the teaching generator node into the final
    format expected by the main graph and client streamer.
    """
    logger.info("---Executing Teaching Output Formatter---")
    
    # Defensively get the payload from the generator
    payload = state.get("intermediate_teaching_payload")
    if not payload:
        logger.warning("Teaching formatter received no intermediate payload. Using error fallback.")
        return {
            "final_text_for_tts": "Error generating teaching content.",
            "final_ui_actions": []
        }
    
    # --- Assemble the final TTS script from the raw components ---
    tts_parts = [
        payload.get("opening_hook"),
        payload.get("core_explanation"),
        "For example: " + payload.get("key_examples", ""),
        payload.get("comprehension_check")
    ]
    text_for_tts = " ".join(filter(None, tts_parts)).strip()
    
    # --- Assemble the UI actions ---
    ui_actions = []
    visual_aid = payload.get("visual_aid_instructions")
    if visual_aid:
        ui_actions.append({
            "action_type": "DISPLAY_VISUAL_AID",
            "parameters": visual_aid # The LLM has already generated the correct JSON structure
        })
        
    logger.info(f"Teaching formatter created TTS and {len(ui_actions)} UI actions.")

    # Return the final, client-ready keys directly
    return {
        "final_text_for_tts": text_for_tts,
        "final_ui_actions": ui_actions
    }