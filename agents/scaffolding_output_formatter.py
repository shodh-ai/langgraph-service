# new graph/scaffolding_output_formatter.py
import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def scaffolding_output_formatter_node(state: AgentGraphState) -> dict:
    """
    Standardizes the output from the scaffolding generator node.
    """
    logger.info("---Executing Scaffolding Output Formatter---")
    
    # Defensively get the payload from the generator
    payload = state.get("intermediate_scaffolding_payload", {})
    if not payload:
        return {
            "final_text_for_tts": "Error in scaffolding.",
            "final_ui_actions": []
        }
    
    # The generator already created the correct structure, so we just pass it on
    text_for_tts = payload.get("text_for_tts", "Here is some help.")
    ui_actions = payload.get("ui_actions", [])
    
    # Return the final, client-ready keys directly
    return {
        "final_text_for_tts": text_for_tts,
        "final_ui_actions": ui_actions
    }