# graph/feedback_output_formatter.py
import logging
from state import AgentGraphState


logger = logging.getLogger(__name__)

async def feedback_output_formatter_node(state: AgentGraphState) -> dict:
    logger.info("---Executing Feedback Output Formatter---")
    
    payload = state.get("intermediate_feedback_payload", {})
    if not payload:
        # Handle error case
        return {
            "final_text_for_tts": "Error generating feedback.",
            "final_ui_actions": []
        }
    
    # --- LOGIC TO FORMAT THE OUTPUT ---
    # Combine the parts into a single speech string
    tts_parts = [
        payload.get("acknowledgement"),
        payload.get("explanation"),
        "Here is a corrected example:",
        payload.get("corrected_example"),
        payload.get("follow_up_task")
    ]
    text_for_tts = " ".join(filter(None, tts_parts))
    
    # Create any UI actions needed, e.g., show the corrected example in a special UI box
    ui_actions = [{
        "action_type": "SHOW_EXAMPLE_BOX",
        "parameters": {
            "title": "Corrected Example",
            "text": payload.get("corrected_example")
        }
    }]
    
    # Return the final, client-ready keys directly
    return {
        "final_text_for_tts": text_for_tts,
        "final_ui_actions": ui_actions
    }