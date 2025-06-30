import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def acknowledge_interrupt_node(state: AgentGraphState) -> dict:
    """
    Creates a full conversational sequence for handling an interruption.
    It first speaks an acknowledgment, then issues a command to listen.
    """
    user_id = state.get("user_id", "unknown_user")
    logger.info(f"Acknowledge Interrupt Node: Generating sequence for user {user_id}.")

    # --- 1. Create the sequence ---
    sequence = [
        {
            "type": "tts",
            "content": "Of course, I can help with that. What's on your mind?"
        },
        {
            "type": "listen",
            "expected_intent": "INTERRUPTION_QUESTION",
            "timeout_ms": 10000 
        }
    ]
    
    # --- 2. Package the sequence into a standard UI action ---
    ui_actions = [{
        "action_type": "EXECUTE_CONVERSATIONAL_SEQUENCE",
        "parameters": { "sequence": sequence }
    }]
    
    # --- 3. THIS IS THE FIX: Return the final, client-ready output ---
    # This node is a "finalizing node," so it must produce the final keys.
    return {
        "final_text_for_tts": "", # The TTS is now handled by the sequence.
        "final_ui_actions": ui_actions
    }
