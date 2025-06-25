import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def acknowledge_interrupt_node(state: AgentGraphState) -> dict:
    """
    Generates a brief acknowledgment to a user's interruption and prompts
    them to ask their question.
    """
    user_id = state.get("user_id", "unknown_user")
    logger.info(f"Acknowledge Interrupt Node: Generating response for user {user_id}.")

    response_text = "I see you have a question. Of course, what's on your mind?"
    
    # This UI action tells the frontend to activate the mic or show a listening indicator.
    ui_actions = [{
        "action_type": "PROMPT_FOR_USER_INPUT",
        "parameters": {"prompt_text": "Listening..."}
    }]

    # This node returns the final, client-ready output directly.
    return {
        "final_text_for_tts": response_text,
        "final_ui_actions": ui_actions
    }
