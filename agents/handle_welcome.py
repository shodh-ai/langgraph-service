from state import AgentGraphState
import logging
import json

logger = logging.getLogger(__name__)


async def handle_welcome_node(state: AgentGraphState) -> dict:
    """
    Greets the user upon visiting the dashboard and navigates them to a starting task.
    """
    user_id = state.get('user_id', 'there')
    logger.info(f"Executing welcome node for user: {user_id}")

    # The text the AI will "speak" via TTS
    welcome_text = "Hello, welcome to your personalized TOEFL training session! Let's get started with a modelling exercise to warm you up. I'm taking you to the page now."

    # The UI action to navigate the user to the modellingcopy page - EXACTLY match the structure expected by the frontend
    navigation_action = {
        "action_type": "NAVIGATE_TO_PAGE",
        "parameters": {
            "page_name": "modellingcopy"
        }
    }
    
    # Create separate action for TTS text to ensure it's properly processed
    tts_action = {
        "action_type": "SPEAK_TEXT",
        "parameters": {
            "text": welcome_text
        }
    }
    
    # Log the exact actions we're returning
    logger.info(f"Welcome node UI actions: {json.dumps([tts_action, navigation_action])}")

    # The final payload, structured EXACTLY like the working modelling_output_formatter
    result = {
        "final_text_for_tts": welcome_text,
        "final_ui_actions": [tts_action, navigation_action]
    }
    
    # Debug log the full return state
    logger.info(f"Welcome node returning: {json.dumps(result, default=str)}")
    
    return result