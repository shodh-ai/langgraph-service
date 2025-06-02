# agents/special_feedback_node.py
import logging
# Assuming state.py is in the parent directory relative to agents/
# Adjust the import path if your directory structure is different.
# For example, if 'agents' and 'state.py' are siblings in 'backend_ai_service_langgraph':
# from ..state import AgentGraphState 
# If 'state.py' is directly in 'backend_ai_service_langgraph' and this file is in 'backend_ai_service_langgraph/agents':
from state import AgentGraphState # MODIFIED LINE

logger = logging.getLogger(__name__)

async def generate_test_button_feedback_stub_node(state: AgentGraphState) -> dict:
    user_id = state.get("user_id", "unknown_user_in_test_button_node")
    logger.info(f"FeedbackNodeStub (TestButton): Generating specific feedback for TEST BUTTON context for user '{user_id}'.")
    
    text_response_for_tts = "Backend UI action test initiated." # Optional TTS message
    
    # Define the UI action to be sent to the frontend
    ui_actions = [
        {
            "action_type": "SHOW_ALERT",  # Updated from action_type_str
            "parameters": {                   # Payload for the action
                "message": "Backend UI Action Test: Success!"
            }
            # target_element_id is optional and not needed for SHOW_ALERT
        }
    ]
    
    # This node is responsible for setting the content that app.py will use.
    # 'output_content' is the preferred key, with 'ui_actions' nested within.
    # 'response' (formerly 'text_for_tts') can also be included if TTS is desired for this step.
    return {
        "output_content": {
            "response": text_response_for_tts, # Updated from text_for_tts
            "ui_actions": ui_actions
        }
    }
