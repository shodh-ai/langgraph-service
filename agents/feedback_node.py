import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def generate_feedback_stub_node(state: AgentGraphState) -> dict:
    diagnosis = state.get("diagnosis_result", {})
    user_id = state['user_id']
    logger.info(f"FeedbackNodeStub: Generating feedback for user_id: {user_id}, Diagnosis: {diagnosis}")
    response_text = "FeedbackStub: Default feedback. Everything seems okay!"
    ui_actions = [] # Initialize with an empty list

    if diagnosis.get("errors"):
        error_detail = diagnosis["errors"][0].get("details", "a problem")
        response_text = f"FeedbackStub: I noticed {error_detail}. Let's review this aspect."
        ui_actions = [
            {
                "action_type": "HIGHLIGHT_ELEMENT", 
                "target_element_id": "#transcript_display", 
                "parameters": {"color": "yellow", "duration_ms": 3000}
            }
        ]
    elif diagnosis.get("needs_assistance"):
        response_text = "FeedbackStub: It sounds like you're looking for some help. I can guide you through this."
        ui_actions = [
            {
                "action_type": "SHOW_MODAL", 
                "parameters": {"modal_type": "help", "content_key": "current_task_help"}
            }
        ]
    elif diagnosis.get("strengths"):
        strength_detail = diagnosis["strengths"][0]
        response_text = f"FeedbackStub: Good job on {strength_detail.replace('stub_speaking_', '')}! Keep it up."
        ui_actions = [
            {
                "action_type": "SHOW_ALERT", 
                "parameters": {"message": "Excellent work!"}
            }
        ]

    return {
        "output_content": {
            "response": response_text,
            "ui_actions": ui_actions
        }
    }

