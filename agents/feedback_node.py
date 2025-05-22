import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def generate_feedback_stub_node(state: AgentGraphState) -> dict:
    diagnosis = state.get("diagnosis_result", {})
    user_id = state['user_id']
    logger.info(f"FeedbackNodeStub: Generating feedback for user_id: {user_id}, Diagnosis: {diagnosis}")
    text_response = "FeedbackStub: Default feedback. Everything seems okay!"
    dom_actions = None # Optional DOM actions

    if diagnosis.get("errors"):
        error_detail = diagnosis["errors"][0].get("details", "a problem")
        text_response = f"FeedbackStub: I noticed {error_detail}. Let's review this aspect."
        dom_actions = [{"action": "highlight_error", "payload": {"text_to_highlight": error_detail, "element_id": "transcript_display"}}]
    elif diagnosis.get("needs_assistance"):
        text_response = "FeedbackStub: It sounds like you're looking for some help. I can guide you through this."
        dom_actions = [{"action": "show_help_modal", "payload": {"topic_id": "current_task_help"}}]
    elif diagnosis.get("strengths"):
        strength_detail = diagnosis["strengths"][0]
        text_response = f"FeedbackStub: Good job on {strength_detail.replace('stub_speaking_', '')}! Keep it up."
        dom_actions = [{"action": "show_positive_reinforcement", "payload": {"message": "Excellent work!"}}]
        
    return {"feedback_content": {"text": text_response, "dom_actions": dom_actions}}
