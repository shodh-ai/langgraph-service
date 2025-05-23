import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def generate_feedback_stub_node(state: AgentGraphState) -> dict:
    diagnosis = state.get("diagnosis_result", {})
    user_id = state['user_id']
    logger.info(f"FeedbackNodeStub: Generating feedback for user_id: {user_id}, Diagnosis: {diagnosis}")
    text_response = "FeedbackStub: Default feedback. Everything seems okay!"
    frontend_rpc_calls = None

    if diagnosis.get("errors"):
        error_detail = diagnosis["errors"][0].get("details", "a problem")
        text_response = f"FeedbackStub: I noticed {error_detail}. Let's review this aspect."
        frontend_rpc_calls = [{"function_name": "highlightElement", "args": ["#transcript_display", "yellow", 3000]}]
    elif diagnosis.get("needs_assistance"):
        text_response = "FeedbackStub: It sounds like you're looking for some help. I can guide you through this."
        frontend_rpc_calls = [{"function_name": "showHelpModal", "args": ["current_task_help"]}]
    elif diagnosis.get("strengths"):
        strength_detail = diagnosis["strengths"][0]
        text_response = f"FeedbackStub: Good job on {strength_detail.replace('stub_speaking_', '')}! Keep it up."
        frontend_rpc_calls = [{"function_name": "showAlert", "args": ["Excellent work!"]}]

    return {"feedback_content": {"text": text_response, "frontend_rpc_calls": frontend_rpc_calls}}

