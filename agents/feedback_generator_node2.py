import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def generate_speaking_feedback_stub_node(state: AgentGraphState) -> dict:
    """
    Stub implementation for generating speaking feedback based on diagnosis results.
    Creates output_content with text for TTS and UI actions for highlighting transcript.
    
    Args:
        state: The current agent graph state
        
    Returns:
        Dict with output_content update
    """
    # Get diagnosis result from state
    diagnosis = state.get("diagnosis_result", {})
    diagnosis_summary = diagnosis.get("summary", "No diagnosis available")
    
    logger.info(f"FeedbackGeneratorNode: Generating speaking feedback for user_id: {state['user_id']}")
    logger.info(f"FeedbackGeneratorNode: Using diagnosis summary: '{diagnosis_summary}'")
    
    # Extract scores if available
    scores = diagnosis.get("score", {})
    overall_score = scores.get("overall", "not available")
    
    # Create feedback content
    feedback_text = f"Stub Feedback for speaking: {diagnosis_summary} Your overall score is {overall_score}/5."
    
    # Add specific feedback points based on diagnosis
    if "errors" in diagnosis and diagnosis["errors"]:
        error_details = diagnosis["errors"][0].get("details", "")
        feedback_text += f" {error_details}"
    
    # Create UI actions for highlighting transcript
    ui_actions = [
        {
            "action_type": "HIGHLIGHT_TRANSCRIPT", 
            "parameters": {"area": "overall", "color": "yellow"} # Updated from payload
        },
        {
            "action_type": "SHOW_FEEDBACK_PANEL",
            "parameters": {"visible": True} # Updated from payload
        }
    ]
    
    # Set output content
    output_content = {
        "response": feedback_text, # Updated from text_for_tts
        "ui_actions": ui_actions
    }
    
    logger.info(f"FeedbackGeneratorNode: Generated feedback text: '{feedback_text}'")
    logger.info(f"FeedbackGeneratorNode: Added UI actions: {ui_actions}")
    
    return {"output_content": output_content}
