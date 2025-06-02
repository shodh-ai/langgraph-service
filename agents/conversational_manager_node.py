import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def handle_home_greeting_node(state: AgentGraphState) -> dict:
    """
    Creates a generic greeting for the home screen.
    Sets output_content with a greeting message using the student's name if available.
    
    Args:
        state: The current agent graph state
        
    Returns:
        Dict with output_content update
    """
    # Extract student name from memory context if available
    student_name = "student"
    if state.get("student_memory_context") and isinstance(state["student_memory_context"], dict):
        student_name = state["student_memory_context"].get("name", "student")
    
    logger.info(f"ConversationalManagerNode: Generating home greeting for {student_name}")
    
    # Create simple greeting output
    greeting_output = {
        "response": f"Hello {student_name}! Welcome to your TOEFL speaking practice session.", # Updated from text_for_tts
        "ui_actions": []  # Initialize as empty list, consistent with model
    }
    
    logger.info(f"ConversationalManagerNode: Generated greeting: {greeting_output['response']}")
    
    # Return state update
    return {"output_content": greeting_output}
