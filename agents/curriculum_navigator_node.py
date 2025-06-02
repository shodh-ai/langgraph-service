import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def determine_next_pedagogical_step_stub_node(state: AgentGraphState) -> dict:
    """
    Stub implementation that determines the next task for the student.
    Sets next_task_details and output_content with information about a speaking task.
    
    Args:
        state: The current agent graph state
        
    Returns:
        Dict with next_task_details and output_content updates
    """
    logger.info("CurriculumNavigatorNode: Determining next pedagogical step (stub)")
    
    # Hardcoded first speaking task details
    next_task = {
        "type": "SPEAKING",
        "question_type": "Q1", 
        "prompt_id": "SPK_Q1_P1_001",
        "title": "Your Favorite City",
        "description": "Describe a city you have visited that you particularly enjoyed. Explain what you liked about it and why you would recommend others to visit.",
        "prep_time_seconds": 15,
        "response_time_seconds": 45
    }
    
    logger.info(f"CurriculumNavigatorNode: Selected task: {next_task['title']} ({next_task['prompt_id']})")
    
    # Create output content to inform user about the task
    output_content = {
        "response": f"Let's practice with a speaking task about '{next_task['title']}'. You'll have {next_task['prep_time_seconds']} seconds to prepare and {next_task['response_time_seconds']} seconds to respond.", # Updated from text_for_tts
        "ui_actions": [
            {
                "action_type": "DISPLAY_NEXT_TASK_BUTTON",
                "parameters": next_task # Updated from payload
            }
        ]
    }
    
    logger.info(f"CurriculumNavigatorNode: Generated output with task information")
    
    # Return state updates
    return {
        "next_task_details": next_task,
        "output_content": output_content
    }
