# langgraph-service/agents/pedagogy_output_formatter.py
import logging
from state import AgentGraphState
from models import ReactUIAction

logger = logging.getLogger(__name__)

async def pedagogy_output_formatter_node(state: AgentGraphState) -> dict:
    """
    Formats the generated pedagogical suggestion into the final, client-ready
    text-to-speech script and UI actions.
    """
    logger.info("---PEDAGOGY OUTPUT FORMATTER---")

    pedagogy_content = state.get("pedagogy_content")

    if not pedagogy_content or not pedagogy_content.get("suggestion_text"):
        logger.warning("No valid pedagogy content found. Providing a fallback response.")
        return {
            "final_text_for_tts": "I seem to be having trouble finding the next step. Please try again or choose a task from the menu.",
            "final_ui_actions": []
        }

    # The text for TTS is the direct suggestion from the generator
    final_text = pedagogy_content.get("suggestion_text")
    
    # Create UI actions based on the next task details
    ui_actions = []
    next_task_details = pedagogy_content.get("next_task_details")

    if next_task_details and next_task_details.get("title"):
        # Create a button that allows the user to immediately start the suggested task
        ui_actions.append(
            ReactUIAction(
                action_type="ADD_TASK_BUTTON",
                parameters={
                    "button_text": f"Let's Start: {next_task_details['title']}",
                    "task_details": next_task_details
                }
            ).model_dump()
        )
        logger.info(f"Created 'ADD_TASK_BUTTON' for task: {next_task_details['title']}")
    else:
        logger.info("No specific next task details provided. No UI action will be created.")

    logger.info(f"Formatted pedagogy output. TTS: '{final_text[:70]}...'")
    
    return {
        "final_text_for_tts": final_text,
        "final_ui_actions": ui_actions
    }
