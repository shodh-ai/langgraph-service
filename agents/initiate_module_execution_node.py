import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def initiate_module_execution_node(state: AgentGraphState) -> dict:
    """
    Announces the start of the new lesson plan and sends the visual data
    to the frontend for display.
    """
    logger.info("--- Announcing and Displaying New Lesson Plan ---")
    try:
        plan = state.get("pedagogical_plan")
        lesson_plan_graph_data = state.get("lesson_plan_graph_data")
        
        if not plan or not lesson_plan_graph_data:
            return {"final_text_for_tts": "Something went wrong creating your lesson plan. Let's try again."}
            
        announcement_text = f"Great! I've created a {len(plan)}-step plan for this topic. Here's what it looks like. We'll start with the first step: {plan[0].get('focus')}."

        return {
            "final_text_for_tts": announcement_text,
            "final_ui_actions": [
                {
                    "action_type": "DISPLAY_LESSON_PLAN", # A new, specific action type
                    "parameters": {
                        "map_data": lesson_plan_graph_data,
                        "map_title": "Our Plan for This Lesson"
                    }
                },
                # This action is now separate from displaying the plan
                {
                    "action_type": "SPEAK_THEN_LISTEN",
                    "parameters": {"text_to_speak": announcement_text}
                }
            ]
        }
    except Exception as e:
        logger.error(f"Initiate Module Execution: CRITICAL FAILURE: {e}", exc_info=True)
        return {
            "final_text_for_tts": "I encountered an error while starting our lesson. Please try again.",
            "final_ui_actions": []
        }
