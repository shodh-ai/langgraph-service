import logging
from state import AgentGraphState
import yaml
import os

logger = logging.getLogger(__name__)
PROMPTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "llm_prompts.yaml")

try:
    with open(PROMPTS_PATH, 'r') as file:
        PROMPTS = yaml.safe_load(file)
except Exception as e:
    logger.error(f"Failed to load LLM prompts: {e}")
    PROMPTS = {}

async def generate_feedback_for_task_node(state: AgentGraphState) -> dict:
    diagnosis = state.get("diagnosis_result", {})
    persona = state.get("current_teacher_persona", "nurturer")
    student_data = state.get("student_memory_context", {})
    context = state.get("current_context")
    
    logger.info(f"FeedbackGeneratorNode: Generating feedback with persona '{persona}'")
    
    #stub implementation
    feedback = {
        "text": f"I noticed your {diagnosis.get('primary_strength', 'response')} was particularly strong. " +
                f"One area to focus on might be {diagnosis.get('primary_error', 'continuing to practice')}.",
        "frontend_rpc_calls": [
            {
                "function_name": "highlightText",
                "args": ["#transcript_display", "Your vocabulary choice here was excellent!", 10, 25, "green"]
            },
            {
                "function_name": "showTip",
                "args": ["Consider using more complex sentence structures to express your ideas."]
            }
        ]
    }
    
    if persona == "structuralist":
        feedback["text"] += " Let's look at how we can organize your ideas more effectively."
    elif persona == "challenger":
        feedback["text"] += " I'd like you to think more deeply about how you could strengthen your argument."
    elif persona == "socratic":
        feedback["text"] += " What do you think would make your response even stronger?"
    
    logger.info(f"FeedbackGeneratorNode: Completed feedback generation")
    
    return {"feedback_content": feedback}