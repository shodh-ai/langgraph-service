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

async def generate_socratic_question_node(state: AgentGraphState) -> dict:
    diagnosis = state.get("diagnosis_result", {})
    transcript = state.get("transcript", "")
    task_prompt = state.get("task_prompt", {})
    
    logger.info(f"SocraticQuestioningNode: Generating Socratic questions")
    
    #stub implementation
    questions = [
        f"How might you expand on your point about {diagnosis.get('primary_strength', 'this topic')}?",
        f"What evidence could strengthen your argument about {task_prompt.get('main_topic', 'this issue')}?",
        f"Have you considered alternative perspectives on {task_prompt.get('main_topic', 'this matter')}?"
    ]
    
    logger.info(f"SocraticQuestioningNode: Generated {len(questions)} questions")
    
    return {"socratic_questions": questions}