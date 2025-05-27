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

async def generate_motivational_message_node(state: AgentGraphState) -> dict:
    affect = state.get("student_affective_state", {})
    student_data = state.get("student_memory_context", {})
    diagnosis = state.get("diagnosis_result", {})
    
    logger.info(f"MotivationalSpeakerNode: Generating motivational message for affect: {affect.get('primary_emotion', 'neutral')}")
    
    #stub implementation
    message = "You're making good progress! Keep up the great work."
    
    # adjust message based on affect
    if affect.get("primary_emotion") == "frustrated":
        message = "I understand this can be challenging. Remember that making mistakes is part of the learning process. Let's break this down into smaller steps."
    elif affect.get("primary_emotion") == "satisfied":
        message = "I'm glad you're feeling confident! Your hard work is clearly paying off. Let's build on this momentum."
    
    # personalize based on student history
    profile = student_data.get("profile", {})
    if profile.get("recent_improvements"):
        message += f" I've noticed your {profile.get('recent_improvements')[0]} has really improved lately!"
    
    logger.info(f"MotivationalSpeakerNode: Generated motivational message")
    
    return {"motivational_message": message}