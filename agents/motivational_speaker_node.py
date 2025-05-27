import logging
from state import AgentGraphState
import yaml
import os
import json
import vertexai
from vertexai.generative_models import GenerativeModel, Content

logger = logging.getLogger(__name__)
PROMPTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "llm_prompts.yaml")

try:
    with open(PROMPTS_PATH, 'r') as file:
        PROMPTS = yaml.safe_load(file)
except Exception as e:
    logger.error(f"Failed to load LLM prompts: {e}")
    PROMPTS = {}

try:
    if 'vertexai' in globals() and not hasattr(vertexai, '_initialized'):
        vertexai.init(project="windy-orb-460108-t0", location="us-central1")
        vertexai._initialized = True
        logger.info("Vertex AI initialized in motivational_speaker_node")
except Exception as e:
    logger.error(f"Failed to initialize Vertex AI in motivational_speaker_node: {e}")

try:
    gemini_model = GenerativeModel("gemini-1.5-pro")
    logger.info("Gemini model loaded in motivational_speaker_node")
except Exception as e:
    logger.error(f"Failed to load Gemini model in motivational_speaker_node: {e}")
    gemini_model = None

async def generate_motivational_message_node(state: AgentGraphState) -> dict:
    affect = state.get("student_affective_state", {})
    student_data = state.get("student_memory_context", {})
    diagnosis = state.get("diagnosis_result", {})
    feedback = state.get("feedback_content", {})
    
    logger.info(f"MotivationalSpeakerNode: Generating motivational message for affect: {affect.get('primary_emotion', 'neutral')}")
    
    if not gemini_model:
        logger.warning("MotivationalSpeakerNode: Gemini model not available, using stub implementation")
        # Fallback to stub implementation
        message = "You're making good progress! Keep up the great work."
        
        # Adjust message based on affect
        if affect.get("primary_emotion") == "frustrated":
            message = "I understand this can be challenging. Remember that making mistakes is part of the learning process. Let's break this down into smaller steps."
        elif affect.get("primary_emotion") == "satisfied":
            message = "I'm glad you're feeling confident! Your hard work is clearly paying off. Let's build on this momentum."
    else:
        try:
            # Get prompt templates from config
            system_prompt = PROMPTS.get("motivational", {}).get("message", {}).get("system", "")
            user_prompt = PROMPTS.get("motivational", {}).get("message", {}).get("user_template", "")
            
            # Replace placeholders in prompts
            system_prompt = system_prompt.replace("{{student_affective_state}}", json.dumps(affect))
            system_prompt = system_prompt.replace("{{student_memory_context}}", json.dumps(student_data))
            system_prompt = system_prompt.replace("{{diagnosis_result}}", json.dumps(diagnosis))
            
            # Replace specific placeholders in user prompt
            user_prompt = user_prompt.replace("{{student_affective_state.primary_emotion}}", affect.get("primary_emotion", "neutral"))
            
            # Create Gemini content
            contents = [
                Content(role="user", parts=[system_prompt]),
                Content(role="model", parts=["I understand. I'll generate a motivational message tailored to the student's emotional state and progress."]),
                Content(role="user", parts=[user_prompt])
            ]
            
            # Generate response from Gemini
            response = gemini_model.generate_content(contents, generation_config={
                "temperature": 0.7,
                "max_output_tokens": 256 
            })
            
            # Use the response directly
            message = response.text.strip()
            
            # Ensure the message is not too long
            if len(message) > 500:
                message = message[:497] + "..."
                
        except Exception as e:
            logger.error(f"MotivationalSpeakerNode: Error calling Gemini API: {e}")
            # Fallback to stub implementation
            message = "You're making good progress! Keep up the great work."
            
            # Adjust message based on affect
            if affect.get("primary_emotion") == "frustrated":
                message = "I understand this can be challenging. Remember that making mistakes is part of the learning process. Let's break this down into smaller steps."
            elif affect.get("primary_emotion") == "satisfied":
                message = "I'm glad you're feeling confident! Your hard work is clearly paying off. Let's build on this momentum."
    
    profile = student_data.get("profile", {})
    if profile.get("recent_improvements") and "improved" not in message.lower():
        message += f" I've noticed your {profile.get('recent_improvements')[0]} has really improved lately!"
    
    logger.info(f"MotivationalSpeakerNode: Generated motivational message of {len(message)} characters")
    
    return {"motivational_message": message}