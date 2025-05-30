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
        logger.info("Vertex AI initialized in teaching_delivery_node")
except Exception as e:
    logger.error(f"Failed to initialize Vertex AI in teaching_delivery_node: {e}")

try:
    gemini_model = GenerativeModel("gemini-1.5-pro")
    logger.info("Gemini model loaded in teaching_delivery_node")
except Exception as e:
    logger.error(f"Failed to load Gemini model in teaching_delivery_node: {e}")
    gemini_model = None

async def deliver_teaching_module_node(state: AgentGraphState) -> dict:
    """
    Delivers teaching content based on the current context and student needs.
    Takes teaching materials and transforms them into engaging, personalized content.
    
    Args:
        state (AgentGraphState): The current state of the agent graph
        
    Returns:
        dict: Contains the teaching_content with text_for_tts and frontend_rpc_calls
    """
    context = state.get("current_context", {})
    student_data = state.get("student_memory_context", {})
    teaching_material = state.get("teaching_material", {})
    active_persona = state.get("active_persona", {})
    
    logger.info(f"TeachingDeliveryNode: Delivering teaching module for context: {context.get('task_id', 'unknown')}")
    
    if not gemini_model:
        logger.warning("TeachingDeliveryNode: Gemini model not available, using stub implementation")
        # Fallback to stub implementation
        teaching_content = {
            "text_for_tts": "Here's a lesson on this topic. [Teaching content would be delivered here]",
            "frontend_rpc_calls": [
                {"type": "DISPLAY_TEXT", "payload": {"text": "Here's a lesson on this topic."}}
            ]
        }
        return {"teaching_content": teaching_content}
    
    try:
        # Get the appropriate prompt template
        system_prompt = PROMPTS.get("teaching_delivery", {}).get("system_prompt", 
            """You are an expert TOEFL tutor delivering a teaching module.
            Use the teaching material to create an engaging lesson for the student.
            Consider the student's profile and adapt your teaching style accordingly.
            Format your response as a JSON with:
            {
                "text_for_tts": "The text to be read aloud to the student",
                "frontend_rpc_calls": [
                    {"type": "DISPLAY_TEXT", "payload": {"text": "Text to display"}},
                    {"type": "DISPLAY_IMAGE", "payload": {"url": "image_url", "alt_text": "description"}},
                    {"type": "HIGHLIGHT_KEY_POINT", "payload": {"text": "Important point to highlight"}}
                ]
            }
            """)
        
        # Replace placeholders with actual data
        system_prompt = system_prompt.replace("{{teaching_material}}", json.dumps(teaching_material))
        system_prompt = system_prompt.replace("{{student_memory_context}}", json.dumps(student_data))
        system_prompt = system_prompt.replace("{{active_persona}}", json.dumps(active_persona))
        
        user_prompt = PROMPTS.get("teaching_delivery", {}).get("user_prompt", 
            "Please deliver a teaching module on this topic in an engaging way.")
        
        # Create content for the model
        contents = [
            Content(role="user", parts=[system_prompt]),
            Content(role="model", parts=["I'll create an engaging teaching module based on the material."]),
            Content(role="user", parts=[user_prompt])
        ]
        
        # Call the Gemini model
        response = gemini_model.generate_content(contents, generation_config={
            "temperature": 0.7,
            "max_output_tokens": 1024
        })
        
        response_text = response.text
        
        # Parse the JSON response
        try:
            # Extract JSON if it's embedded in a code block
            if "```json" in response_text and "```" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
                teaching_content = json.loads(json_text)
            elif response_text.strip().startswith("{") and response_text.strip().endswith("}"):
                teaching_content = json.loads(response_text)
            else:
                # If not in JSON format, create a simple structure
                teaching_content = {
                    "text_for_tts": response_text,
                    "frontend_rpc_calls": [
                        {"type": "DISPLAY_TEXT", "payload": {"text": response_text}}
                    ]
                }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            teaching_content = {
                "text_for_tts": "I've prepared a lesson on this topic.",
                "frontend_rpc_calls": [
                    {"type": "DISPLAY_TEXT", "payload": {"text": response_text}}
                ]
            }
        
        return {"teaching_content": teaching_content}
        
    except Exception as e:
        logger.error(f"Error in deliver_teaching_module_node: {e}")
        # Fallback response
        teaching_content = {
            "text_for_tts": "Here's a lesson on this topic. [Teaching content would be delivered here]",
            "frontend_rpc_calls": [
                {"type": "DISPLAY_TEXT", "payload": {"text": "Here's a lesson on this topic."}}
            ]
        }
        return {"teaching_content": teaching_content}

async def manage_skill_drill_node(state: AgentGraphState) -> dict:
    """
    Manages the delivery and interaction with skill drills or practice exercises.
    Adapts the drill based on student performance and provides guidance.
    
    Args:
        state (AgentGraphState): The current state of the agent graph
        
    Returns:
        dict: Contains the skill_drill_content with instructions and interactive elements
    """
    context = state.get("current_context", {})
    student_data = state.get("student_memory_context", {})
    skill_drill = state.get("skill_drill_content", {})
    active_persona = state.get("active_persona", {})
    
    logger.info(f"TeachingDeliveryNode: Managing skill drill for context: {context.get('task_id', 'unknown')}")
    
    if not gemini_model:
        logger.warning("TeachingDeliveryNode: Gemini model not available, using stub implementation")
        # Fallback to stub implementation
        drill_content = {
            "text_for_tts": "Let's practice this skill with a focused exercise.",
            "frontend_rpc_calls": [
                {"type": "DISPLAY_TEXT", "payload": {"text": "Let's practice this skill with a focused exercise."}},
                {"type": "DISPLAY_EXERCISE", "payload": {"exercise_id": "default_exercise"}}
            ]
        }
        return {"skill_drill_content": drill_content}
    
    try:
        # Get the appropriate prompt template
        system_prompt = PROMPTS.get("skill_drill_management", {}).get("system_prompt", 
            """You are an expert TOEFL tutor managing a skill drill exercise.
            Use the skill drill content to create an interactive practice session.
            Adapt the difficulty based on the student's profile and previous performance.
            Format your response as a JSON with:
            {
                "text_for_tts": "The text to be read aloud to the student",
                "frontend_rpc_calls": [
                    {"type": "DISPLAY_TEXT", "payload": {"text": "Text to display"}},
                    {"type": "DISPLAY_EXERCISE", "payload": {"exercise_id": "exercise_id", "instructions": "instructions"}},
                    {"type": "SHOW_TIMER", "payload": {"duration_seconds": 60}}
                ]
            }
            """)
        
        # Replace placeholders with actual data
        system_prompt = system_prompt.replace("{{skill_drill_content}}", json.dumps(skill_drill))
        system_prompt = system_prompt.replace("{{student_memory_context}}", json.dumps(student_data))
        system_prompt = system_prompt.replace("{{active_persona}}", json.dumps(active_persona))
        
        user_prompt = PROMPTS.get("skill_drill_management", {}).get("user_prompt", 
            "Please guide the student through this skill drill exercise.")
        
        # Create content for the model
        contents = [
            Content(role="user", parts=[system_prompt]),
            Content(role="model", parts=["I'll create an engaging skill drill exercise."]),
            Content(role="user", parts=[user_prompt])
        ]
        
        # Call the Gemini model
        response = gemini_model.generate_content(contents, generation_config={
            "temperature": 0.7,
            "max_output_tokens": 1024
        })
        
        response_text = response.text
        
        # Parse the JSON response
        try:
            # Extract JSON if it's embedded in a code block
            if "```json" in response_text and "```" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
                drill_content = json.loads(json_text)
            elif response_text.strip().startswith("{") and response_text.strip().endswith("}"):
                drill_content = json.loads(response_text)
            else:
                # If not in JSON format, create a simple structure
                drill_content = {
                    "text_for_tts": response_text,
                    "frontend_rpc_calls": [
                        {"type": "DISPLAY_TEXT", "payload": {"text": response_text}},
                        {"type": "DISPLAY_EXERCISE", "payload": {"exercise_id": "generated_exercise", "instructions": response_text}}
                    ]
                }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            drill_content = {
                "text_for_tts": "Let's practice this skill with a focused exercise.",
                "frontend_rpc_calls": [
                    {"type": "DISPLAY_TEXT", "payload": {"text": response_text}},
                    {"type": "DISPLAY_EXERCISE", "payload": {"exercise_id": "default_exercise", "instructions": response_text}}
                ]
            }
        
        return {"skill_drill_content": drill_content}
        
    except Exception as e:
        logger.error(f"Error in manage_skill_drill_node: {e}")
        # Fallback response
        drill_content = {
            "text_for_tts": "Let's practice this skill with a focused exercise.",
            "frontend_rpc_calls": [
                {"type": "DISPLAY_TEXT", "payload": {"text": "Let's practice this skill with a focused exercise."}},
                {"type": "DISPLAY_EXERCISE", "payload": {"exercise_id": "default_exercise"}}
            ]
        }
        return {"skill_drill_content": drill_content}
