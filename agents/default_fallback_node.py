import logging
from state import AgentGraphState
import yaml
import os
import json
# Try to import vertexai, but don't fail if it's not available
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Content, Part
    vertexai_available = True
except ImportError:
    vertexai_available = False
    
# Import our fallback utilities
from utils.fallback_utils import get_model_with_fallback

logger = logging.getLogger(__name__)
PROMPTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "llm_prompts.yaml")

try:
    with open(PROMPTS_PATH, 'r') as file:
        PROMPTS = yaml.safe_load(file)
except Exception as e:
    logger.error(f"Failed to load LLM prompts: {e}")
    PROMPTS = {}

# Use our fallback utility to get a model (either real or fallback)
gemini_model = get_model_with_fallback("gemini-2.5-flash-preview-05-20")
logger.info(f"Using model: {type(gemini_model).__name__} in default_fallback_node")

async def handle_unmatched_interaction_node(state: AgentGraphState) -> dict:
    """
    Handles interactions that don't match any specific task stage or pattern.
    Provides a graceful fallback response and attempts to guide the conversation back on track.
    
    Args:
        state (AgentGraphState): The current state of the agent graph
        
    Returns:
        dict: Contains the fallback_response with appropriate guidance
    """
    context = state.get("current_context", {})
    student_data = state.get("student_memory_context", {})
    active_persona = state.get("active_persona", {})
    current_message = state.get("current_message", "")
    
    logger.info(f"DefaultFallbackNode: Handling unmatched interaction for user: {state.get('user_id', 'unknown')}")
    
    if not gemini_model:
        logger.warning("DefaultFallbackNode: Gemini model not available, using stub implementation")
        # Fallback to stub implementation
        fallback_response = {
            "text_for_tts": "I'm not sure I understand. Let's focus on your TOEFL preparation. Would you like to continue with your current task or try something else?",
            "frontend_rpc_calls": [
                {"type": "DISPLAY_TEXT", "payload": {"text": "I'm not sure I understand. Let's focus on your TOEFL preparation. Would you like to continue with your current task or try something else?"}}
            ],
            "suggested_next_steps": [
                "Continue current task",
                "Review previous feedback",
                "Try a different practice exercise"
            ]
        }
        return {"fallback_response": fallback_response}
    
    try:
        # Get the appropriate prompt template
        system_prompt = PROMPTS.get("default_fallback", {}).get("system_prompt", 
            """You are an expert TOEFL tutor handling an interaction that doesn't match expected patterns.
            Provide a helpful response that acknowledges the student's message and guides them back to productive learning.
            Consider the current context and the student's profile when crafting your response.
            Be empathetic but maintain focus on TOEFL preparation.
            Format your response as a JSON with:
            {
                "text_for_tts": "The text to be read aloud to the student",
                "frontend_rpc_calls": [
                    {"type": "DISPLAY_TEXT", "payload": {"text": "Text to display"}}
                ],
                "suggested_next_steps": ["List of 2-3 suggested actions the student could take"],
                "inferred_intent": "Your best guess at what the student was trying to accomplish"
            }
            """)
        
        # Replace placeholders with actual data
        system_prompt = system_prompt.replace("{{current_context}}", json.dumps(context))
        system_prompt = system_prompt.replace("{{student_memory_context}}", json.dumps(student_data))
        system_prompt = system_prompt.replace("{{active_persona}}", json.dumps(active_persona))
        system_prompt = system_prompt.replace("{{current_message}}", current_message)
        
        user_prompt = PROMPTS.get("default_fallback", {}).get("user_prompt", 
            "Please handle this unmatched interaction and guide the student back to productive learning.")
        
        # Create content for the model
        contents = [
            Content(role="user", parts=[system_prompt]),
            Content(role="model", parts=["I'll handle this unmatched interaction."]),
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
                fallback_response = json.loads(json_text)
            elif response_text.strip().startswith("{") and response_text.strip().endswith("}"):
                fallback_response = json.loads(response_text)
            else:
                # If not in JSON format, create a simple structure
                fallback_response = {
                    "text_for_tts": response_text,
                    "frontend_rpc_calls": [
                        {"type": "DISPLAY_TEXT", "payload": {"text": response_text}}
                    ],
                    "suggested_next_steps": [
                        "Continue with your current task",
                        "Try a different practice exercise"
                    ],
                    "inferred_intent": "General question or comment"
                }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            fallback_response = {
                "text_for_tts": "I'm not sure I understand. Let's focus on your TOEFL preparation. Would you like to continue with your current task or try something else?",
                "frontend_rpc_calls": [
                    {"type": "DISPLAY_TEXT", "payload": {"text": "I'm not sure I understand. Let's focus on your TOEFL preparation. Would you like to continue with your current task or try something else?"}}
                ],
                "suggested_next_steps": [
                    "Continue current task",
                    "Review previous feedback",
                    "Try a different practice exercise"
                ],
                "inferred_intent": "Unclear"
            }
        
        return {"fallback_response": fallback_response}
        
    except Exception as e:
        logger.error(f"Error in handle_unmatched_interaction_node: {e}")
        # Fallback response
        fallback_response = {
            "text_for_tts": "I'm not sure I understand. Let's focus on your TOEFL preparation. Would you like to continue with your current task or try something else?",
            "frontend_rpc_calls": [
                {"type": "DISPLAY_TEXT", "payload": {"text": "I'm not sure I understand. Let's focus on your TOEFL preparation. Would you like to continue with your current task or try something else?"}}
            ],
            "suggested_next_steps": [
                "Continue current task",
                "Review previous feedback",
                "Try a different practice exercise"
            ],
            "inferred_intent": "Unclear"
        }
        return {"fallback_response": fallback_response}
