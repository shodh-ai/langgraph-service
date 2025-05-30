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
        logger.info("Vertex AI initialized in conversational_turn_manager_node")
except Exception as e:
    logger.error(f"Failed to initialize Vertex AI in conversational_turn_manager_node: {e}")

try:
    gemini_model = GenerativeModel("gemini-1.5-pro")
    logger.info("Gemini model loaded in conversational_turn_manager_node")
except Exception as e:
    logger.error(f"Failed to load Gemini model in conversational_turn_manager_node: {e}")
    gemini_model = None

async def process_conversational_turn_node(state: AgentGraphState) -> dict:
    """
    Processes conversational turns between the student and tutor.
    Manages dialogue flow, ensures coherence, and maintains context.
    
    Args:
        state (AgentGraphState): The current state of the agent graph
        
    Returns:
        dict: Contains the processed_turn with response and updated context
    """
    context = state.get("current_context", {})
    student_data = state.get("student_memory_context", {})
    active_persona = state.get("active_persona", {})
    conversation_history = state.get("conversation_history", [])
    current_message = state.get("current_message", "")
    
    logger.info(f"ConversationalTurnManagerNode: Processing turn for user: {state.get('user_id', 'unknown')}")
    
    if not gemini_model:
        logger.warning("ConversationalTurnManagerNode: Gemini model not available, using stub implementation")
        # Fallback to stub implementation
        processed_turn = {
            "response": "I understand what you're saying. Let's continue with the lesson.",
            "updated_context": context,
            "dialogue_state": "continuing",
            "frontend_rpc_calls": [
                {"type": "DISPLAY_TEXT", "payload": {"text": "I understand what you're saying. Let's continue with the lesson."}}
            ]
        }
        
        # Update conversation history
        updated_history = conversation_history.copy() if conversation_history else []
        updated_history.append({
            "role": "student",
            "content": current_message
        })
        updated_history.append({
            "role": "tutor",
            "content": processed_turn["response"]
        })
        
        return {
            "processed_turn": processed_turn,
            "conversation_history": updated_history
        }
    
    try:
        # Get the appropriate prompt template
        system_prompt = PROMPTS.get("conversational_turn", {}).get("system_prompt", 
            """You are an expert TOEFL tutor managing a conversation with a student.
            Process the current message in the context of the conversation history.
            Maintain coherence and context while advancing the pedagogical goals.
            Format your response as a JSON with:
            {
                "response": "The tutor's response to the student",
                "dialogue_state": "One of: greeting, continuing, questioning, clarifying, concluding",
                "updated_context": {
                    "topic": "Current topic of conversation",
                    "student_understanding": "Assessment of student understanding",
                    "next_focus": "What to focus on next"
                },
                "frontend_rpc_calls": [
                    {"type": "DISPLAY_TEXT", "payload": {"text": "Text to display"}}
                ]
            }
            """)
        
        # Replace placeholders with actual data
        system_prompt = system_prompt.replace("{{current_context}}", json.dumps(context))
        system_prompt = system_prompt.replace("{{student_memory_context}}", json.dumps(student_data))
        system_prompt = system_prompt.replace("{{active_persona}}", json.dumps(active_persona))
        system_prompt = system_prompt.replace("{{conversation_history}}", json.dumps(conversation_history))
        system_prompt = system_prompt.replace("{{current_message}}", current_message)
        
        user_prompt = PROMPTS.get("conversational_turn", {}).get("user_prompt", 
            "Please process this conversational turn and provide an appropriate response.")
        
        # Create content for the model
        contents = [
            Content(role="user", parts=[system_prompt]),
            Content(role="model", parts=["I'll process this conversational turn."]),
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
                processed_turn = json.loads(json_text)
            elif response_text.strip().startswith("{") and response_text.strip().endswith("}"):
                processed_turn = json.loads(response_text)
            else:
                # If not in JSON format, create a simple structure
                processed_turn = {
                    "response": response_text,
                    "dialogue_state": "continuing",
                    "updated_context": context,
                    "frontend_rpc_calls": [
                        {"type": "DISPLAY_TEXT", "payload": {"text": response_text}}
                    ]
                }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            processed_turn = {
                "response": "I understand what you're saying. Let's continue with the lesson.",
                "dialogue_state": "continuing",
                "updated_context": context,
                "frontend_rpc_calls": [
                    {"type": "DISPLAY_TEXT", "payload": {"text": "I understand what you're saying. Let's continue with the lesson."}}
                ]
            }
        
        # Update conversation history
        updated_history = conversation_history.copy() if conversation_history else []
        updated_history.append({
            "role": "student",
            "content": current_message
        })
        updated_history.append({
            "role": "tutor",
            "content": processed_turn["response"]
        })
        
        return {
            "processed_turn": processed_turn,
            "conversation_history": updated_history
        }
        
    except Exception as e:
        logger.error(f"Error in process_conversational_turn_node: {e}")
        # Fallback response
        processed_turn = {
            "response": "I understand what you're saying. Let's continue with the lesson.",
            "dialogue_state": "continuing",
            "updated_context": context,
            "frontend_rpc_calls": [
                {"type": "DISPLAY_TEXT", "payload": {"text": "I understand what you're saying. Let's continue with the lesson."}}
            ]
        }
        
        # Update conversation history even in case of error
        updated_history = conversation_history.copy() if conversation_history else []
        updated_history.append({
            "role": "student",
            "content": current_message
        })
        updated_history.append({
            "role": "tutor",
            "content": processed_turn["response"]
        })
        
        return {
            "processed_turn": processed_turn,
            "conversation_history": updated_history
        }
