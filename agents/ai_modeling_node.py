import logging
from state import AgentGraphState
import yaml
import os
import json
# Try to import vertexai, but don't fail if it's not available
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Content
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
logger.info(f"Using model: {type(gemini_model).__name__} in ai_modeling_node")

async def prepare_speaking_model_response_node(state: AgentGraphState) -> dict:
    """
    Prepares an exemplary speaking response that models good speaking practices.
    This serves as a demonstration for students to learn from.
    
    Args:
        state (AgentGraphState): The current state of the agent graph
        
    Returns:
        dict: Contains the model_speaking_response with audio transcript and analysis
    """
    context = state.get("current_context", {})
    task_prompt = state.get("task_prompt", {})
    rubric = state.get("rubric_details", {})
    student_data = state.get("student_memory_context", {})
    
    logger.info(f"AIModelingNode: Preparing speaking model response for task: {context.get('task_id', 'unknown')}")
    
    if not gemini_model:
        logger.warning("AIModelingNode: Gemini model not available, using stub implementation")
        # Fallback to stub implementation
        model_response = {
            "transcript": "This is a model speaking response that demonstrates good fluency, pronunciation, and organization.",
            "analysis": {
                "strengths": ["Clear organization", "Good fluency", "Appropriate vocabulary"],
                "key_techniques": ["Use of transitional phrases", "Natural intonation", "Concise examples"]
            },
            "frontend_rpc_calls": [
                {"type": "DISPLAY_TEXT", "payload": {"text": "Here's a model response for this speaking task."}}
            ]
        }
        return {"model_speaking_response": model_response}
    
    try:
        # Get the appropriate prompt template
        system_prompt = PROMPTS.get("speaking_model", {}).get("system_prompt", 
            """You are an expert TOEFL speaking tutor creating a model speaking response.
            Create a high-quality speaking response based on the task prompt and rubric.
            The response should demonstrate excellent speaking skills that align with TOEFL scoring criteria.
            Include natural speech patterns, appropriate vocabulary, and clear organization.
            Format your response as a JSON with:
            {
                "transcript": "The full transcript of the model speaking response",
                "analysis": {
                    "strengths": ["List of strengths in this response"],
                    "key_techniques": ["List of techniques demonstrated"]
                },
                "frontend_rpc_calls": [
                    {"type": "DISPLAY_TEXT", "payload": {"text": "Text to display"}},
                    {"type": "HIGHLIGHT_TECHNIQUE", "payload": {"technique": "technique name", "example": "example from transcript"}}
                ]
            }
            """)
        
        # Replace placeholders with actual data
        system_prompt = system_prompt.replace("{{task_prompt}}", json.dumps(task_prompt))
        system_prompt = system_prompt.replace("{{rubric_details}}", json.dumps(rubric))
        system_prompt = system_prompt.replace("{{student_memory_context}}", json.dumps(student_data))
        
        user_prompt = PROMPTS.get("speaking_model", {}).get("user_prompt", 
            "Please create a model speaking response for this TOEFL task that demonstrates excellent speaking skills.")
        
        # Create content for the model
        contents = [
            Content(role="user", parts=[system_prompt]),
            Content(role="model", parts=["I'll create an exemplary speaking response."]),
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
                model_response = json.loads(json_text)
            elif response_text.strip().startswith("{") and response_text.strip().endswith("}"):
                model_response = json.loads(response_text)
            else:
                # If not in JSON format, create a simple structure
                model_response = {
                    "transcript": response_text,
                    "analysis": {
                        "strengths": ["Clear organization", "Good fluency", "Appropriate vocabulary"],
                        "key_techniques": ["Use of transitional phrases", "Natural intonation", "Concise examples"]
                    },
                    "frontend_rpc_calls": [
                        {"type": "DISPLAY_TEXT", "payload": {"text": "Here's a model response for this speaking task."}}
                    ]
                }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            model_response = {
                "transcript": response_text,
                "analysis": {
                    "strengths": ["Clear organization", "Good fluency", "Appropriate vocabulary"],
                    "key_techniques": ["Use of transitional phrases", "Natural intonation", "Concise examples"]
                },
                "frontend_rpc_calls": [
                    {"type": "DISPLAY_TEXT", "payload": {"text": "Here's a model response for this speaking task."}}
                ]
            }
        
        return {"model_speaking_response": model_response}
        
    except Exception as e:
        logger.error(f"Error in prepare_speaking_model_response_node: {e}")
        # Fallback response
        model_response = {
            "transcript": "This is a model speaking response that demonstrates good fluency, pronunciation, and organization.",
            "analysis": {
                "strengths": ["Clear organization", "Good fluency", "Appropriate vocabulary"],
                "key_techniques": ["Use of transitional phrases", "Natural intonation", "Concise examples"]
            },
            "frontend_rpc_calls": [
                {"type": "DISPLAY_TEXT", "payload": {"text": "Here's a model response for this speaking task."}}
            ]
        }
        return {"model_speaking_response": model_response}

async def prepare_writing_model_response_node(state: AgentGraphState) -> dict:
    """
    Prepares an exemplary writing response that models good writing practices.
    This serves as a demonstration for students to learn from.
    
    Args:
        state (AgentGraphState): The current state of the agent graph
        
    Returns:
        dict: Contains the model_writing_response with essay text and analysis
    """
    context = state.get("current_context", {})
    task_prompt = state.get("task_prompt", {})
    rubric = state.get("rubric_details", {})
    student_data = state.get("student_memory_context", {})
    
    logger.info(f"AIModelingNode: Preparing writing model response for task: {context.get('task_id', 'unknown')}")
    
    if not gemini_model:
        logger.warning("AIModelingNode: Gemini model not available, using stub implementation")
        # Fallback to stub implementation
        model_response = {
            "essay_text": "This is a model essay that demonstrates good organization, coherence, and language use.",
            "analysis": {
                "strengths": ["Clear thesis statement", "Logical organization", "Varied sentence structure"],
                "key_techniques": ["Use of transitional phrases", "Effective examples", "Strong conclusion"]
            },
            "frontend_rpc_calls": [
                {"type": "DISPLAY_TEXT", "payload": {"text": "Here's a model essay for this writing task."}}
            ]
        }
        return {"model_writing_response": model_response}
    
    try:
        # Get the appropriate prompt template
        system_prompt = PROMPTS.get("writing_model", {}).get("system_prompt", 
            """You are an expert TOEFL writing tutor creating a model essay response.
            Create a high-quality essay based on the task prompt and rubric.
            The essay should demonstrate excellent writing skills that align with TOEFL scoring criteria.
            Include clear organization, strong thesis, supporting evidence, and varied sentence structure.
            Format your response as a JSON with:
            {
                "essay_text": "The full text of the model essay",
                "analysis": {
                    "strengths": ["List of strengths in this essay"],
                    "key_techniques": ["List of techniques demonstrated"]
                },
                "frontend_rpc_calls": [
                    {"type": "DISPLAY_TEXT", "payload": {"text": "Text to display"}},
                    {"type": "HIGHLIGHT_PARAGRAPH", "payload": {"paragraph": "paragraph text", "annotation": "annotation text"}}
                ]
            }
            """)
        
        # Replace placeholders with actual data
        system_prompt = system_prompt.replace("{{task_prompt}}", json.dumps(task_prompt))
        system_prompt = system_prompt.replace("{{rubric_details}}", json.dumps(rubric))
        system_prompt = system_prompt.replace("{{student_memory_context}}", json.dumps(student_data))
        
        user_prompt = PROMPTS.get("writing_model", {}).get("user_prompt", 
            "Please create a model essay for this TOEFL task that demonstrates excellent writing skills.")
        
        # Create content for the model
        contents = [
            Content(role="user", parts=[system_prompt]),
            Content(role="model", parts=["I'll create an exemplary essay response."]),
            Content(role="user", parts=[user_prompt])
        ]
        
        # Call the Gemini model
        response = gemini_model.generate_content(contents, generation_config={
            "temperature": 0.7,
            "max_output_tokens": 2048
        })
        
        response_text = response.text
        
        # Parse the JSON response
        try:
            # Extract JSON if it's embedded in a code block
            if "```json" in response_text and "```" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
                model_response = json.loads(json_text)
            elif response_text.strip().startswith("{") and response_text.strip().endswith("}"):
                model_response = json.loads(response_text)
            else:
                # If not in JSON format, create a simple structure
                model_response = {
                    "essay_text": response_text,
                    "analysis": {
                        "strengths": ["Clear thesis statement", "Logical organization", "Varied sentence structure"],
                        "key_techniques": ["Use of transitional phrases", "Effective examples", "Strong conclusion"]
                    },
                    "frontend_rpc_calls": [
                        {"type": "DISPLAY_TEXT", "payload": {"text": "Here's a model essay for this writing task."}}
                    ]
                }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            model_response = {
                "essay_text": response_text,
                "analysis": {
                    "strengths": ["Clear thesis statement", "Logical organization", "Varied sentence structure"],
                    "key_techniques": ["Use of transitional phrases", "Effective examples", "Strong conclusion"]
                },
                "frontend_rpc_calls": [
                    {"type": "DISPLAY_TEXT", "payload": {"text": "Here's a model essay for this writing task."}}
                ]
            }
        
        return {"model_writing_response": model_response}
        
    except Exception as e:
        logger.error(f"Error in prepare_writing_model_response_node: {e}")
        # Fallback response
        model_response = {
            "essay_text": "This is a model essay that demonstrates good organization, coherence, and language use.",
            "analysis": {
                "strengths": ["Clear thesis statement", "Logical organization", "Varied sentence structure"],
                "key_techniques": ["Use of transitional phrases", "Effective examples", "Strong conclusion"]
            },
            "frontend_rpc_calls": [
                {"type": "DISPLAY_TEXT", "payload": {"text": "Here's a model essay for this writing task."}}
            ]
        }
        return {"model_writing_response": model_response}
