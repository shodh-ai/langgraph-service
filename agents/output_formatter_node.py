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
        logger.info("Vertex AI initialized in output_formatter_node")
except Exception as e:
    logger.error(f"Failed to initialize Vertex AI in output_formatter_node: {e}")

try:
    gemini_model = GenerativeModel("gemini-1.5-pro")
    logger.info("Gemini model loaded in output_formatter_node")
except Exception as e:
    logger.error(f"Failed to load Gemini model in output_formatter_node: {e}")
    gemini_model = None

async def format_final_output_for_client_node(state: AgentGraphState) -> dict:
    """
    Formats the final output to be sent to the client application.
    Standardizes response format, ensures all necessary fields are present,
    and optimizes for the client's display capabilities.
    
    Args:
        state (AgentGraphState): The current state of the agent graph
        
    Returns:
        dict: Contains the formatted_output with standardized structure
    """
    context = state.get("current_context", {})
    feedback = state.get("feedback_content", {})
    motivation = state.get("motivational_message", "")
    teaching = state.get("teaching_content", {})
    practice = state.get("next_practice", {})
    next_step = state.get("next_task_details", {})
    
    logger.info(f"OutputFormatterNode: Formatting final output for user: {state.get('user_id', 'unknown')}")
    
    # Start building the formatted output
    formatted_output = {
        "text_for_tts": "",
        "frontend_rpc_calls": []
    }
    
    # Combine all frontend_rpc_calls from various nodes
    rpc_calls = []
    
    # Add feedback RPC calls if available
    if isinstance(feedback, dict) and "frontend_rpc_calls" in feedback:
        rpc_calls.extend(feedback.get("frontend_rpc_calls", []))
        formatted_output["text_for_tts"] += feedback.get("text_for_tts", "") + " "
    
    # Add teaching content RPC calls if available
    if isinstance(teaching, dict) and "frontend_rpc_calls" in teaching:
        rpc_calls.extend(teaching.get("frontend_rpc_calls", []))
        if not formatted_output["text_for_tts"]:  # Only add if no feedback text
            formatted_output["text_for_tts"] += teaching.get("text_for_tts", "") + " "
    
    # Add practice RPC calls if available
    if isinstance(practice, dict) and "frontend_rpc_calls" in practice:
        rpc_calls.extend(practice.get("frontend_rpc_calls", []))
        if not formatted_output["text_for_tts"]:  # Only add if no previous text
            formatted_output["text_for_tts"] += practice.get("text_for_tts", "") + " "
    
    # Add motivational message as text if available
    if motivation:
        formatted_output["text_for_tts"] += motivation + " "
        rpc_calls.append({"type": "DISPLAY_TEXT", "payload": {"text": motivation}})
    
    # If we have next step information, add it
    if isinstance(next_step, dict) and "next_step" in next_step:
        next_step_text = f"Next, we'll {next_step.get('next_step', 'continue with your learning')}."
        formatted_output["text_for_tts"] += next_step_text
        rpc_calls.append({"type": "DISPLAY_NEXT_STEP", "payload": {"text": next_step_text}})
    
    # If no text has been added yet, use a default message
    if not formatted_output["text_for_tts"].strip():
        formatted_output["text_for_tts"] = "Let's continue with your TOEFL preparation."
        rpc_calls.append({"type": "DISPLAY_TEXT", "payload": {"text": "Let's continue with your TOEFL preparation."}})
    
    # Add all collected RPC calls to the formatted output
    formatted_output["frontend_rpc_calls"] = rpc_calls
    
    # If Gemini is available, we can optionally use it to polish the output
    if gemini_model:
        try:
            # Get the appropriate prompt template
            system_prompt = PROMPTS.get("output_formatter", {}).get("system_prompt", 
                """You are an expert at formatting educational content for display.
                Review the combined output and ensure it flows naturally.
                Maintain all the instructional content but improve the transitions between sections.
                Format your response as a JSON with:
                {
                    "text_for_tts": "The polished text to be read aloud to the student",
                    "frontend_rpc_calls": [Array of existing RPC calls]
                }
                """)
            
            # Replace placeholders with actual data
            system_prompt = system_prompt.replace("{{formatted_output}}", json.dumps(formatted_output))
            
            user_prompt = PROMPTS.get("output_formatter", {}).get("user_prompt", 
                "Please polish this output to ensure it flows naturally for the student.")
            
            # Create content for the model
            contents = [
                Content(role="user", parts=[system_prompt]),
                Content(role="model", parts=["I'll polish the output for better flow."]),
                Content(role="user", parts=[user_prompt])
            ]
            
            # Call the Gemini model
            response = gemini_model.generate_content(contents, generation_config={
                "temperature": 0.3,  # Lower temperature for more consistent output
                "max_output_tokens": 1024
            })
            
            response_text = response.text
            
            # Parse the JSON response
            try:
                # Extract JSON if it's embedded in a code block
                if "```json" in response_text and "```" in response_text:
                    json_text = response_text.split("```json")[1].split("```")[0].strip()
                    polished_output = json.loads(json_text)
                    
                    # Only update the text_for_tts, keep the original RPC calls
                    if "text_for_tts" in polished_output:
                        formatted_output["text_for_tts"] = polished_output["text_for_tts"]
                        
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for output polishing: {e}")
                # Continue with the original formatted output
                
        except Exception as e:
            logger.error(f"Error in polishing output: {e}")
            # Continue with the original formatted output
    
    # Add metadata about the task and context
    formatted_output["metadata"] = {
        "task_id": getattr(context, 'task_id', 'unknown'),
        "toefl_section": getattr(context, 'toefl_section', 'unknown'),
        "task_stage": getattr(context, 'task_stage', 'unknown'),
        "timestamp": getattr(context, 'timestamp', '')
    }
    
    return {"formatted_output": formatted_output}
