import logging
import yaml
import os
import json
from state import AgentGraphState
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)

# Define the path to the prompts file
PROMPTS_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'llm_prompts.yaml')

# Load prompts once when the module is loaded
PROMPTS = {}
try:
    with open(PROMPTS_FILE_PATH, 'r') as f:
        loaded_prompts = yaml.safe_load(f)
        if loaded_prompts and 'PROMPTS' in loaded_prompts:
            PROMPTS = loaded_prompts['PROMPTS']
        else:
            logger.error(f"Could not find 'PROMPTS' key in {PROMPTS_FILE_PATH}")
except FileNotFoundError:
    logger.error(f"Prompts file not found at {PROMPTS_FILE_PATH}")
except yaml.YAMLError as e:
    logger.error(f"Error parsing YAML from {PROMPTS_FILE_PATH}: {e}")

async def handle_home_greeting_node(state: AgentGraphState) -> dict:
    """
    Generates a personalized welcome greeting using an LLM based on the student's name and persona.
    The LLM is expected to return a JSON object with a 'greeting_tts' field.
    
    Args:
        state: The current agent graph state
        
    Returns:
        Dict with output_content update, including the LLM-generated greeting.
    """
    student_name = "Harshit" # Temporarily hardcoded for testing
    logger.info(f"ConversationalManagerNode: Using hardcoded student_name: '{student_name}' for testing LLM call.")

    # logger.info(f"ConversationalManagerNode: Attempting to generate LLM-based home greeting for {student_name}") # Original log line

    greeting_prompt_config = PROMPTS.get('welcome_greeting')
    if not greeting_prompt_config:
        logger.error("Welcome greeting prompt configuration not found. Falling back to default.")
        return {"output_content": {"response": f"Hello {student_name}! Welcome.", "ui_actions": []}}

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment. Falling back to default greeting.")
        return {"output_content": {"response": f"Hello {student_name}! Welcome.", "ui_actions": []}}

    try:
        logger.debug("Attempting genai.configure()...")
        genai.configure(api_key=api_key)
        logger.debug("genai.configure() successful.")
        
        logger.debug(f"Attempting to initialize GenerativeModel: gemini-2.5-flash-preview-05-20")
        model = genai.GenerativeModel(
            'gemini-2.5-flash-preview-05-20',
            generation_config=GenerationConfig(
                response_mime_type="application/json"
            )
        )
        logger.debug("GenerativeModel initialized successfully.")

        persona_details = "Your friendly and encouraging AI guide, Rox."
        system_prompt_text = greeting_prompt_config.get('system_prompt', '').format(
            student_name=student_name,
            persona_details=persona_details
        )
        user_prompt_text = greeting_prompt_config.get('user_prompt', '')
        full_prompt = f"{system_prompt_text}\n\n{user_prompt_text}"
        
        logger.info(f"GOOGLE_API_KEY loaded: {api_key[:5]}...{api_key[-5:] if len(api_key) > 10 else ''}")
        logger.info(f"Full prompt for greeting LLM: {full_prompt}")
        
        raw_llm_response_text = ""
        try:
            logger.debug("Attempting model.generate_content_async()...")
            response = await model.generate_content_async(full_prompt)
            logger.debug("model.generate_content_async() successful.")
            raw_llm_response_text = response.text
            logger.info(f"Raw LLM Response for greeting: {raw_llm_response_text}")
        except Exception as gen_err:
            logger.error(f"Error during model.generate_content_async(): {gen_err}", exc_info=True)
            # Fallback if generate_content_async fails
            fallback_tts = f"Hello {student_name}! Welcome. (LLM Generation Error)"
            logger.info(f"ConversationalManagerNode: Using fallback greeting due to generation error: {fallback_tts}")
            return {"greeting_data": {"greeting_tts": fallback_tts}}

        try:
            llm_output_json = json.loads(raw_llm_response_text)
            greeting_tts = llm_output_json.get("greeting_tts", f"Hello {student_name}! I'm Rox, welcome! (JSON Key Missing)")
        except json.JSONDecodeError as json_err:
            logger.error(f"JSONDecodeError parsing LLM greeting response. Error: {json_err}. Raw text: {raw_llm_response_text}")
            greeting_tts = f"Hello {student_name}! I'm Rox. (JSON Parse Error)"
        except Exception as parse_err: # Catch any other parsing related error
            logger.error(f"Unexpected error parsing LLM greeting response. Error: {parse_err}. Raw text: {raw_llm_response_text}")
            greeting_tts = f"Hello {student_name}! I'm Rox. (Unexpected Parse Error)"

        logger.info(f"ConversationalManagerNode: LLM-generated greeting: {greeting_tts}")
        return {"greeting_data": {"greeting_tts": greeting_tts}}

    except Exception as e:
        logger.error(f"Outer error in handle_home_greeting_node (e.g., config, model init): {e}", exc_info=True)
        fallback_tts = f"Hello {student_name}! Welcome. (LLM Setup Error)"
        logger.info(f"ConversationalManagerNode: Using fallback greeting due to setup error: {fallback_tts}")
        return {"greeting_data": {"greeting_tts": fallback_tts}}
