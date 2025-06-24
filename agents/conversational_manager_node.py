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
    Generates a personalized welcome greeting using an LLM and returns it in the
    standardized 'final_flow_output' format.
    """
    student_name = "Harshit" # Temporarily hardcoded for testing
    logger.info(f"ConversationalManagerNode: Using hardcoded student_name: '{student_name}' for testing LLM call.")

    greeting_prompt_config = PROMPTS.get('welcome_greeting')
    if not greeting_prompt_config:
        logger.error("Welcome greeting prompt configuration not found. Falling back to default.")
        fallback_output = {
            "text_for_tts": f"Hello {student_name}! Welcome.",
            "ui_actions": []
        }
        return {"final_flow_output": fallback_output}

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment. Falling back to default greeting.")
        fallback_output = {
            "text_for_tts": f"Hello {student_name}! Welcome.",
            "ui_actions": []
        }
        return {"final_flow_output": fallback_output}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            'gemini-2.0-flash',
            generation_config=GenerationConfig(response_mime_type="application/json")
        )

        persona_details = "Your friendly and encouraging AI guide, Rox."
        system_prompt_text = greeting_prompt_config.get('system_prompt', '').format(
            student_name=student_name,
            persona_details=persona_details
        )
        user_prompt_text = greeting_prompt_config.get('user_prompt', '')
        full_prompt = f"{system_prompt_text}\n\n{user_prompt_text}"
        
        logger.info(f"Full prompt for greeting LLM: {full_prompt}")
        
        raw_llm_response_text = ""
        try:
            response = await model.generate_content_async(full_prompt)
            raw_llm_response_text = response.text
            logger.info(f"Raw LLM Response for greeting: {raw_llm_response_text}")
        except Exception as gen_err:
            logger.error(f"Error during model.generate_content_async(): {gen_err}", exc_info=True)
            fallback_tts = f"Hello {student_name}! Welcome. (LLM Generation Error)"
            logger.info(f"ConversationalManagerNode: Using fallback greeting due to generation error: {fallback_tts}")
            error_output = {"text_for_tts": fallback_tts, "ui_actions": []}
            return {"final_flow_output": error_output}

        try:
            llm_output_json = json.loads(raw_llm_response_text)
            greeting_tts = llm_output_json.get("greeting_tts", f"Hello {student_name}! I'm Rox, welcome! (JSON Key Missing)")
        except json.JSONDecodeError as json_err:
            logger.error(f"JSONDecodeError parsing LLM greeting response. Error: {json_err}. Raw text: {raw_llm_response_text}")
            greeting_tts = f"Hello {student_name}! I'm Rox. (JSON Parse Error)"
        except Exception as parse_err:
            logger.error(f"Unexpected error parsing LLM greeting response. Error: {parse_err}. Raw text: {raw_llm_response_text}")
            greeting_tts = f"Hello {student_name}! I'm Rox. (Unexpected Parse Error)"

        logger.info(f"ConversationalManagerNode: LLM-generated greeting: {greeting_tts}")
        final_output = {"text_for_tts": greeting_tts, "ui_actions": []}
        return {"final_flow_output": final_output}

    except Exception as e:
        logger.error(f"Outer error in handle_home_greeting_node (e.g., config, model init): {e}", exc_info=True)
        fallback_tts = f"Hello {student_name}! Welcome. (LLM Setup Error)"
        logger.info(f"ConversationalManagerNode: Using fallback greeting due to setup error: {fallback_tts}")
        error_output = {"text_for_tts": fallback_tts, "ui_actions": []}
        return {"final_flow_output": error_output}
