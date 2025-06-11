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
            logger.error(f"Could not find 'PROMPTS' key in {PROMPTS_FILE_PATH} for curriculum_navigator_node")
except FileNotFoundError:
    logger.error(f"Prompts file not found at {PROMPTS_FILE_PATH} for curriculum_navigator_node")
except yaml.YAMLError as e:
    logger.error(f"Error parsing YAML from {PROMPTS_FILE_PATH} for curriculum_navigator_node: {e}")

async def determine_next_pedagogical_step_stub_node(state: AgentGraphState) -> dict:
    """
    Determines the next task and generates an LLM-based suggestion for it.
    Sets 'next_task_details' and a new 'task_suggestion_llm_output' field in the state.
    The 'output_content' from this node will primarily contain UI actions for the task button.
    
    Args:
        state: The current agent graph state
        
    Returns:
        Dict with updates for 'next_task_details', 'task_suggestion_llm_output', and 'output_content'.
    """
    logger.info("CurriculumNavigatorNode: Determining next pedagogical step and generating LLM task suggestion.")
    
    # Hardcoded first speaking task details (matches your P1 goal example)
    next_task = {
        "type": "SPEAKING",
        "question_type": "Q1", 
        "prompt_id": "SPK_Q1_P1_FAV_HOLIDAY", # Example prompt_id
        "title": "Your Favorite Holiday", # Changed to match your example
        "description": "Tell me about your favorite holiday. What do you usually do, and why is it special to you?",
        "prep_time_seconds": 15,
        "response_time_seconds": 45
    }
    
    logger.info(f"CurriculumNavigatorNode: Selected task: {next_task['title']} ({next_task['prompt_id']})")

    task_suggestion_tts = f"Would you like to start with a task about {next_task['title']}?" # Default fallback
    task_suggestion_llm_output = {"task_suggestion_tts": task_suggestion_tts} # Default structure

    prompt_config = PROMPTS.get('welcome_task_suggestion')
    if not prompt_config:
        logger.error("Welcome task suggestion prompt configuration not found. Using default suggestion.")
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found in environment. Using default task suggestion.")
        else:
            try:
                logger.debug("TaskSuggestion: Attempting genai.configure()...")
                genai.configure(api_key=api_key)
                logger.debug("TaskSuggestion: genai.configure() successful.")
                
                logger.debug(f"TaskSuggestion: Attempting to initialize GenerativeModel: gemini-2.0-flash")
                model = genai.GenerativeModel(
                    'gemini-2.0-flash',
                    generation_config=GenerationConfig(response_mime_type="application/json")
                )
                logger.debug("TaskSuggestion: GenerativeModel initialized successfully.")

                persona_details = "Your friendly and encouraging AI guide, Rox."
                system_prompt_text = prompt_config.get('system_prompt', '').format(
                    persona_details=persona_details,
                    task_title=next_task['title']
                )
                user_prompt_text = prompt_config.get('user_prompt', '')
                full_prompt = f"{system_prompt_text}\n\n{user_prompt_text}"
                
                logger.info(f"TaskSuggestion: GOOGLE_API_KEY loaded: {api_key[:5]}...{api_key[-5:] if len(api_key) > 10 else ''}")
                logger.info(f"TaskSuggestion: Full prompt for LLM: {full_prompt}")
                
                raw_llm_response_text_task = ""
                try:
                    logger.debug("TaskSuggestion: Attempting model.generate_content_async()...")
                    response = await model.generate_content_async(full_prompt)
                    logger.debug("TaskSuggestion: model.generate_content_async() successful.")
                    raw_llm_response_text_task = response.text
                    logger.info(f"TaskSuggestion: Raw LLM Response: {raw_llm_response_text_task}")
                except Exception as gen_err:
                    logger.error(f"TaskSuggestion: Error during model.generate_content_async(): {gen_err}", exc_info=True)
                    task_suggestion_tts += " (LLM Generation Error)" # Append to default
                    task_suggestion_llm_output = {"task_suggestion_tts": task_suggestion_tts}
                    logger.info(f"CurriculumNavigatorNode: Using fallback task suggestion due to generation error.")
                    # Skip further JSON processing if generation failed
                    return {
                        "output_content": {
                            "response": "", 
                            "ui_actions": [
                                {
                                    "action_type": "DISPLAY_NEXT_TASK_BUTTON", 
                                    "parameters": next_task
                                },
                                {
                                     "action_type": "ENABLE_START_TASK_BUTTON", 
                                     "parameters": {"button_id": "start_task_button_id"} 
                                }
                            ]
                        },
                        "task_suggestion_llm_output": task_suggestion_llm_output
                    }

                try:
                    llm_json_output = json.loads(raw_llm_response_text_task)
                    task_suggestion_tts = llm_json_output.get("task_suggestion_tts", task_suggestion_tts + " (JSON Key Missing)")
                except json.JSONDecodeError as json_err:
                    logger.error(f"TaskSuggestion: JSONDecodeError parsing LLM response. Error: {json_err}. Raw text: {raw_llm_response_text_task}")
                    task_suggestion_tts += " (JSON Parse Error)"
                except Exception as parse_err:
                    logger.error(f"TaskSuggestion: Unexpected error parsing LLM response. Error: {parse_err}. Raw text: {raw_llm_response_text_task}")
                    task_suggestion_tts += " (Unexpected Parse Error)"
                
                task_suggestion_llm_output = {"task_suggestion_tts": task_suggestion_tts}
                logger.info(f"CurriculumNavigatorNode: LLM-generated task suggestion: {task_suggestion_tts}")

            except Exception as e:
                logger.error(f"TaskSuggestion: Outer error in LLM call block (e.g., config, model init): {e}", exc_info=True)
                task_suggestion_tts += " (LLM Setup Error)" # Append to default
                task_suggestion_llm_output = {"task_suggestion_tts": task_suggestion_tts}
                logger.info(f"CurriculumNavigatorNode: Using default task suggestion due to LLM setup error.")

    # Output content for this node focuses on the UI action for the task button.
    # The actual TTS for suggesting the task is in task_suggestion_llm_output and will be handled by the formatter.
    output_for_this_node = {
        "response": "", # No direct TTS from this node's output_content.response
        "ui_actions": [
            {
                "action_type": "DISPLAY_NEXT_TASK_BUTTON", # Or a more generic "SHOW_TASK_PROMPT"
                "parameters": next_task
            },
            {
                 "action_type": "ENABLE_START_TASK_BUTTON", # As per your goal
                 "parameters": {"button_id": "start_task_button_id"} # Assuming a button ID
            }
        ]
    }
    
    # Return state updates
    return {
        "next_task_details": next_task,
        "task_suggestion_llm_output": task_suggestion_llm_output, # New field for the formatter
        "output_content": output_for_this_node # Contains UI actions
    }
