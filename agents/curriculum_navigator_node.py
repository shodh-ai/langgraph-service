#TBD
#not connected to backend DB

import logging
import yaml
import os
import json
import enum
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from state import AgentGraphState
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)

PROMPTS_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'llm_prompts.yaml')

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


class CurriculumNavigatorNode:
    """
    The primary pedagogical decision-maker for "what's next?" in the student's overall learning journey.
    
    Purpose:
    - Determine the next learning task or activity based on student model and progress
    - Generate appropriate task suggestions and UI actions
    - Guide the pedagogical flow between different pages in the application
    - Support adaptive learning paths based on student performance and needs
    """
    
    class TaskType(enum.Enum):
        """Enumeration of task types"""
        SPEAKING = "SPEAKING"
        WRITING = "WRITING"
        READING = "READING"
        LISTENING = "LISTENING"
        TEACHING = "TEACHING"
        DRILL = "DRILL"
        FEEDBACK = "FEEDBACK"
        CONVERSATION = "CONVERSATION"
    
    class TaskDifficulty(enum.Enum):
        """Enumeration of task difficulty levels"""
        BEGINNER = "beginner"
        INTERMEDIATE = "intermediate"
        ADVANCED = "advanced"
        EXAM_LEVEL = "exam_level"
    
    def __init__(self):
        """Initialize the CurriculumNavigatorNode"""
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        else:
            logger.error("GOOGLE_API_KEY not found in environment.")
            
        try:
            self.model = genai.GenerativeModel(
                'gemini-2.5-flash-preview-05-20',
                generation_config=GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            logger.debug("CurriculumNavigatorNode: GenerativeModel initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing GenerativeModel: {e}")
            self.model = None
            
        self._instance = self
        
        self.task_database = {
            "SPEAKING": [
                {
                    "question_type": "Q1",
                    "prompt_id": "SPK_Q1_P1_FAV_HOLIDAY",
                    "title": "Your Favorite Holiday",
                    "description": "Tell me about your favorite holiday. What do you usually do, and why is it special to you?",
                    "difficulty": "beginner",
                    "topic": "personal_experience",
                    "prep_time_seconds": 15,
                    "response_time_seconds": 45
                },
                {
                    "question_type": "Q1",
                    "prompt_id": "SPK_Q1_P1_HOMETOWN",
                    "title": "Your Hometown",
                    "description": "Describe your hometown. What is it known for, and what do you like about it?",
                    "difficulty": "beginner",
                    "topic": "personal_experience",
                    "prep_time_seconds": 15,
                    "response_time_seconds": 45
                },
                {
                    "question_type": "Q2",
                    "prompt_id": "SPK_Q2_P1_SOCIAL_MEDIA",
                    "title": "Social Media Impact",
                    "description": "Some people say social media has improved communication, while others say it has made it worse. Which view do you agree with and why?",
                    "difficulty": "intermediate",
                    "topic": "technology",
                    "prep_time_seconds": 20,
                    "response_time_seconds": 60
                },
                {
                    "question_type": "Q3",
                    "prompt_id": "SPK_Q3_P1_CAMPUS_NOTICE",
                    "title": "Campus Library Hours",
                    "description": "Read the campus notice about changes to library hours and explain how this might affect students' study habits.",
                    "difficulty": "intermediate",
                    "topic": "campus_life",
                    "prep_time_seconds": 30,
                    "response_time_seconds": 60,
                    "additional_materials": {
                        "reading": "The university library will now be open 24 hours during exam periods. Outside of exam periods, the library will close at 10 PM instead of midnight to reduce operational costs."
                    }
                },
                {
                    "question_type": "Q4",
                    "prompt_id": "SPK_Q4_P1_LECTURE",
                    "title": "Psychology Lecture",
                    "description": "Listen to the lecture on cognitive biases and explain how the availability heuristic affects decision making.",
                    "difficulty": "advanced",
                    "topic": "psychology",
                    "prep_time_seconds": 30,
                    "response_time_seconds": 60,
                    "additional_materials": {
                        "audio": "psychology_lecture.mp3"
                    }
                }
            ],
            "WRITING": [
                {
                    "question_type": "integrated",
                    "prompt_id": "WRT_INT_P1_URBANIZATION",
                    "title": "Effects of Urbanization",
                    "description": "Read the passage about urbanization and listen to the lecture. Then, write an essay summarizing the main points from the lecture and explaining how they relate to the reading.",
                    "difficulty": "advanced",
                    "topic": "urban_development",
                    "time_minutes": 20,
                    "additional_materials": {
                        "reading": "urbanization_passage.txt",
                        "audio": "urbanization_lecture.mp3"
                    }
                },
                {
                    "question_type": "independent",
                    "prompt_id": "WRT_IND_P1_TECHNOLOGY",
                    "title": "Technology in Education",
                    "description": "Do you agree or disagree with the following statement? Technology has improved the quality of education. Use specific reasons and examples to support your answer.",
                    "difficulty": "intermediate",
                    "topic": "education",
                    "time_minutes": 30
                }
            ],
            "TEACHING": [
                {
                    "prompt_id": "TCH_SPEAKING_ORGANIZATION",
                    "title": "Organizing Your Speaking Response",
                    "description": "Learn effective strategies for organizing your speaking responses in TOEFL tasks.",
                    "difficulty": "beginner",
                    "topic": "speaking_skills",
                    "skill_focus": "organization"
                },
                {
                    "prompt_id": "TCH_WRITING_THESIS",
                    "title": "Writing Effective Thesis Statements",
                    "description": "Learn how to write clear and strong thesis statements for your TOEFL essays.",
                    "difficulty": "intermediate",
                    "topic": "writing_skills",
                    "skill_focus": "thesis_development"
                }
            ],
            "DRILL": [
                {
                    "prompt_id": "DRL_TRANSITIONS",
                    "title": "Transition Phrases Practice",
                    "description": "Practice using transition phrases to connect ideas smoothly.",
                    "difficulty": "intermediate",
                    "topic": "language_skills",
                    "skill_focus": "transitions"
                },
                {
                    "prompt_id": "DRL_PRONUNCIATION",
                    "title": "Pronunciation Drill",
                    "description": "Practice pronouncing commonly confused sounds in English.",
                    "difficulty": "beginner",
                    "topic": "speaking_skills",
                    "skill_focus": "pronunciation"
                }
            ]
        }

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
    
    next_task = {
        "type": "SPEAKING",
        "question_type": "Q1", 
        "prompt_id": "SPK_Q1_P1_FAV_HOLIDAY",
        "title": "Your Favorite Holiday",
        "description": "Tell me about your favorite holiday. What do you usually do, and why is it special to you?",
        "prep_time_seconds": 15,
        "response_time_seconds": 45
    }
    
    logger.info(f"CurriculumNavigatorNode: Selected task: {next_task['title']} ({next_task['prompt_id']})")

    task_suggestion_tts = f"Would you like to start with a task about {next_task['title']}?"
    task_suggestion_llm_output = {"task_suggestion_tts": task_suggestion_tts}

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
                
                logger.debug(f"TaskSuggestion: Attempting to initialize GenerativeModel: gemini-2.5-flash-preview-05-20")
                model = genai.GenerativeModel(
                    'gemini-2.5-flash-preview-05-20',
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
                    task_suggestion_tts += " (LLM Generation Error)"
                    task_suggestion_llm_output = {"task_suggestion_tts": task_suggestion_tts}
                    logger.info(f"CurriculumNavigatorNode: Using fallback task suggestion due to generation error.")
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
                task_suggestion_tts += " (LLM Setup Error)"
                task_suggestion_llm_output = {"task_suggestion_tts": task_suggestion_tts}
                logger.info(f"CurriculumNavigatorNode: Using default task suggestion due to LLM setup error.")

    output_for_this_node = {
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
    }
    
    return {
        "next_task_details": next_task,
        "task_suggestion_llm_output": task_suggestion_llm_output,
        "output_content": output_for_this_node
    }
