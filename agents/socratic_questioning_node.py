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
logger.info(f"Using model: {type(gemini_model).__name__} in socratic_questioning_node")

async def generate_socratic_question_node(state: AgentGraphState) -> dict:
    diagnosis = state.get("diagnosis_result", {})
    transcript = state.get("transcript", "")
    task_prompt = state.get("task_prompt", {})
    
    logger.info(f"SocraticQuestioningNode: Generating Socratic questions")
    
    if not gemini_model:
        logger.warning("SocraticQuestioningNode: Gemini model not available, using stub implementation")
        # Fallback to stub implementation
        questions = [
            f"How might you expand on your point about {diagnosis.get('primary_strength', 'this topic')}?",
            f"What evidence could strengthen your argument about {task_prompt.get('main_topic', 'this issue')}?",
            f"Have you considered alternative perspectives on {task_prompt.get('main_topic', 'this matter')}?"
        ]
    else:
        try:
            system_prompt = PROMPTS.get("socratic", {}).get("questioning", {}).get("system", "")
            user_prompt = PROMPTS.get("socratic", {}).get("questioning", {}).get("user_template", "")
            
            system_prompt = system_prompt.replace("{{transcript}}", transcript)
            system_prompt = system_prompt.replace("{{diagnosis_result}}", json.dumps(diagnosis))
            system_prompt = system_prompt.replace("{{task_prompt}}", json.dumps(task_prompt))
            
            user_prompt = user_prompt.replace("{{diagnosis_result.primary_error}}", diagnosis.get("primary_error", "areas for improvement"))
            user_prompt = user_prompt.replace("{{diagnosis_result.primary_strength}}", diagnosis.get("primary_strength", "strengths"))
            user_prompt = user_prompt.replace("{{task_prompt.main_topic}}", task_prompt.get("main_topic", "the topic"))
            
            contents = [
                Content(role="user", parts=[Part.from_text(system_prompt)]),
                Content(role="model", parts=[Part.from_text("I understand. I'll generate Socratic questions to help the student reflect on their response.")]),
                Content(role="user", parts=[Part.from_text(user_prompt)])
            ]
            
            response = gemini_model.generate_content(contents, generation_config={
                "temperature": 0.8,
                "max_output_tokens": 1024
            })
            
            response_text = response.text
            
            if response_text.strip().startswith("[") and response_text.strip().endswith("]"):
                try:
                    questions = json.loads(response_text)
                    logger.info(f"SocraticQuestioningNode: Successfully parsed JSON list of {len(questions)} questions")
                except json.JSONDecodeError:
                    questions = [line.strip() for line in response_text.split("\n") if line.strip() and "?" in line]
                    logger.info(f"SocraticQuestioningNode: Extracted {len(questions)} questions from text response")
            else:
                # extract questions line by line
                questions = [line.strip() for line in response_text.split("\n") if line.strip() and "?" in line]
                
                # if no questions with question marks were found, try to split by numbers
                if not questions:
                    import re
                    questions = re.findall(r'\d+\.\s*(.*?(?:\?|$))', response_text)
                
                logger.info(f"SocraticQuestioningNode: Extracted {len(questions)} questions from text response")
            
            # ensure we have at least some questions
            if not questions:
                logger.warning(f"SocraticQuestioningNode: No questions extracted from response: {response_text}")
                questions = [
                    f"How might you expand on your point about {diagnosis.get('primary_strength', 'this topic')}?",
                    f"What evidence could strengthen your argument about {task_prompt.get('main_topic', 'this issue')}?",
                    f"Have you considered alternative perspectives on {task_prompt.get('main_topic', 'this matter')}?"
                ]
                
        except Exception as e:
            logger.error(f"SocraticQuestioningNode: Error calling Gemini API: {e}")
            # Fallback to stub implementation
            questions = [
                "What do you think is the strongest part of your response?",
                "How might you improve your response if you had more time?",
                "What additional examples could you include to support your main point?"
            ]
    
    logger.info(f"SocraticQuestioningNode: Generated {len(questions)} questions")
    
    return {"socratic_questions": questions}