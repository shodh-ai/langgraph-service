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
        logger.info("Vertex AI initialized in feedback_generator_node")
except Exception as e:
    logger.error(f"Failed to initialize Vertex AI in feedback_generator_node: {e}")

try:
    gemini_model = GenerativeModel("gemini-1.5-pro")
    logger.info("Gemini model loaded in feedback_generator_node")
except Exception as e:
    logger.error(f"Failed to load Gemini model in feedback_generator_node: {e}")
    gemini_model = None

async def generate_feedback_for_task_node(state: AgentGraphState) -> dict:
    diagnosis = state.get("diagnosis_result", {})
    persona = state.get("current_teacher_persona", "nurturer")
    student_data = state.get("student_memory_context", {})
    context = state.get("current_context")
    socratic_questions = state.get("socratic_questions", [])
    
    logger.info(f"FeedbackGeneratorNode: Generating feedback with persona '{persona}'")
    
    if not gemini_model:
        logger.warning("FeedbackGeneratorNode: Gemini model not available, using stub implementation")
        # fallback to stub implementation
        feedback = {
            "text": f"I noticed your {diagnosis.get('primary_strength', 'response')} was particularly strong. " +
                    f"One area to focus on might be {diagnosis.get('primary_error', 'continuing to practice')}.",
            "frontend_rpc_calls": [
                {
                    "function_name": "highlightText",
                    "args": ["#transcript_display", "Your vocabulary choice here was excellent!", 10, 25, "green"]
                },
                {
                    "function_name": "showTip",
                    "args": ["Consider using more complex sentence structures to express your ideas."]
                }
            ]
        }
    else:
        try:
            system_prompt = PROMPTS.get("feedback", {}).get("general", {}).get("system", "")
            user_prompt = PROMPTS.get("feedback", {}).get("general", {}).get("user_template", "")
            
            system_prompt = system_prompt.replace("{{current_teacher_persona}}", persona)
            system_prompt = system_prompt.replace("{{student_memory_context.profile}}", json.dumps(student_data.get("profile", {})))
            system_prompt = system_prompt.replace("{{diagnosis_result}}", json.dumps(diagnosis))
            system_prompt = system_prompt.replace("{{current_context}}", json.dumps(context.dict() if hasattr(context, "dict") else context))
            
            socratic_context = ""
            if socratic_questions:
                socratic_context = "Here are Socratic questions that have been generated for this student:\n"
                for i, question in enumerate(socratic_questions):
                    socratic_context += f"{i+1}. {question}\n"
                socratic_context += "\nConsider incorporating these questions into your feedback."
            
            contents = [
                Content(role="user", parts=[system_prompt]),
                Content(role="model", parts=["I understand. I'll generate personalized feedback as a TOEFL tutor with the specified persona."]),
                Content(role="user", parts=[user_prompt + "\n" + socratic_context])
            ]
            
            response = gemini_model.generate_content(contents, generation_config={
                "temperature": 0.7,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json"
            })
            
            try:
                feedback = json.loads(response.text)
                logger.info("FeedbackGeneratorNode: Successfully parsed Gemini response")
                
                if not isinstance(feedback, dict) or "text" not in feedback:
                    logger.warning(f"FeedbackGeneratorNode: Unexpected response structure: {feedback}")
                    feedback = {
                        "text": response.text,
                        "frontend_rpc_calls": []
                    }
                
                # ensure frontend_rpc_calls exists
                if "frontend_rpc_calls" not in feedback:
                    feedback["frontend_rpc_calls"] = []
                
            except json.JSONDecodeError:
                logger.error(f"FeedbackGeneratorNode: Failed to parse Gemini response as JSON: {response.text}")
                feedback = {
                    "text": response.text,
                    "frontend_rpc_calls": []
                }
                
        except Exception as e:
            logger.error(f"FeedbackGeneratorNode: Error calling Gemini API: {e}")
            # Fallback to stub implementation
            feedback = {
                "text": f"I noticed your {diagnosis.get('primary_strength', 'response')} was particularly strong. " +
                        f"One area to focus on might be {diagnosis.get('primary_error', 'continuing to practice')}.",
                "frontend_rpc_calls": []
            }
    
    # adjust feedback based on teacher persona if not already done by the model
    if persona == "structuralist" and "structure" not in feedback["text"].lower():
        feedback["text"] += " Let's look at how we can organize your ideas more effectively."
    elif persona == "challenger" and "think" not in feedback["text"].lower():
        feedback["text"] += " I'd like you to think more deeply about how you could strengthen your argument."
    elif persona == "socratic" and "?" not in feedback["text"]:
        feedback["text"] += " What do you think would make your response even stronger?"
    
    logger.info(f"FeedbackGeneratorNode: Completed feedback generation with {len(feedback.get('frontend_rpc_calls', []))} frontend actions")
    
    return {"feedback_content": feedback}