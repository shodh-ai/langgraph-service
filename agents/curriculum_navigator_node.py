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

# Initialize Vertex AI if not already initialized in another module
try:
    if 'vertexai' in globals() and not hasattr(vertexai, '_initialized'):
        vertexai.init(project="windy-orb-460108-t0", location="us-central1")
        vertexai._initialized = True
        logger.info("Vertex AI initialized in curriculum_navigator_node")
except Exception as e:
    logger.error(f"Failed to initialize Vertex AI in curriculum_navigator_node: {e}")

# Load Gemini model if not already loaded
try:
    gemini_model = GenerativeModel("gemini-1.5-pro")
    logger.info("Gemini model loaded in curriculum_navigator_node")
except Exception as e:
    logger.error(f"Failed to load Gemini model in curriculum_navigator_node: {e}")
    gemini_model = None

async def determine_next_pedagogical_step_node(state: AgentGraphState) -> dict:
    """The primary decision-maker for what the student should do next using Vertex AI Gemini."""
    context = state.get("current_context")
    student_data = state.get("student_memory_context", {})
    diagnosis = state.get("diagnosis_result", {})
    feedback = state.get("feedback_content", {})
    next_practice = state.get("next_practice", {})
    
    logger.info(f"CurriculumNavigatorNode: Determining next pedagogical step")
    
    if not gemini_model:
        logger.warning("CurriculumNavigatorNode: Gemini model not available, using rule-based implementation")
        # Fallback to rule-based implementation
        # Default next step
        next_step = {
            "action": "continue_current_task",
            "task_id": getattr(context, "current_prompt_id", None),
            "rationale": "Continue with the current task to build mastery."
        }
        
        # Decision logic based on context and diagnosis
        task_stage = getattr(context, "task_stage", None)
        
        if task_stage == "reviewing_feedback":
            # After feedback, determine if student needs practice, new task, or lesson
            if diagnosis.get("overall_score", 3) < 3:
                # Low score - provide targeted practice
                next_step = {
                    "action": "provide_targeted_practice",
                    "task_id": f"practice_{diagnosis.get('primary_error', 'general').replace(' ', '_')}",
                    "rationale": f"Additional practice needed in {diagnosis.get('primary_error', 'fundamental skills')}."
                }
            else:
                # Good score - move to next task in sequence
                next_step = {
                    "action": "advance_to_next_task",
                    "task_id": f"next_in_sequence_after_{getattr(context, 'current_prompt_id', 'current')}",
                    "rationale": "Ready to progress to the next challenge."
                }
        elif task_stage == "skill_drill_complete":
            # After completing a drill, return to main task or provide another drill
            if diagnosis.get("drill_mastery", False):
                next_step = {
                    "action": "return_to_main_task",
                    "task_id": getattr(context, "current_prompt_id", None),
                    "rationale": "Sufficient mastery achieved in drill, ready to apply skills to main task."
                }
            else:
                next_step = {
                    "action": "continue_skill_drill",
                    "task_id": f"next_level_{getattr(context, 'current_prompt_id', 'current')}",
                    "rationale": "Additional practice needed to master this skill."
                }
    else:
        try:
            # Get prompt templates from config
            system_prompt = PROMPTS.get("curriculum", {}).get("navigator", {}).get("system", "")
            user_prompt = PROMPTS.get("curriculum", {}).get("navigator", {}).get("user_template", "")
            
            # If prompts are not found in config, use default prompts
            if not system_prompt:
                system_prompt = """You are an expert TOEFL curriculum navigator AI responsible for determining the optimal next step in a student's learning journey.
                
                Based on the student's context, profile data, diagnosis results, and feedback, determine what they should do next in their TOEFL preparation.
                
                Student context: {{current_context}}
                Student profile data: {{student_memory_context}}
                Diagnosis results: {{diagnosis_result}}
                Feedback content: {{feedback_content}}
                Next practice suggestion: {{next_practice}}
                
                As a curriculum navigator, your goal is to create an optimal learning path that addresses weaknesses while building on strengths, maintaining student motivation, and ensuring comprehensive TOEFL preparation."""
            
            if not user_prompt:
                user_prompt = """Determine the next pedagogical step for this student based on their current context, profile, and performance.
                
                Return your decision in the following JSON format:
                {
                  "action": "[continue_current_task|provide_targeted_practice|advance_to_next_task|review_previous_material|take_assessment|return_to_main_task|continue_skill_drill]",
                  "task_id": "[identifier for the next task/activity]",
                  "rationale": "[explanation of why this is the optimal next step]",
                  "estimated_duration_minutes": [approximate time to complete],
                  "learning_objectives": ["list", "of", "specific", "learning", "objectives"]
                }
                
                Make sure your decision is pedagogically sound and tailored to the student's specific needs and progress."""
            
            # Replace placeholders in prompts
            system_prompt = system_prompt.replace("{{current_context}}", json.dumps(context.dict() if hasattr(context, "dict") else context))
            system_prompt = system_prompt.replace("{{student_memory_context}}", json.dumps(student_data))
            system_prompt = system_prompt.replace("{{diagnosis_result}}", json.dumps(diagnosis))
            system_prompt = system_prompt.replace("{{feedback_content}}", json.dumps(feedback))
            system_prompt = system_prompt.replace("{{next_practice}}", json.dumps(next_practice))
            
            # Create Gemini content
            contents = [
                Content(role="user", parts=[system_prompt]),
                Content(role="model", parts=["I understand. I'll determine the optimal next pedagogical step based on the student's context, profile, and performance."]),
                Content(role="user", parts=[user_prompt])
            ]
            
            # Generate response from Gemini with lower temperature for more consistent output
            response = gemini_model.generate_content(contents, generation_config={
                "temperature": 0.2,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json"
            })
            
            # Process the response to extract JSON
            response_text = response.text
            
            # Try to parse JSON from the response
            try:
                # Extract JSON if it's embedded in a code block
                if "```json" in response_text and "```" in response_text:
                    json_text = response_text.split("```json")[1].split("```")[0].strip()
                    next_step = json.loads(json_text)
                elif "```" in response_text:
                    json_text = response_text.split("```")[1].split("```")[0].strip()
                    next_step = json.loads(json_text)
                else:
                    # Try to parse the whole response as JSON
                    next_step = json.loads(response_text)
                
                logger.info(f"CurriculumNavigatorNode: Successfully parsed JSON next step")
                
                # Ensure the next_step has the expected structure
                required_fields = ["action", "task_id", "rationale"]
                for field in required_fields:
                    if field not in next_step:
                        if field == "action":
                            next_step[field] = "continue_current_task"
                        elif field == "task_id":
                            next_step[field] = getattr(context, "current_prompt_id", "current_task")
                        elif field == "rationale":
                            next_step[field] = "Continuing with the current learning path."
                
                # Add optional fields if not present
                if "estimated_duration_minutes" not in next_step:
                    next_step["estimated_duration_minutes"] = 15
                if "learning_objectives" not in next_step:
                    primary_error = diagnosis.get("primary_error", "general skills")
                    next_step["learning_objectives"] = [f"Improve {primary_error}"]
                    
            except json.JSONDecodeError as e:
                logger.error(f"CurriculumNavigatorNode: Failed to parse JSON response: {e}")
                logger.error(f"Response text: {response_text}")
                
                # Extract key information from text response
                action = "continue_current_task"
                task_id = getattr(context, "current_prompt_id", "current_task")
                rationale = "Continuing with the current learning path based on text analysis."
                
                if "practice" in response_text.lower() or "drill" in response_text.lower():
                    action = "provide_targeted_practice"
                    task_id = f"practice_{diagnosis.get('primary_error', 'general').replace(' ', '_')}"
                elif "next" in response_text.lower() or "advance" in response_text.lower():
                    action = "advance_to_next_task"
                    task_id = f"next_in_sequence_after_{getattr(context, 'current_prompt_id', 'current')}"
                elif "review" in response_text.lower():
                    action = "review_previous_material"
                    task_id = f"review_{getattr(context, 'current_prompt_id', 'current')}"
                
                # Extract rationale if possible
                rationale_indicators = ["because", "as", "since", "reason"]
                for indicator in rationale_indicators:
                    if indicator in response_text.lower():
                        parts = response_text.lower().split(indicator, 1)
                        if len(parts) > 1:
                            potential_rationale = parts[1].strip()
                            if len(potential_rationale) > 10:
                                rationale = potential_rationale
                                break
                
                # Create a next_step object from extracted information
                next_step = {
                    "action": action,
                    "task_id": task_id,
                    "rationale": rationale,
                    "estimated_duration_minutes": 15,
                    "learning_objectives": [f"Improve {diagnosis.get('primary_error', 'general skills')}"]
                }
                
        except Exception as e:
            logger.error(f"CurriculumNavigatorNode: Error calling Gemini API: {e}")
            # Fallback to rule-based implementation
            next_step = {
                "action": "continue_current_task",
                "task_id": getattr(context, "current_prompt_id", None),
                "rationale": "Continuing with current task due to processing error.",
                "estimated_duration_minutes": 15,
                "learning_objectives": ["Continue building skills in current area"]
            }
            
            # Apply basic logic based on diagnosis
            if diagnosis.get("overall_score", 3) < 3:
                next_step = {
                    "action": "provide_targeted_practice",
                    "task_id": f"practice_{diagnosis.get('primary_error', 'general').replace(' ', '_')}",
                    "rationale": f"Additional practice needed in {diagnosis.get('primary_error', 'fundamental skills')}.",
                    "estimated_duration_minutes": 10,
                    "learning_objectives": [f"Address weakness in {diagnosis.get('primary_error', 'fundamental skills')}"]
                }
    
    logger.info(f"CurriculumNavigatorNode: Next step - {next_step['action']}: {next_step['task_id']}")
    
    return {"next_task_details": next_step}