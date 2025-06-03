import logging
from state import AgentGraphState
import yaml
import os
import json
import vertexai
from vertexai.generative_models import GenerativeModel, Content, Part

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
        logger.info("Vertex AI initialized in session_notes_node")
except Exception as e:
    logger.error(f"Failed to initialize Vertex AI in session_notes_node: {e}")

try:
    gemini_model = GenerativeModel("gemini-1.5-pro")
    logger.info("Gemini model loaded in session_notes_node")
except Exception as e:
    logger.error(f"Failed to load Gemini model in session_notes_node: {e}")
    gemini_model = None

async def compile_session_notes_node(state: AgentGraphState) -> dict:
    """
    Compiles comprehensive session notes based on the interaction history.
    These notes summarize key points, student progress, and recommendations.
    
    Args:
        state (AgentGraphState): The current state of the agent graph
        
    Returns:
        dict: Contains the session_notes with summary, progress, and recommendations
    """
    context = state.get("current_context", {})
    student_data = state.get("student_memory_context", {})
    diagnosis = state.get("diagnosis_result", {})
    feedback = state.get("feedback_content", {})
    next_step = state.get("next_task_details", {})
    
    logger.info(f"SessionNotesNode: Compiling session notes for user: {state.get('user_id', 'unknown')}")
    
    if not gemini_model:
        logger.warning("SessionNotesNode: Gemini model not available, using stub implementation")
        # Fallback to stub implementation
        session_notes = {
            "summary": "Student completed a TOEFL practice task.",
            "progress": {
                "strengths": ["Good effort", "Completed task"],
                "areas_for_improvement": ["Continue practicing"]
            },
            "recommendations": ["Practice more", "Review feedback"],
            "next_steps": "Continue with the next lesson"
        }
        return {"session_notes": session_notes}
    
    try:
        # Get the appropriate prompt template
        system_prompt = PROMPTS.get("session_notes", {}).get("system_prompt", 
            """You are an expert TOEFL tutor compiling session notes.
            Create comprehensive notes based on the student's interaction, diagnosis, and feedback.
            These notes should summarize the session, highlight progress, and provide recommendations.
            Format your response as a JSON with:
            {
                "summary": "Brief summary of the session",
                "progress": {
                    "strengths": ["List of student strengths observed"],
                    "areas_for_improvement": ["List of areas needing improvement"]
                },
                "recommendations": ["List of specific recommendations"],
                "next_steps": "Description of suggested next steps"
            }
            """)
        
        # Replace placeholders with actual data
        system_prompt = system_prompt.replace("{{current_context}}", json.dumps(context))
        system_prompt = system_prompt.replace("{{student_memory_context}}", json.dumps(student_data))
        system_prompt = system_prompt.replace("{{diagnosis_result}}", json.dumps(diagnosis))
        system_prompt = system_prompt.replace("{{feedback_content}}", json.dumps(feedback))
        system_prompt = system_prompt.replace("{{next_task_details}}", json.dumps(next_step))
        
        user_prompt = PROMPTS.get("session_notes", {}).get("user_prompt", 
            "Please compile comprehensive session notes for this student based on their interaction.")
        
        # Create content for the model
        contents = [
            Content(role="user", parts=[Part.from_text(system_prompt)]),
            Content(role="model", parts=[Part.from_text("I'll compile detailed session notes.")]),
            Content(role="user", parts=[Part.from_text(user_prompt)])
        ]
        
        # Call the Gemini model
        response = gemini_model.generate_content(contents, generation_config={
            "temperature": 0.3,  # Lower temperature for more factual output
            "max_output_tokens": 1024
        })
        
        response_text = response.text
        
        # Parse the JSON response
        try:
            # Extract JSON if it's embedded in a code block
            if "```json" in response_text and "```" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
                session_notes = json.loads(json_text)
            elif response_text.strip().startswith("{") and response_text.strip().endswith("}"):
                session_notes = json.loads(response_text)
            else:
                # If not in JSON format, create a simple structure
                session_notes = {
                    "summary": response_text[:100] + "...",  # First 100 chars as summary
                    "progress": {
                        "strengths": ["Completed the task"],
                        "areas_for_improvement": ["Continue practicing"]
                    },
                    "recommendations": ["Review feedback", "Practice regularly"],
                    "next_steps": "Continue with the learning plan"
                }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            session_notes = {
                "summary": "Student completed a TOEFL practice task.",
                "progress": {
                    "strengths": ["Good effort", "Completed task"],
                    "areas_for_improvement": ["Continue practicing"]
                },
                "recommendations": ["Practice more", "Review feedback"],
                "next_steps": "Continue with the next lesson"
            }
        
        return {"session_notes": session_notes}
        
    except Exception as e:
        logger.error(f"Error in compile_session_notes_node: {e}")
        # Fallback response
        session_notes = {
            "summary": "Student completed a TOEFL practice task.",
            "progress": {
                "strengths": ["Good effort", "Completed task"],
                "areas_for_improvement": ["Continue practicing"]
            },
            "recommendations": ["Practice more", "Review feedback"],
            "next_steps": "Continue with the next lesson"
        }
        return {"session_notes": session_notes}
