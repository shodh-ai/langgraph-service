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
        logger.info("Vertex AI initialized in practice_selector_node")
except Exception as e:
    logger.error(f"Failed to initialize Vertex AI in practice_selector_node: {e}")

# Load Gemini model if not already loaded
try:
    gemini_model = GenerativeModel("gemini-1.5-pro")
    logger.info("Gemini model loaded in practice_selector_node")
except Exception as e:
    logger.error(f"Failed to load Gemini model in practice_selector_node: {e}")
    gemini_model = None

async def select_next_practice_or_drill_node(state: AgentGraphState) -> dict:
    """Chooses or generates skill drills/practice tasks using Vertex AI Gemini."""
    diagnosis = state.get("diagnosis_result", {})
    student_data = state.get("student_memory_context", {})
    context = state.get("current_context")
    task_prompt = state.get("task_prompt", {})
    
    logger.info(f"PracticeSelectorNode: Selecting next practice or drill based on diagnosis")
    
    if not gemini_model:
        logger.warning("PracticeSelectorNode: Gemini model not available, using stub implementation")
        # Fallback to stub implementation
        practice = {
            "drill_id": "general_speaking_practice_1",
            "drill_type": "speaking_practice",
            "focus_area": "general",
            "difficulty": "medium"
        }
        
        # Select practice based on diagnosis
        if diagnosis.get("primary_error") == "limited vocabulary range":
            practice = {
                "drill_id": "vocabulary_expansion_drill_1",
                "drill_type": "vocabulary_drill",
                "focus_area": "academic_vocabulary",
                "difficulty": "medium"
            }
        elif diagnosis.get("primary_error") == "some grammatical inconsistencies":
            practice = {
                "drill_id": "grammar_correction_drill_1",
                "drill_type": "grammar_drill",
                "focus_area": "verb_tense_consistency",
                "difficulty": "medium"
            }
        
        # Adjust difficulty based on student profile
        profile = student_data.get("profile", {})
        level = profile.get("level", "Intermediate")
        
        if level == "Beginner":
            practice["difficulty"] = "easy"
        elif level == "Advanced":
            practice["difficulty"] = "hard"
    else:
        try:
            # Get prompt templates from config
            system_prompt = PROMPTS.get("practice", {}).get("selector", {}).get("system", "")
            user_prompt = PROMPTS.get("practice", {}).get("selector", {}).get("user_template", "")
            
            # If prompts are not found in config, use default prompts
            if not system_prompt:
                system_prompt = """You are an expert TOEFL tutor AI responsible for selecting appropriate practice exercises and skill drills for students.
                
                Based on the student's diagnosis results, profile data, and current task context, select or generate a practice exercise that will help them improve their weakest areas while building on their strengths.
                
                Student diagnosis: {{diagnosis_result}}
                Student profile data: {{student_memory_context}}
                Current task context: {{current_context}}
                
                Select a practice exercise that is tailored to the student's specific needs."""
            
            if not user_prompt:
                user_prompt = """Select a practice exercise for this student based on their diagnosis and profile.
                
                Return your selection in the following JSON format:
                {
                  "drill_id": "[unique identifier for the drill]",
                  "drill_type": "[speaking_practice|writing_practice|vocabulary_drill|grammar_drill|listening_practice|reading_practice]",
                  "focus_area": "[specific skill area to focus on]",
                  "difficulty": "[easy|medium|hard]",
                  "description": "[brief description of the exercise]",
                  "rationale": "[explanation of why this exercise was selected]"
                }
                
                Make sure the practice is specifically tailored to address the student's primary error while being appropriate for their level."""
            
            # Replace placeholders in prompts
            system_prompt = system_prompt.replace("{{diagnosis_result}}", json.dumps(diagnosis))
            system_prompt = system_prompt.replace("{{student_memory_context}}", json.dumps(student_data))
            system_prompt = system_prompt.replace("{{current_context}}", json.dumps(context.dict() if hasattr(context, "dict") else context))
            system_prompt = system_prompt.replace("{{task_prompt}}", json.dumps(task_prompt))
            
            # Create Gemini content
            contents = [
                Content(role="user", parts=[system_prompt]),
                Content(role="model", parts=["I understand. I'll select an appropriate practice exercise based on the student's diagnosis and profile."]),
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
                    practice = json.loads(json_text)
                elif "```" in response_text:
                    json_text = response_text.split("```")[1].split("```")[0].strip()
                    practice = json.loads(json_text)
                else:
                    # Try to parse the whole response as JSON
                    practice = json.loads(response_text)
                
                logger.info(f"PracticeSelectorNode: Successfully parsed JSON practice selection")
                
                # Ensure the practice has the expected structure
                required_fields = ["drill_id", "drill_type", "focus_area", "difficulty"]
                for field in required_fields:
                    if field not in practice:
                        if field == "drill_id":
                            practice[field] = f"generated_drill_{hash(response_text) % 10000}"
                        elif field == "drill_type":
                            section = getattr(context, "toefl_section", "Speaking").lower()
                            practice[field] = f"{section}_practice"
                        elif field == "focus_area":
                            practice[field] = diagnosis.get("primary_error", "general").replace(" ", "_").lower()
                        elif field == "difficulty":
                            practice[field] = "medium"
                
                # Add optional fields if not present
                if "description" not in practice:
                    practice["description"] = f"Practice exercise focusing on {practice['focus_area'].replace('_', ' ')}"
                if "rationale" not in practice:
                    practice["rationale"] = f"Selected to address {diagnosis.get('primary_error', 'skill development')}"
                    
            except json.JSONDecodeError as e:
                logger.error(f"PracticeSelectorNode: Failed to parse JSON response: {e}")
                logger.error(f"Response text: {response_text}")
                
                # Extract key information from text response
                focus_area = "general"
                drill_type = "speaking_practice"
                difficulty = "medium"
                
                if "vocabulary" in response_text.lower():
                    focus_area = "vocabulary"
                    drill_type = "vocabulary_drill"
                elif "grammar" in response_text.lower():
                    focus_area = "grammar"
                    drill_type = "grammar_drill"
                elif "writing" in response_text.lower():
                    drill_type = "writing_practice"
                elif "reading" in response_text.lower():
                    drill_type = "reading_practice"
                elif "listening" in response_text.lower():
                    drill_type = "listening_practice"
                
                if "beginner" in response_text.lower() or "easy" in response_text.lower():
                    difficulty = "easy"
                elif "advanced" in response_text.lower() or "hard" in response_text.lower():
                    difficulty = "hard"
                
                # Create a practice object from extracted information
                practice = {
                    "drill_id": f"extracted_drill_{hash(response_text) % 10000}",
                    "drill_type": drill_type,
                    "focus_area": focus_area,
                    "difficulty": difficulty,
                    "description": "Practice exercise generated from text response",
                    "rationale": "Selected based on diagnosis and profile"
                }
                
        except Exception as e:
            logger.error(f"PracticeSelectorNode: Error calling Gemini API: {e}")
            # Fallback to stub implementation
            practice = {
                "drill_id": "fallback_practice_1",
                "drill_type": "general_practice",
                "focus_area": diagnosis.get("primary_error", "general").replace(" ", "_").lower(),
                "difficulty": "medium",
                "description": "General practice exercise",
                "rationale": "Fallback selection due to API error"
            }
            
            # Adjust difficulty based on student profile
            profile = student_data.get("profile", {})
            level = profile.get("level", "Intermediate")
            
            if level == "Beginner":
                practice["difficulty"] = "easy"
            elif level == "Advanced":
                practice["difficulty"] = "hard"
    
    logger.info(f"PracticeSelectorNode: Selected drill_id: {practice['drill_id']}, focus: {practice['focus_area']}, difficulty: {practice['difficulty']}")
    
    return {"next_practice": practice}