import logging
from state import AgentGraphState
import yaml
import os
import json
# Try to import vertexai, but don't fail if it's not available
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Content
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
logger.info(f"Using model: {type(gemini_model).__name__} in diagnostic_nodes")

async def diagnose_submitted_speaking_response_node(state: AgentGraphState) -> dict:
    transcript = state.get("transcript", "")
    rubric = state.get("rubric_details", {})
    prompt_data = state.get("task_prompt", {})
    
    logger.info(f"SpeakingDiagnosticNode: Diagnosing speaking response")
    
    if not gemini_model:
        logger.warning("SpeakingDiagnosticNode: Gemini model not available, using stub implementation")
        # Fallback to stub implementation
        diagnosis = {
            "overall_score": 4,
            "primary_strength": "clear organization of ideas",
            "primary_error": "limited vocabulary range",
            "skill_scores": {
                "delivery": 3.5,
                "language_use": 4.0,
                "topic_development": 4.2
            },
            "detailed_feedback": {
                "delivery": "Good pace but occasional hesitations",
                "language_use": "Appropriate grammar with some minor errors",
                "topic_development": "Well-organized with good supporting examples"
            }
        }
    else:
        try:
            # get prompt templates from config
            system_prompt = PROMPTS.get("diagnostic", {}).get("speaking", {}).get("system", "")
            user_prompt = PROMPTS.get("diagnostic", {}).get("speaking", {}).get("user_template", "")
            
            # replace placeholders in prompts
            system_prompt = system_prompt.replace("{{rubric_details}}", json.dumps(rubric))
            system_prompt = system_prompt.replace("{{task_prompt}}", json.dumps(prompt_data))
            
            user_prompt = user_prompt.replace("{{transcript}}", transcript)
            
            contents = [
                Content(role="user", parts=[system_prompt]),
                Content(role="model", parts=["I understand. I'll evaluate the speaking response based on the provided rubric and prompt."]),
                Content(role="user", parts=[user_prompt])
            ]
            
            response = gemini_model.generate_content(contents, generation_config={
                "temperature": 0.2,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json"
            })
            
            try:
                diagnosis = json.loads(response.text)
                logger.info("SpeakingDiagnosticNode: Successfully parsed Gemini response")
            except json.JSONDecodeError:
                logger.error(f"SpeakingDiagnosticNode: Failed to parse Gemini response as JSON: {response.text}")
                # Fallback to stub implementation
                diagnosis = {
                    "overall_score": 3.5,
                    "primary_strength": "addressing the prompt directly",
                    "primary_error": "inconsistent pronunciation",
                    "skill_scores": {
                        "delivery": 3.0,
                        "language_use": 3.5,
                        "topic_development": 4.0
                    },
                    "detailed_feedback": {
                        "delivery": "Some pronunciation issues affect clarity",
                        "language_use": "Good vocabulary with some grammatical errors",
                        "topic_development": "Clear main points but could use more specific examples"
                    }
                }
        except Exception as e:
            logger.error(f"SpeakingDiagnosticNode: Error calling Gemini API: {e}")
            # Fallback to stub implementation
            diagnosis = {
                "overall_score": 3.0,
                "primary_strength": "attempt to address the prompt",
                "primary_error": "technical error during evaluation",
                "skill_scores": {
                    "delivery": 3.0,
                    "language_use": 3.0,
                    "topic_development": 3.0
                },
                "detailed_feedback": {
                    "delivery": "Unable to fully evaluate due to technical issues",
                    "language_use": "Unable to fully evaluate due to technical issues",
                    "topic_development": "Unable to fully evaluate due to technical issues"
                }
            }
    
    logger.info(f"SpeakingDiagnosticNode: Completed diagnosis with overall score {diagnosis.get('overall_score')}")
    
    return {"speaking_diagnosis": diagnosis, "diagnosis_result": diagnosis}

async def diagnose_submitted_writing_response_node(state: AgentGraphState) -> dict:
    transcript = state.get("transcript", "")
    rubric = state.get("rubric_details", {})
    prompt_data = state.get("task_prompt", {})
    
    logger.info(f"WritingDiagnosticNode: Diagnosing writing response")
    
    diagnosis = {
        "overall_score": 4.2,  # 1-5 scale
        "primary_strength": "strong thesis and supporting arguments",
        "primary_error": "some grammatical inconsistencies",
        "skill_scores": {
            "organization": 4.5,
            "language_use": 3.8,
            "content_development": 4.3
        },
        "detailed_feedback": {
            "organization": "Well-structured with clear introduction and conclusion",
            "language_use": "Good vocabulary with some grammatical errors",
            "content_development": "Strong arguments with relevant examples"
        }
    }
    
    logger.info(f"WritingDiagnosticNode: Completed diagnosis")
    
    return {"writing_diagnosis": diagnosis, "diagnosis_result": diagnosis}

async def analyze_live_writing_chunk_node(state: AgentGraphState) -> dict:
    transcript = state.get("transcript", "")
    context = state.get("current_context")
    
    logger.info(f"LiveWritingAnalysisNode: Analyzing live writing chunk")
    
    analysis = {
        "suggestions": [],
        "strengths": [],
        "immediate_issues": []
    }
    #stub implementation
    if "however" in transcript and "but" in transcript:
        analysis["suggestions"].append({
            "type": "redundancy",
            "text": "Consider using either 'however' or 'but', not both",
            "severity": "low"
        })
    
    if transcript.count(".") > 3 and len(transcript) < 100:
        analysis["suggestions"].append({
            "type": "sentence_length",
            "text": "Consider combining some short sentences for better flow",
            "severity": "medium"
        })
    
    logger.info(f"LiveWritingAnalysisNode: Completed analysis with {len(analysis['suggestions'])} suggestions")
    
    return {"live_writing_analysis": analysis}