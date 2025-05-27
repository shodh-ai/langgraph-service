import logging
from state import AgentGraphState
import yaml
import os

logger = logging.getLogger(__name__)
PROMPTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "llm_prompts.yaml")

try:
    with open(PROMPTS_PATH, 'r') as file:
        PROMPTS = yaml.safe_load(file)
except Exception as e:
    logger.error(f"Failed to load LLM prompts: {e}")
    PROMPTS = {}

async def diagnose_submitted_speaking_response_node(state: AgentGraphState) -> dict:
    transcript = state.get("transcript", "")
    rubric = state.get("rubric_details", {})
    prompt_data = state.get("task_prompt", {})
    
    logger.info(f"SpeakingDiagnosticNode: Diagnosing speaking response")
    
    diagnosis = {
        "overall_score": 4,  # 1-5 scale
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
    
    logger.info(f"SpeakingDiagnosticNode: Completed diagnosis")
    
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