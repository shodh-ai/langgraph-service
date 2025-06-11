import json
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)


async def feedback_planner_node(state: AgentGraphState) -> dict:
    logger.info(
        f"FeedbackPlannerNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    diagnosis = state.get("explanation", "")
    user_data = state.get("user_data", {})
    data = state.get("document_data", [])

    if data == []:
        raise ValueError("No data provided")

    filtered_data = [
        {
            "Error": entry.get("Error", ""),
            "Diagnose": entry.get("Diagnose", ""),
            "Prioritize": entry.get("Prioritize", ""),
            "Explain Strategy": entry.get("Explain Strategy", ""),
            "Suggest Followup": entry.get("Suggest Followup", ""),
            "Consider Student State": entry.get("Consider Student State", ""),
        }
        for entry in data
    ]

    prompt = f"""
    You are a pedagogical planner for Rox AI Tutor.
    Current Student Diagnosis: {diagnosis}
    Student Profile: {user_data}

    Here are examples of how the experts have handled similar past situations:
    {filtered_data}

    Based on the current diagnosis and these expert examples:
    1. What specific error/issue should be prioritized for feedback right now?
    2. What is the core pedagogical strategy (e.g., 'SimpleSentenceTemplate_Fluency', 'MultiSensoryPhoneme_TH', 'RuleExplanation_SVA') that should be employed from the 'Explain Strategy' examples?
    
    Return a JSON object: 
    {{
        "prioritized_issue": "diagnosis_result",
        "chosen_pedagogical_strategy": "brief summary of strategy",
    }}
    """

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = model.generate_content(prompt)
        response_json = json.loads(response.text)
        prioritized_issue = response_json.get("prioritized_issue", "Unknown Issue")
        chosen_pedagogical_strategy = response_json.get(
            "chosen_pedagogical_strategy", ""
        )
        logger.info(f"Prioritized Issue: {prioritized_issue}")
        logger.info(f"Chosen Pedagogical Strategy: {chosen_pedagogical_strategy}")
        return {
            "prioritized_issue": prioritized_issue,
            "chosen_pedagogical_strategy": chosen_pedagogical_strategy,
        }
    except Exception as e:
        logger.error(f"Error processing with GenerativeModel: {e}")
        return {
            "prioritized_issue": "Unknown Issue",
            "chosen_pedagogical_strategy": "Error processing with GenerativeModel",
        }
