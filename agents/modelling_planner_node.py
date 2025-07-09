# agents/modelling_planner_node.py
import logging
import json
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def modelling_planner_node(state: AgentGraphState) -> dict:
    """
    Generates a concrete, step-by-step plan for a modelling session AND promotes
    the initial context from the nested 'current_context' to top-level state keys,
    establishing the state for the entire modelling session.
    """
    logger.info("--- Executing Modelling Planner Node (with State Promotion) ---")
    try:
        current_context = state.get("current_context", {})
        
        # Extract data from the initial turn's context to use in the prompt and to promote to state.
        learning_objective = current_context.get('Learning_Objective_Focus')
        student_proficiency = current_context.get('STUDENT_PROFICIENCY')
        student_affect = current_context.get('STUDENT_AFFECTIVE_STATE')
        lesson_id = current_context.get('lesson_id')
        student_submission = current_context.get('student_task_submission')

        llm_prompt = f"""
        You are an AI Pedagogical Strategist. Your task is to create a step-by-step plan for a modelling session where you will provide feedback and show a model answer.
        CONTEXT:
        - Learning Objective: '{learning_objective}'
        - Student's Proficiency: {student_proficiency}
        - Student's Submission: {student_submission}

        INSTRUCTIONS:
        You MUST devise an ordered pedagogical plan as a JSON list of step objects.
        Each object MUST have a "step_type" and a short, descriptive "focus".
        Possible step_types: 'DELIVER_FEEDBACK', 'SHOW_MODEL_ANSWER', 'EXPLAIN_KEY_CONCEPT'.
        Example of the required output format:
        [
          {{"step_type": "DELIVER_FEEDBACK", "focus": "Overall impression and key strengths"}},
          {{"step_type": "SHOW_MODEL_ANSWER", "focus": "Presenting a complete model answer for comparison"}}
        ]
        """
        
        model = genai.GenerativeModel("gemini-2.0-flash", generation_config=GenerationConfig(response_mime_type="application/json"))
        response = await model.generate_content_async(llm_prompt)
        
        plan_json = json.loads(response.text)
        if not isinstance(plan_json, list):
            raise ValueError("LLM did not return a valid list for the plan.")

        logger.info(f"LLM generated modelling plan with {len(plan_json)} steps.")

        # Promote the initial context to top-level state keys for the session.
        return {
            "pedagogical_plan": plan_json,
            "current_plan_step_index": 0,
            "current_plan_active": True,
            "Learning_Objective_Focus": learning_objective,
            "STUDENT_PROFICIENCY": student_proficiency,
            "STUDENT_AFFECTIVE_STATE": student_affect,
            "lesson_id": lesson_id
        }

    except Exception as e:
        logger.error(f"ModellingPlannerNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": f"Failed to plan modelling session: {e}"}
