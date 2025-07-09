# agents/modellingg_planner_node.py

import logging
import json
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def modellingg_planner_node(state: AgentGraphState) -> dict:
    """
    Generates a step-by-step plan for a modellingg activity and promotes
    initial context to top-level state keys for the entire session.
    This is the entry point for a new modellingg session.
    """
    logger.info("---Executing modellingg Planner Node---")
    try:
        context = state.get("current_context", {})
        learning_objective = context.get("Learning_Objective_Focus")
        student_proficiency = context.get("STUDENT_PROFICIENCY")
        activity_id = context.get("activity_id") # Unique ID for the modellingg activity

        if not all([learning_objective, student_proficiency, activity_id]):
            raise ValueError("Missing critical context for modellingg plan generation.")

        json_output_example = """
        [
            {"step": 1, "focus": "Introduction to the core concept and demonstration of the first part."},
            {"step": 2, "focus": "Guided practice on the second part of the concept."},
            {"step": 3, "focus": "Student attempts the full concept with scaffolding."},
            {"step": 4, "focus": "Review and feedback on the student's attempt."}
        ]
        """

        prompt = f"""
        You are an expert AI instructional designer for TOEFL students.
        Your task is to create a step-by-step plan for a modellingg activity.

        **Learning Objective:** "{learning_objective}"
        **Student's Proficiency:** "{student_proficiency}"

        Based on the objective and proficiency, create a 3-5 step plan for a modellingg exercise.
        The plan should guide the student through observing, practicing, and applying the concept.

        Your output MUST be ONLY a single, valid JSON array of objects, where each object has a "step" and "focus" key.
        Do not include any other text, just the JSON array.

        Example:
        {json_output_example}
        """

        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(prompt)
        modellingg_plan = json.loads(response.text)

        if not isinstance(modellingg_plan, list):
            raise TypeError("LLM did not return a valid list for the modellingg plan.")

        logger.info(f"Successfully generated a {len(modellingg_plan)}-stepmodellingng plan.")

        # This is the one-time promotion of context to top-level state
        return {
            "modellingg_plan":modellingng_plan,
            "current_plan_step_index": 0,
            # Promote keys for the entire session
            "activity_id": activity_id,
            "Learning_Objective_Focus": learning_objective,
            "STUDENT_PROFICIENCY": student_proficiency,
            "STUDENT_AFFECTIVE_STATE": context.get("STUDENT_AFFECTIVE_STATE"),
        }

    except Exception as e:
        logger.error(f"modellinggPlannerNode: CRITICAL FAILURE: {e}", exc_info=True)
        # Route to an error handler if something goes wrong
        return {"error_message": f"Failed to generate modellingg plan: {e}", "route_to_error_handler": True}
