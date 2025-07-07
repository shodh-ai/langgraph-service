# agents/teaching_planner_node.py (The Definitive, Fixed Version)
import logging
import json
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def teaching_planner_node(state: AgentGraphState) -> dict:
    """
    Generates a concrete, step-by-step lesson plan AND promotes the initial
    context from the nested 'current_context' to top-level state keys,
    establishing the state for the entire teaching session.
    """
    logger.info("--- Executing Teaching Planner Node (with State Promotion) ---")
    try:
        rag_documents = state.get("rag_document_data")
        rag_context_examples = json.dumps(rag_documents, indent=2) if rag_documents else "[]"
        current_context = state.get("current_context", {})
        
        # --- THIS IS THE FIX (Part 1): Read from the nested context ---
        # Extract data from the initial turn's context to use in the prompt and to promote to state.
        learning_objective = current_context.get('Learning_Objective_Focus')
        student_proficiency = current_context.get('STUDENT_PROFICIENCY')
        student_affect = current_context.get('STUDENT_AFFECTIVE_STATE')
        lesson_id = current_context.get('lesson_id')

        llm_prompt = f"""
        You are an AI Lesson Planner. Your SOLE task is to create a step-by-step lesson plan.
        CONTEXT:
        - Learning Objective: '{learning_objective}'
        - Student's Proficiency: {student_proficiency}
        INSTRUCTIONS:
        You MUST devise an ordered pedagogical plan as a JSON list of step objects.
        Each object in the list MUST have a "step_type" and a short, descriptive "focus".
        Example of the required output format:
        [
          {{"step_type": "EXPLAIN_CONCEPT", "focus": "Core principles of the topic"}},
          {{"step_type": "SHOW_MODEL", "focus": "Modeling the concept in an example"}}
        ]
        """
        
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config=GenerationConfig(response_mime_type="application/json"))
        response = await model.generate_content_async(llm_prompt)
        
        lesson_plan_json = json.loads(response.text)
        if not isinstance(lesson_plan_json, list):
            raise ValueError("LLM did not return a valid list for the lesson plan.")

        logger.info(f"LLM generated lesson plan with {len(lesson_plan_json)} steps.")

        # --- THIS IS THE FIX (Part 2): Return the promoted keys at the top level ---
        # This establishes the state for the rest of the teaching session.
        return {
            "pedagogical_plan": lesson_plan_json,
            "current_plan_step_index": 0,
            "current_plan_active": True,
            # Promote the initial context to top-level state keys
            "Learning_Objective_Focus": learning_objective,
            "STUDENT_PROFICIENCY": student_proficiency,
            "STUDENT_AFFECTIVE_STATE": student_affect,
            "lesson_id": lesson_id
        }

    except Exception as e:
        logger.error(f"TeachingPlannerNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": f"Failed to plan lesson: {e}", "route_to_error_handler": True}