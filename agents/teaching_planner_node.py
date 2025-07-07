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
    Generates a concrete, step-by-step lesson plan and promotes initial
    context to top-level state keys for the rest of the session.
    """
    logger.info("--- Executing Teaching Planner Node (Strict Prompting) ---")
    try:
        rag_documents = state.get("rag_document_data")
        rag_context_examples = json.dumps(rag_documents, indent=2) if rag_documents else "[]"
        current_context = state.get("current_context", {})
        
        # Extract data from the initial turn's context to use in the prompt and to promote to state
        learning_objective = current_context.get('Learning_Objective_Focus')
        student_proficiency = current_context.get('STUDENT_PROFICIENCY')
        student_affect = current_context.get('STUDENT_AFFECTIVE_STATE')
        lesson_id = current_context.get('lesson_id')

        # --- THIS IS THE NEW, STRICTER PROMPT ---
        llm_prompt = f"""
        You are an AI Lesson Planner. Your SOLE task is to create a step-by-step lesson plan.

        CONTEXT:
        - Learning Objective: '{learning_objective}'
        - Student's Proficiency: {student_proficiency}

        INSTRUCTIONS:
        You MUST devise an ordered pedagogical plan as a JSON list of step objects.
        Each object in the list MUST have a "step_type" and a short, descriptive "focus".
        Possible step_types: 'EXPLAIN_CONCEPT', 'SHOW_MODEL', 'PROVIDE_SCAFFOLDED_PRACTICE', 'CHECK_UNDERSTANDING_QA'.
        
        Do NOT output a high-level strategy document. Do NOT use keys like 'PedagogicalStrategies'.
        You MUST return ONLY the JSON list of steps.

        Example of the required output format:
        [
          {{"step_type": "EXPLAIN_CONCEPT", "focus": "Core principles of the topic"}},
          {{"step_type": "CHECK_UNDERSTANDING_QA", "focus": "Check understanding of core principles"}},
          {{"step_type": "SHOW_MODEL", "focus": "Modeling the concept in an example"}},
          {{"step_type": "PROVIDE_SCAFFOLDED_PRACTICE", "focus": "Write a guided practice item"}}
        ]
        """
        
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config=GenerationConfig(response_mime_type="application/json"))
        response = await model.generate_content_async(llm_prompt)
        
        # It's safer to load the JSON and validate it's a list
        parsed_response = json.loads(response.text)
        if not isinstance(parsed_response, list):
            logger.error(f"LLM returned a plan that is not a list: {parsed_response}")
            raise ValueError("LLM did not return a valid list for the lesson plan.")

        lesson_plan_json = parsed_response
        logger.info(f"LLM generated lesson plan with {len(lesson_plan_json)} steps.")

        # Promote the initial context to top-level state keys for the session
        return {
            "pedagogical_plan": lesson_plan_json,
            "current_plan_step_index": 0,
            "current_plan_active": True,
            "Learning_Objective_Focus": learning_objective,
            "STUDENT_PROFICIENCY": student_proficiency,
            "STUDENT_AFFECTIVE_STATE": student_affect,
            "lesson_id": lesson_id
        }

    except Exception as e:
        logger.error(f"TeachingPlannerNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": f"Failed to plan lesson: {e}", "route_to_error_handler": True}