import logging
import json
from typing import Dict, Any, List
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from state import AgentGraphState
from graph.utils import query_knowledge_base # Import the robust RAG query utility

logger = logging.getLogger(__name__)

def _create_lesson_plan_data(lesson_plan_steps: List[Dict]) -> Dict:
    """Formats the lesson plan steps into a React Flow compatible JSON graph."""
    nodes = []
    edges = []
    
    if not lesson_plan_steps:
        return {"nodes": [], "edges": []}

    for i, step in enumerate(lesson_plan_steps):
        step_id = step.get("step_id")
        # Determine status: first step is 'next_up', others are 'unlocked'
        status = "next_up" if i == 0 else "unlocked"
        
        nodes.append({
            "id": f"step_{step_id}",
            "data": {
                "label": step.get("focus", "Unnamed Step"),
                "status": status,
                "modality": step.get("modality", "UNKNOWN")
            }
        })
        
        # Create an edge from the previous step to the current one
        if i > 0:
            prev_step_id = lesson_plan_steps[i-1].get("step_id")
            edges.append({
                "id": f"e-step_{prev_step_id}-step_{step_id}",
                "source": f"step_{prev_step_id}",
                "target": f"step_{step_id}"
            })
            
    return {"nodes": nodes, "edges": edges}


# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def pedagogical_strategy_planner_node(state: AgentGraphState) -> dict:
    """
    The MESO PLANNER.
    Creates a multi-step lesson plan and the visual data for it.
    """
    logger.info("--- Executing Pedagogical Strategy Planner Node (Meso Planner) ---")
    try:
        user_id = state["user_id"]
        student_profile = state.get("student_memory_context", {})
        active_persona = state.get("active_persona", "The Structuralist")
        current_lo = state.get("current_lo_to_address", {})

        if not current_lo:
            raise ValueError("Cannot create a lesson plan without a 'current_lo_to_address' in the state.")

        lo_title = current_lo.get('title')
        rag_query = f"How to teach a student to '{lo_title}'?"
        retrieved_teaching_strategies = await query_knowledge_base(rag_query, category="pedagogy_for_lo")
        
        prompt = f"""
        You are an expert AI Lesson Planner for a TOEFL tutor, embodying the '{active_persona}' persona.
        Your task is to create a concrete, multi-step lesson plan for a specific Learning Objective (LO).

        STUDENT'S PROFILE:
        {json.dumps(student_profile, indent=2)}

        LEARNING OBJECTIVE TO TEACH:
        {json.dumps(current_lo, indent=2)}

        EXPERT TEACHING STRATEGIES for this LO:
        {json.dumps(retrieved_teaching_strategies, indent=2)}

        INSTRUCTIONS:
        1.  Create a sequence of 2-4 lesson steps to teach the specified LO.
        2.  Each step must have a `modality` (TEACH, MODEL, SCAFFOLDED_PRACTICE, or UNGUIDED_PRACTICE) and a `focus` (a short description of the step's goal).
        3.  The `modality` determines which page/UI the student will be sent to.
        4.  The sequence should follow a logical pedagogical progression (e.g., teach, then model, then practice).
        5.  The `focus` for each step should be a clear, student-facing instruction.

        Return a single, valid JSON object with one key, "lesson_plan_steps", which is a list of step objects.
        
        Example Output:
        {{
            "lesson_plan_steps": [
                {{ "step_id": 1, "modality": "TEACH", "focus": "Introduce the core definition of a thesis statement and its purpose.", "target_page": "P7_Teaching" }},
                {{ "step_id": 2, "modality": "MODEL", "focus": "Analyze examples of strong and weak thesis statements.", "target_page": "P8_Modeling" }},
                {{ "step_id": 3, "modality": "SCAFFOLDED_PRACTICE", "focus": "Guide the student in identifying flaws in weak thesis statements.", "target_page": "P4_Cowriting" }}
            ]
        }}
        """

        model = genai.GenerativeModel("gemini-2.0-flash", generation_config=GenerationConfig(response_mime_type="application/json"))
        response = await model.generate_content_async(prompt)
        planner_output = json.loads(response.text)
        lesson_plan_steps = planner_output.get("lesson_plan_steps", [])

        logger.info(f"Generated a {len(lesson_plan_steps)}-step lesson plan for LO '{lo_title}'.")
        
        lesson_plan_graph_data = _create_lesson_plan_data(lesson_plan_steps)
        logger.info(f"Generated lesson plan graph data with {len(lesson_plan_graph_data['nodes'])} steps.")
        
        return {
            "pedagogical_plan": lesson_plan_steps,
            "current_plan_step_index": 0,
            "current_plan_active": True,
            "lesson_plan_graph_data": lesson_plan_graph_data
        }
        
    except Exception as e:
        logger.error(f"PedagogicalStrategyPlannerNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": f"Failed to create lesson plan: {e}"}