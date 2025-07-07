# langgraph-service/agents/modelling_plan_advancer_node.py

import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def modelling_plan_advancer_node(state: AgentGraphState) -> dict:
    """
    Increments the current step index of the modelling plan while preserving
    the FULL session state.
    """
    logger.info("--- Executing Fully State-Preserving Modelling Plan Advancer Node ---")

    # --- Start of State Preservation ---
    modeling_plan = state.get("modeling_plan")
    activity_id = state.get("activity_id")
    learning_objective = state.get("Learning_Objective_Focus")
    student_proficiency = state.get("STUDENT_PROFICIENCY")
    student_affective_state = state.get("STUDENT_AFFECTIVE_STATE")
    # --- End of State Preservation ---
    
    current_index = state.get("current_plan_step_index", 0)
    new_index = current_index + 1
    
    logger.info(f"Advancing modelling plan from step {current_index} to {new_index}.")
    
    # Return the incremented index AND all the preserved session state keys.
    return {
        "current_plan_step_index": new_index,

        # Pass through the entire session state
        "modeling_plan": modeling_plan,
        "activity_id": activity_id,
        "Learning_Objective_Focus": learning_objective,
        "STUDENT_PROFICIENCY": student_proficiency,
        "STUDENT_AFFECTIVE_STATE": student_affective_state,
    }
