# agents/teaching_plan_advancer_node.py (The Definitive Version)
import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def teaching_plan_advancer_node(state: AgentGraphState) -> dict:
    """
    Increments the plan step index AND preserves the plan itself for the next node.
    """
    logger.info("--- Executing Teaching Plan Advancer Node (State-Preserving) ---")
    
    current_index = state.get("current_plan_step_index", 0)
    new_index = current_index + 1
    
    logger.info(f"Advancing lesson plan from step {current_index} to {new_index}.")
    
    # Get the existing plan from the state that was passed to this node.
    plan = state.get("pedagogical_plan")
    
    # Return BOTH the new index AND the existing plan.
    return {
        "current_plan_step_index": new_index,
        "pedagogical_plan": plan 
    }