# agents/modelling_plan_advancer_node.py
import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def modelling_plan_advancer_node(state: AgentGraphState) -> dict:
    """
    Increments the plan step index for the modelling flow.
    """
    logger.info("--- Executing Modelling Plan Advancer Node ---")
    
    current_index = state.get("current_plan_step_index", 0)
    new_index = current_index + 1
    
    logger.info(f"Advancing modelling plan from step {current_index} to {new_index}.")
    
    # Return ONLY the updated index. The checkpointer handles preserving the rest of the state.
    return {"current_plan_step_index": new_index}
