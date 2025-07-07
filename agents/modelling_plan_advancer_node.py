# langgraph-service/agents/modelling_plan_advancer_node.py
import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def modelling_plan_advancer_node(state: AgentGraphState) -> dict:
    """
    Increments the current step index of the modelling plan.
    """
    logger.info("--- Executing Modelling Plan Advancer Node ---")
    
    current_index = state.get("current_plan_step_index", 0)
    new_index = current_index + 1
    
    logger.info(f"Advancing modelling plan from step {current_index} to {new_index}.")
    
    # Get the existing plan and other critical state to ensure they are not lost.
    plan = state.get("modelling_plan")
    intermediate_payload = state.get("intermediate_modelling_payload")
    current_context = state.get("current_context")

    # Persist the plan, the new index, and all other critical state keys.
    return {
        "current_plan_step_index": new_index,
        "modelling_plan": plan,
        "intermediate_modelling_payload": intermediate_payload,
        "current_context": current_context
    }
