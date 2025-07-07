# graph/modeling_flow.py

import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# 1. Import all state-preserving nodes for the modeling flow
from agents.modelling_RAG_document_node import modelling_RAG_document_node
from agents.modeling_planner_node import modeling_planner_node
from agents.modelling_delivery_generator_node import modelling_delivery_generator_node
from agents.modelling_output_formatter import modelling_output_formatter_node
from agents.modelling_plan_advancer_node import modelling_plan_advancer_node
from agents.modelling_nlu_node import modelling_nlu_node

logger = logging.getLogger(__name__)

# 2. Define standardized node names for clarity and consistency
NODE_MODELLING_RAG = "modelling_rag"
NODE_MODELLING_PLANNER = "modelling_planner"
NODE_MODELLING_DELIVERY = "modelling_delivery_generator"
NODE_MODELLING_FORMATTER = "modelling_output_formatter" # CRITICAL: Add formatter node
NODE_MODELLING_PLAN_ADVANCER = "modelling_plan_advancer"
NODE_MODELLING_NLU = "modelling_nlu"

# 3. Define conditional routing functions

async def check_plan_completion(state: AgentGraphState) -> str:
    """Checks if the modeling plan is complete before delivering the next step."""
    plan = state.get("modeling_plan") # Note: planner uses 'modeling_plan'
    current_index = state.get("current_plan_step_index", 0)
    if not plan or current_index >= len(plan):
        logger.info("Modelling plan is complete or absent. Ending flow.")
        return END
    else:
        logger.info(f"Modelling plan active at step {current_index + 1}. Routing to delivery.")
        return NODE_MODELLING_DELIVERY

async def route_after_nlu(state: AgentGraphState) -> str:
    """After NLU, decides whether to re-explain the model or advance the plan."""
    intent = state.get("student_intent_for_model_turn", "CONFIRM_UNDERSTANDING")
    logger.info(f"Post-Interaction Router: Student intent is '{intent}'")
    if intent == "CONFIRM_UNDERSTANDING":
        logger.info("Student understands. Advancing the modeling plan.")
        return NODE_MODELLING_PLAN_ADVANCER
    else: # ASK_ABOUT_MODEL, GENERAL_QUESTION, STATE_CONFUSION
        logger.info(f"Student has a question or is confused. Re-running delivery for the current step.")
        return NODE_MODELLING_DELIVERY # Re-run delivery to re-explain

# 4. Define the subgraph creation function

def create_modeling_subgraph():
    """Creates the fully state-preserving, conversational LangGraph subgraph for the modeling module."""
    workflow = StateGraph(AgentGraphState)

    # Add all the state-preserving nodes
    workflow.add_node(NODE_MODELLING_RAG, modelling_RAG_document_node)
    workflow.add_node(NODE_MODELLING_PLANNER, modeling_planner_node)
    workflow.add_node(NODE_MODELLING_DELIVERY, modelling_delivery_generator_node)
    workflow.add_node(NODE_MODELLING_FORMATTER, modelling_output_formatter_node)
    workflow.add_node(NODE_MODELLING_PLAN_ADVANCER, modelling_plan_advancer_node)
    workflow.add_node(NODE_MODELLING_NLU, modelling_nlu_node)

    # --- Define the Conversational Flow ---
    # The entry point is now handled by the main graph's router.
    # This subgraph starts with either RAG (for planning) or NLU (for conversation).
    workflow.set_entry_point(NODE_MODELLING_RAG) # Default entry for planning

    # 1. Planning Flow: RAG -> Planner -> Check Completion
    workflow.add_edge(NODE_MODELLING_RAG, NODE_MODELLING_PLANNER)
    workflow.add_conditional_edges(
        NODE_MODELLING_PLANNER,
        check_plan_completion,
        {NODE_MODELLING_DELIVERY: NODE_MODELLING_DELIVERY, END: END}
    )

    # 2. Delivery Flow: Delivery -> Formatter -> END
    # The formatter is the TRUE end of the graph for a successful turn.
    workflow.add_edge(NODE_MODELLING_DELIVERY, NODE_MODELLING_FORMATTER)
    workflow.add_edge(NODE_MODELLING_FORMATTER, END)

    # 3. Conversational Loop Entry Point
    # The main graph router will direct conversational turns to this node.
    workflow.add_node("conversational_entry", route_after_nlu)
    workflow.add_conditional_edges(
        "conversational_entry",
        lambda x: x["student_intent_for_model_turn"],
        {
            NODE_MODELLING_PLAN_ADVANCER: NODE_MODELLING_PLAN_ADVANCER,
            NODE_MODELLING_DELIVERY: NODE_MODELLING_DELIVERY
        }
    )
    # This is slightly simplified; the main router would enter at NLU, then to here.
    # For clarity in the subgraph, we show the logic directly.
    # Let's refine this to be more accurate to the final architecture.
    
    # Corrected Conversational Flow:
    # Main router sends to NLU. NLU decides what's next.
    workflow.add_conditional_edges(
        NODE_MODELLING_NLU,
        route_after_nlu,
        {
            NODE_MODELLING_PLAN_ADVANCER: NODE_MODELLING_PLAN_ADVANCER,
            NODE_MODELLING_DELIVERY: NODE_MODELLING_DELIVERY
        }
    )

    # 4. After advancing the plan, check if we should deliver the next step or end.
    workflow.add_conditional_edges(
        NODE_MODELLING_PLAN_ADVANCER,
        check_plan_completion,
        {NODE_MODELLING_DELIVERY: NODE_MODELLING_DELIVERY, END: END}
    )

    # Compile and return the robust, state-preserving subgraph
    # Note: The main graph's router will be responsible for setting the entry point
    # to either NODE_MODELLING_RAG or NODE_MODELLING_NLU.
    return workflow.compile()
