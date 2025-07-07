# graph/modeling_flow.py (Refactored for Conversational Interaction)
import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# 1. Import agent node functions
from agents.modelling_RAG_document_node import modelling_RAG_document_node
from agents.modelling_planner_node import modelling_planner_node
from agents.modelling_delivery_generator_node import modelling_delivery_generator_node
from agents.modelling_plan_advancer_node import modelling_plan_advancer_node
from agents.modeling_nlu_node import modeling_nlu_node # <<< IMPORT NEW NLU NODE

logger = logging.getLogger(__name__)

# 2. Define standardized node names
NODE_MODELING_RAG = "modelling_rag"
NODE_MODELING_PLANNER = "modelling_planner"
NODE_MODELING_DELIVERY = "modelling_delivery_generator"
NODE_MODELING_PLAN_ADVANCER = "modelling_plan_advancer"
NODE_MODELING_NLU = "modeling_nlu" # <<< ADD NEW NLU NODE NAME

# 3. Define conditional routing functions
async def entry_router(state: AgentGraphState) -> str:
    """Routes based on the task stage to either start a new plan or handle a conversational turn."""
    task_stage = state.get("current_context", {}).get("task_stage")
    logger.info(f"Modeling Subgraph: Routing for task '{task_stage}'.")
    if task_stage == "MODELLING_PAGE_TURN":
        logger.info("Modeling Subgraph: Conversational turn. Routing to NLU handler.")
        return NODE_MODELING_NLU
    else:
        logger.info("Modeling Subgraph: Initial turn. Routing to RAG for planning.")
        return NODE_MODELING_RAG

async def after_interaction_router(state: AgentGraphState) -> str:
    """After NLU, decides whether to re-explain the model or advance the plan."""
    intent = state.get("student_intent_for_model_turn", "CONFIRM_UNDERSTANDING")
    logger.info(f"Post-Interaction Router: Student intent is '{intent}'")
    if intent == "CONFIRM_UNDERSTANDING":
        logger.info("Student understands. Advancing the modeling plan.")
        return NODE_MODELING_PLAN_ADVANCER
    else: # ASK_ABOUT_MODEL, GENERAL_QUESTION, STATE_CONFUSION
        logger.info(f"Student has a question or is confused. Re-running delivery for the current step.")
        return NODE_MODELING_DELIVERY # Re-run delivery to re-explain

async def delivery_or_end_router(state: AgentGraphState) -> str:
    """Checks if the modeling plan is complete before delivering the next step."""
    plan = state.get("modelling_plan")
    current_index = state.get("current_plan_step_index", 0)
    if not plan or current_index >= len(plan):
        logger.info("Modeling plan is complete or absent. Ending flow.")
        return END
    else:
        logger.info(f"Modeling plan active at step {current_index + 1}. Routing to delivery.")
        return NODE_MODELING_DELIVERY

# 4. Define the subgraph creation function
def create_modeling_subgraph():
    """Creates the conversational LangGraph subgraph for the modeling module."""
    workflow = StateGraph(AgentGraphState)

    # Add the nodes
    workflow.add_node(NODE_MODELING_RAG, modelling_RAG_document_node)
    workflow.add_node(NODE_MODELING_PLANNER, modelling_planner_node)
    workflow.add_node(NODE_MODELING_DELIVERY, modelling_delivery_generator_node)
    workflow.add_node(NODE_MODELING_PLAN_ADVANCER, modelling_plan_advancer_node)
    workflow.add_node(NODE_MODELING_NLU, modeling_nlu_node)

    # --- Define the Conversational Flow ---
    # 1. Entry Point: Start a new plan or handle a user's question?
    workflow.set_conditional_entry_point(
        entry_router,
        {
            NODE_MODELING_RAG: NODE_MODELING_RAG,
            NODE_MODELING_NLU: NODE_MODELING_NLU
        }
    )

    # 2. Planning Flow
    workflow.add_edge(NODE_MODELING_RAG, NODE_MODELING_PLANNER)
    workflow.add_conditional_edges(
        NODE_MODELING_PLANNER,
        delivery_or_end_router,
        {NODE_MODELING_DELIVERY: NODE_MODELING_DELIVERY, END: END}
    )

    # 3. Delivery -> End (The graph's work for this turn is done)
    workflow.add_edge(NODE_MODELING_DELIVERY, END)

    # 4. Conversational Loop
    workflow.add_conditional_edges(
        NODE_MODELING_NLU,
        after_interaction_router,
        {
            NODE_MODELING_PLAN_ADVANCER: NODE_MODELING_PLAN_ADVANCER,
            NODE_MODELING_DELIVERY: NODE_MODELING_DELIVERY
        }
    )

    # 5. After advancing the plan, check if we should deliver the next step or end
    workflow.add_conditional_edges(
        NODE_MODELING_PLAN_ADVANCER,
        delivery_or_end_router,
        {NODE_MODELING_DELIVERY: NODE_MODELING_DELIVERY, END: END}
    )

    # Compile and return the subgraph
    return workflow.compile()
