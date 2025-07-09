# graph/modelling_flow.py
import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState

logger = logging.getLogger(__name__)

# Import node functions from the modernized agent files
from agents.modelling_planner_node import modelling_planner_node
from agents.modelling_delivery_node import modelling_delivery_generator_node
from agents.modelling_plan_advancer_node import modelling_plan_advancer_node
from agents.modelling_nlu_node import modelling_nlu_node
from agents.modelling_RAG_document_node import modelling_RAG_document_node
from agents.modelling_output_formatter import modelling_output_formatter_node

# --- Node Names ---
NODE_MODELLING_RAG = "modelling_rag_document"
NODE_MODELLING_PLANNER = "modelling_planner"
NODE_MODELLING_DELIVERY_GENERATOR = "modelling_delivery_generator"
NODE_MODELLING_NLU = "modelling_nlu"
NODE_MODELLING_PLAN_ADVANCER = "modelling_plan_advancer"
NODE_CHECK_PLAN_COMPLETION = "check_modelling_plan_completion"
NODE_MODELLING_OUTPUT_FORMATTER = "modelling_output_formatter"

async def check_plan_completion_node(state: AgentGraphState) -> dict:
    """A placeholder node to prepare state for the delivery router."""
    logger.info("--- Checking if modelling plan is complete ---")
    # This node is minimal and relies on the checkpointer for state preservation
    return {}

# --- Routers for the Modelling Flow ---
async def entry_router(state: AgentGraphState) -> str:
    """Routes based on the task and plan existence."""
    task_stage = state.get("current_context", {}).get("task_stage")
    logger.info(f"Modelling Subgraph: Routing for task '{task_stage}'.")

    if task_stage == "MODELLING_USER_REQUESTS_NEXT":
        return NODE_MODELLING_PLAN_ADVANCER

    if task_stage == "MODELLING_PAGE_TURN":
        return NODE_MODELLING_NLU

    if state.get("modelling_plan"):
        return NODE_CHECK_PLAN_COMPLETION
    else:
        return NODE_MODELLING_RAG

async def delivery_or_end_router(state: AgentGraphState) -> str:
    """Checks if the modelling plan is complete before delivering the next step."""
    plan = state.get("modelling_plan")
    current_index = state.get("current_plan_step_index", 0)

    if not plan or current_index >= len(plan):
        logger.info("Modelling plan is complete or absent. Ending modelling module.")
        return END
    else:
        logger.info(f"Modelling plan active at step {current_index + 1}. Routing to RAG for content.")
        return NODE_MODELLING_RAG

async def post_rag_router(state: AgentGraphState) -> str:
    """After RAG, routes to the planner if no plan exists, or delivery if one does."""
    if state.get("modelling_plan"):
        logger.info("Post-RAG: Plan exists. Routing to delivery generator.")
        return NODE_MODELLING_DELIVERY_GENERATOR
    else:
        logger.info("Post-RAG: No plan. Routing to planner.")
        return NODE_MODELLING_PLANNER

async def after_interaction_router(state: AgentGraphState) -> str:
    """Routes after student interaction (NLU). Decides whether to advance or re-explain."""
    intent = state.get("student_intent_for_modelling_turn", "CONFIRM_UNDERSTANDING")
    logger.info(f"Post-Interaction Router: Student intent is '{intent}'")
    
    if intent == "CONFIRM_UNDERSTANDING":
        return NODE_MODELLING_PLAN_ADVANCER
    else: # If student is confused or asks a question, re-run delivery for the same step.
        logger.info(f"Student intent is '{intent}'. Re-running RAG and delivery for the current step.")
        return NODE_MODELLING_RAG

# --- Graph Definition ---
def create_modelling_subgraph():
    workflow = StateGraph(AgentGraphState)

    # Add Nodes
    workflow.add_node(NODE_MODELLING_PLANNER, modelling_planner_node)
    workflow.add_node(NODE_MODELLING_RAG, modelling_RAG_document_node)
    workflow.add_node(NODE_MODELLING_DELIVERY_GENERATOR, modelling_delivery_generator_node)
    workflow.add_node(NODE_MODELLING_OUTPUT_FORMATTER, modelling_output_formatter_node)
    workflow.add_node(NODE_MODELLING_NLU, modelling_nlu_node)
    workflow.add_node(NODE_MODELLING_PLAN_ADVANCER, modelling_plan_advancer_node)
    workflow.add_node(NODE_CHECK_PLAN_COMPLETION, check_plan_completion_node)

    # 1. Entry Point: Plan, NLU, or Deliver?
    workflow.set_conditional_entry_point(
        entry_router,
        {
            NODE_MODELLING_RAG: NODE_MODELLING_RAG,
            NODE_CHECK_PLAN_COMPLETION: NODE_CHECK_PLAN_COMPLETION,
            NODE_MODELLING_NLU: NODE_MODELLING_NLU
        }
    )

    # 2. Post-RAG Routing
    workflow.add_conditional_edges(
        NODE_MODELLING_RAG,
        post_rag_router,
        {
            NODE_MODELLING_PLANNER: NODE_MODELLING_PLANNER,
            NODE_MODELLING_DELIVERY_GENERATOR: NODE_MODELLING_DELIVERY_GENERATOR
        }
    )

    # 3. Planning Flow
    workflow.add_edge(NODE_MODELLING_PLANNER, NODE_CHECK_PLAN_COMPLETION)

    # 4. Delivery Loop Entry
    workflow.add_conditional_edges(
        NODE_CHECK_PLAN_COMPLETION, 
        delivery_or_end_router,
        {
            NODE_MODELLING_RAG: NODE_MODELLING_RAG, # If not done, get content for next step
            END: END # If done, exit
        }
    )

    # 5. Delivery Content Generation -> Format then END
    workflow.add_edge(NODE_MODELLING_DELIVERY_GENERATOR, NODE_MODELLING_OUTPUT_FORMATTER)
    workflow.add_edge(NODE_MODELLING_OUTPUT_FORMATTER, END)

    # 6. Interaction Handling
    workflow.add_conditional_edges(
        NODE_MODELLING_NLU,
        after_interaction_router,
        {
            NODE_MODELLING_PLAN_ADVANCER: NODE_MODELLING_PLAN_ADVANCER,
            NODE_MODELLING_RAG: NODE_MODELLING_RAG # Path for re-explaining
        }
    )

    # 7. Plan Advancer loops back to check completion
    workflow.add_edge(NODE_MODELLING_PLAN_ADVANCER, NODE_CHECK_PLAN_COMPLETION)

    return workflow.compile()
NODE_MODELLING_OUTPUT_FORMATTER = "modelling_output_formatter"

async def check_plan_completion_node(state: AgentGraphState) -> dict:
    """
    A placeholder node that now correctly preserves the ENTIRE session state
    (the plan AND the session context) for the next router.
    """
    logger.info("--- Checking if modelling plan is complete (Full State-Preserving) --- ")
    return {
        "pedagogical_plan": state.get("pedagogical_plan"),
        "current_plan_step_index": state.get("current_plan_step_index"),
        "lesson_id": state.get("lesson_id"),
        "Learning_Objective_Focus": state.get("Learning_Objective_Focus"),
        "STUDENT_PROFICIENCY": state.get("STUDENT_PROFICIENCY"),
        "STUDENT_AFFECTIVE_STATE": state.get("STUDENT_AFFECTIVE_STATE"),
    }

async def entry_router(state: AgentGraphState) -> str:
    """Routes based on the task and plan existence."""
    task_stage = state.get("current_context", {}).get("task_stage")

    logger.info(f"Modelling Subgraph: Routing for task '{task_stage}'.")

    if task_stage == "MODELLING_USER_REQUESTS_NEXT":
        logger.info("Modelling Subgraph: User clicked 'Next'. Routing to plan advancer.")
        return NODE_MODELLING_PLAN_ADVANCER

    if task_stage == "MODELLING_PAGE_TURN":
        logger.info("Modelling Subgraph: Conversational turn. Routing to NLU handler.")
        return NODE_MODELLING_NLU

    if state.get("pedagogical_plan"):
        logger.info("Modelling Subgraph: Active plan found. Routing to check completion.")
        return NODE_CHECK_PLAN_COMPLETION
    else:
        logger.info("Modelling Subgraph: No plan found. Routing to RAG for planning.")
        return NODE_MODELLING_RAG

async def delivery_or_end_router(state: AgentGraphState) -> str:
    """Checks if the lesson plan is complete before delivering the next step."""
    plan = state.get("pedagogical_plan")
    current_index = state.get("current_plan_step_index", 0)

    if not plan or current_index >= len(plan):
        logger.info("Plan is complete or absent. Ending modelling module.")
        return END
    else:
        logger.info(f"Plan active at step {current_index + 1}. Routing to RAG for content.")
        return NODE_MODELLING_RAG

async def post_rag_router(state: AgentGraphState) -> str:
    """After RAG, routes to the planner if no plan exists, or delivery if one does."""
    if state.get("pedagogical_plan"):
        logger.info("Post-RAG: Plan exists. Routing to delivery generator.")
        return NODE_MODELLING_DELIVERY_GENERATOR
    else:
        logger.info("Post-RAG: No plan. Routing to planner.")
        return NODE_MODELLING_PLANNER

async def after_interaction_router(state: AgentGraphState) -> str:
    """Routes after student interaction (NLU). Decides whether to advance or re-explain."""
    intent = state.get("student_intent_for_lesson_turn", "CONFIRM_UNDERSTANDING")
    logger.info(f"Post-Interaction Router: Student intent is '{intent}'")
    
    if intent == "CONFIRM_UNDERSTANDING":
        return NODE_MODELLING_PLAN_ADVANCER
    else:
        logger.info(f"Student intent is '{intent}'. Re-running RAG and delivery for the current step.")
        return NODE_MODELLING_RAG

def create_modelling_subgraph():
    workflow = StateGraph(AgentGraphState)

    workflow.add_node(NODE_MODELLING_PLANNER, modelling_planner_node)
    workflow.add_node(NODE_MODELLING_RAG, modelling_RAG_document_node)
    workflow.add_node(NODE_MODELLING_DELIVERY_GENERATOR, modelling_delivery_generator_node)
    workflow.add_node(NODE_MODELLING_OUTPUT_FORMATTER, modelling_output_formatter_node)
    workflow.add_node(NODE_MODELLING_NLU, modelling_nlu_node)
    workflow.add_node(NODE_MODELLING_PLAN_ADVANCER, modelling_plan_advancer_node)
    workflow.add_node(NODE_CHECK_PLAN_COMPLETION, check_plan_completion_node)

    workflow.set_conditional_entry_point(
        entry_router,
        {
            NODE_MODELLING_RAG: NODE_MODELLING_RAG,
            NODE_CHECK_PLAN_COMPLETION: NODE_CHECK_PLAN_COMPLETION,
            NODE_MODELLING_NLU: NODE_MODELLING_NLU
        }
    )

    workflow.add_conditional_edges(
        NODE_MODELLING_RAG,
        post_rag_router,
        {
            NODE_MODELLING_PLANNER: NODE_MODELLING_PLANNER,
            NODE_MODELLING_DELIVERY_GENERATOR: NODE_MODELLING_DELIVERY_GENERATOR
        }
    )

    workflow.add_edge(NODE_MODELLING_PLANNER, NODE_CHECK_PLAN_COMPLETION)

    workflow.add_conditional_edges(
        NODE_CHECK_PLAN_COMPLETION, 
        delivery_or_end_router,
        {
            NODE_MODELLING_RAG: NODE_MODELLING_RAG,
            END: END
        }
    )

    workflow.add_edge(NODE_MODELLING_DELIVERY_GENERATOR, NODE_MODELLING_OUTPUT_FORMATTER)
    workflow.add_edge(NODE_MODELLING_OUTPUT_FORMATTER, END)

    workflow.add_conditional_edges(
        NODE_MODELLING_NLU,
        after_interaction_router,
        {
            NODE_MODELLING_PLAN_ADVANCER: NODE_MODELLING_PLAN_ADVANCER,
            NODE_MODELLING_RAG: NODE_MODELLING_RAG
        }
    )

    workflow.add_edge(NODE_MODELLING_PLAN_ADVANCER, NODE_CHECK_PLAN_COMPLETION)

    return workflow.compile()
