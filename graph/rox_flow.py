# graph/rox_flow.py

import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# Import all the nodes this flow will use
from agents.rox_nodes import (
    rox_welcome_and_reflection_node,
    rox_propose_plan_node,
    rox_nlu_and_qa_handler_node, 
    rox_navigate_to_task_node
)
from agents.curriculum_navigator_node import curriculum_navigator_node
from agents.student_model_node import load_student_data_node
from agents.pedagogical_strategy_planner_node import pedagogical_strategy_planner_node 

logger = logging.getLogger(__name__)

# --- Node Names for Clarity ---
NODE_LOAD_STUDENT_DATA = "rox_load_student_data"
NODE_WELCOME_AND_REFLECT = "rox_welcome"
NODE_CURRICULUM_NAVIGATOR = "rox_curriculum_navigator"
NODE_PROPOSE_PLAN = "rox_propose_lo"
NODE_NLU_QA_HANDLER = "rox_nlu_handler"
NODE_PEDAGOGY_PLANNER = "rox_pedagogy_planner" 
NODE_NAVIGATE_TO_TASK = "rox_navigate_to_task"
# NODE_SHOW_COURSE_MAP has been removed
NODE_ROX_FALLBACK = "rox_fallback"

# --- Router Functions for Rox Flow ---

async def rox_entry_router(state: AgentGraphState) -> str:
    """
    If a transcript exists, it's a conversational turn that needs NLU.
    If not, it's a page load that needs to start the welcome sequence.
    """
    if state.get("transcript"):
        logger.info("Rox Flow: Transcript found. Routing to NLU handler.")
        return NODE_NLU_QA_HANDLER
    else:
        logger.info("Rox Flow: No transcript. Routing to load student data for welcome.")
        return NODE_LOAD_STUDENT_DATA

async def after_nlu_router(state: AgentGraphState) -> str:
    """Decides what to do after the NLU handler classifies the user's intent."""
    intent = state.get("student_intent_for_rox_turn")
    logger.info(f"Rox NLU Router: Received intent '{intent}'")
    
    if intent == "CONFIRM_START_TASK":
        # The user agreed! Time to build the real plan and navigate.
        return NODE_PEDAGOGY_PLANNER
    
    # For all other cases (rejection, asking for map, general question),
    # the AI needs to consult the curriculum to make a new proposal.
    elif intent in ["REJECT_OR_QUESTION_LO", "REQUEST_COURSE_MAP", "GENERAL_QUESTION"]:
        return NODE_CURRICULUM_NAVIGATOR
        
    else:
        return END

# --- Graph Definition ---
def create_rox_welcome_subgraph():
    """Creates the complete, robust conversational subgraph for the Rox Dashboard."""
    workflow = StateGraph(AgentGraphState)

    # 1. Add all the nodes
    workflow.add_node(NODE_LOAD_STUDENT_DATA, load_student_data_node)
    workflow.add_node(NODE_WELCOME_AND_REFLECT, rox_welcome_and_reflection_node)
    workflow.add_node(NODE_NLU_QA_HANDLER, rox_nlu_and_qa_handler_node)
    workflow.add_node(NODE_CURRICULUM_NAVIGATOR, curriculum_navigator_node)
    workflow.add_node(NODE_PROPOSE_PLAN, rox_propose_plan_node) # Presenter node
    workflow.add_node(NODE_PEDAGOGY_PLANNER, pedagogical_strategy_planner_node)
    workflow.add_node(NODE_NAVIGATE_TO_TASK, rox_navigate_to_task_node)

    # 2. Define the entry point
    workflow.set_conditional_entry_point(
        rox_entry_router,
        {
            NODE_LOAD_STUDENT_DATA: NODE_LOAD_STUDENT_DATA,
            NODE_NLU_QA_HANDLER: NODE_NLU_QA_HANDLER
        }
    )

    # 3. Define the path for the first turn (Welcome)
    workflow.add_edge(NODE_LOAD_STUDENT_DATA, NODE_WELCOME_AND_REFLECT)
    workflow.add_edge(NODE_WELCOME_AND_REFLECT, END)

    # 4. Define the two main paths after the user speaks
    workflow.add_conditional_edges(
        NODE_NLU_QA_HANDLER,
        after_nlu_router,
        {
            # Path 1: The user AGREED. Go plan the lesson.
            NODE_PEDAGOGY_PLANNER: NODE_PEDAGOGY_PLANNER,
            # Path 2: The user wants a proposal (or map). Go get the data.
            NODE_CURRICULUM_NAVIGATOR: NODE_CURRICULUM_NAVIGATOR,
            END: END
        }
    )

    # 5. Define the full, unbreakable chains for each path

    # For the "Proposal Path", after the navigator gets data, it MUST go to the proposer.
    workflow.add_edge(NODE_CURRICULUM_NAVIGATOR, NODE_PROPOSE_PLAN)
    workflow.add_edge(NODE_PROPOSE_PLAN, END) # The proposer ends the turn.

    # For the "Happy Path", after the planner builds the plan, it MUST go to the navigator node.
    workflow.add_edge(NODE_PEDAGOGY_PLANNER, NODE_NAVIGATE_TO_TASK)
    workflow.add_edge(NODE_NAVIGATE_TO_TASK, END) # The navigator node ends the turn.

    return workflow.compile()