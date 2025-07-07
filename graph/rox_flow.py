# graph/rox_flow.py

import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# Import node functions that will be part of the Rox page experience
from agents.rox_nodes import (
    rox_welcome_and_reflection_node,
    rox_propose_plan_node,
    rox_nlu_and_qa_handler_node, 
    rox_navigate_to_task_node,
    show_course_map_node # <<< IMPORT NEW NODE
)
from agents.curriculum_navigator_node import curriculum_navigator_node
from agents.student_model_node import load_student_data_node
from agents.pedagogical_strategy_planner_node import pedagogical_strategy_planner_node 
from agents.plan_advancer_node import plan_advancer_node 

logger = logging.getLogger(__name__)

# --- Node Names ---
NODE_LOAD_STUDENT_DATA = "rox_load_student_data"
NODE_WELCOME_AND_REFLECT = "rox_welcome_and_reflect"
NODE_CURRICULUM_NAVIGATOR = "rox_curriculum_navigator"
NODE_PROPOSE_PLAN = "rox_propose_plan"
NODE_NLU_QA_HANDLER = "rox_nlu_qa_handler" 
NODE_PEDAGOGY_PLANNER = "pedagogy_planner" 
NODE_PLAN_ADVANCER = "plan_advancer" 
NODE_NAVIGATE_TO_TASK = "rox_navigate_to_task"
NODE_ROX_FALLBACK = "rox_fallback"
NODE_SHOW_COURSE_MAP = "rox_show_course_map" # <<< DEFINE NEW NODE NAME

# --- Router Functions for Rox Flow ---

async def rox_entry_router(state: AgentGraphState) -> str:
    """Routes the initial entry into the Rox flow."""
    task_stage = state.get("current_context", {}).get("task_stage")
    logger.info(f"Rox Flow Entry Router: task_stage is '{task_stage}'")
    
    # If it's a fresh welcome, a return after task, or the stage isn't set, start the welcome flow.
    if task_stage in ["ROX_WELCOME_INIT", "ROX_RETURN_AFTER_TASK"] or task_stage is None:
        return NODE_LOAD_STUDENT_DATA
    elif task_stage == "ROX_CONVERSATION_TURN":
        # If it's an ongoing conversation, go directly to the Q&A/NLU handler
        return NODE_NLU_QA_HANDLER
    else:
        logger.warning(f"Rox Flow: Unknown but specific entry task_stage '{task_stage}'. Defaulting to fallback.")
        return NODE_ROX_FALLBACK

async def after_welcome_router(state: AgentGraphState) -> str:
    """Decides what to do after the initial welcome/reflection message."""
    if state.get("is_simple_greeting"):
        logger.info("Rox Flow: Simple greeting complete. Ending turn.")
        return END
    else:
        # If not a simple greeting, proceed with curriculum planning.
        logger.info("Rox Flow: Welcome/Reflection complete. Routing to Curriculum Navigator.")
        return NODE_CURRICULUM_NAVIGATOR

async def after_proposal_router(state: AgentGraphState) -> str:
    """Handles student's response to the AI's proposed plan or a lesson step."""
    intent = state.get("student_intent_for_rox_turn", "ACKNOWLEDGE")
    logger.info(f"Rox Flow Post-Proposal Router: Student intent is '{intent}'")

    # --- NEW ROUTE LOGIC ---
    if intent == "REQUEST_COURSE_MAP":
        logger.info("Student requested course map. Routing to the show course map node.")
        return NODE_SHOW_COURSE_MAP
    # --- END NEW ROUTE LOGIC ---

    if intent in ["CONFIRM_START_TASK", "ACKNOWLEDGE"]:
        # Student agreed to the initial plan. Create the detailed pedagogical plan.
        logger.info("Student confirmed start. Routing to Pedagogy Planner.")
        return NODE_PEDAGOGY_PLANNER
    elif intent == "CONFIRM_UNDERSTANDING":
        # Student confirmed understanding of a lesson step. Advance the plan.
        logger.info("Student confirmed understanding. Routing to Plan Advancer.")
        return NODE_PLAN_ADVANCER
    elif intent == "REQUEST_ALTERNATIVE_TASK":
        logger.info("Student requested alternative. Routing back to Curriculum Navigator.")
        return NODE_CURRICULUM_NAVIGATOR
    else: # Student asked a question or other diversion
        logger.info("Student asked a question. Ending turn to deliver answer.")
        return END # Correctly terminate the turn to send the answer


# --- Graph Definition ---
def create_rox_welcome_subgraph():
    """Creates the LangGraph subgraph for the Rox Welcome & Reflection Page (P1)."""
    workflow = StateGraph(AgentGraphState)

    # Add Nodes
    workflow.add_node(NODE_LOAD_STUDENT_DATA, load_student_data_node)
    workflow.add_node(NODE_WELCOME_AND_REFLECT, rox_welcome_and_reflection_node)
    workflow.add_node(NODE_CURRICULUM_NAVIGATOR, curriculum_navigator_node)
    workflow.add_node(NODE_PROPOSE_PLAN, rox_propose_plan_node)
    workflow.add_node(NODE_NLU_QA_HANDLER, rox_nlu_and_qa_handler_node) 
    workflow.add_node(NODE_PEDAGOGY_PLANNER, pedagogical_strategy_planner_node) 
    workflow.add_node(NODE_PLAN_ADVANCER, plan_advancer_node) 
    workflow.add_node(NODE_NAVIGATE_TO_TASK, rox_navigate_to_task_node)
    workflow.add_node(NODE_ROX_FALLBACK, rox_nlu_and_qa_handler_node) # Fallback to the NLU handler
    workflow.add_node(NODE_SHOW_COURSE_MAP, show_course_map_node) # <<< ADD NEW NODE TO GRAPH

    # --- Define Flow ---
    workflow.set_conditional_entry_point(rox_entry_router, {
        NODE_LOAD_STUDENT_DATA: NODE_LOAD_STUDENT_DATA,
        NODE_NLU_QA_HANDLER: NODE_NLU_QA_HANDLER,
        NODE_ROX_FALLBACK: NODE_ROX_FALLBACK
    })

    # After loading the profile, the AI always welcomes or reflects.
    workflow.add_edge(NODE_LOAD_STUDENT_DATA, NODE_WELCOME_AND_REFLECT)

    # After welcoming/reflecting, we need to decide what's next.
    workflow.add_conditional_edges(
        NODE_WELCOME_AND_REFLECT,
        after_welcome_router, # This router now handles two paths
        {
            NODE_CURRICULUM_NAVIGATOR: NODE_CURRICULUM_NAVIGATOR,
            END: END # Add the missing path to terminate the graph
        }
    )
    
    # After the curriculum navigator proposes an LO, we propose it to the student.
    workflow.add_edge(NODE_CURRICULUM_NAVIGATOR, NODE_PROPOSE_PLAN)
    
    # After proposing the plan, the turn ends. AI waits for the student's response.
    workflow.add_edge(NODE_PROPOSE_PLAN, END)

    # When the student responds, the graph re-enters at the Q&A/NLU handler, which
    # then routes based on the student's intent.
    workflow.add_conditional_edges(
        NODE_NLU_QA_HANDLER, # The source node for the next decision
        after_proposal_router, # The decision logic
        {
            NODE_PEDAGOGY_PLANNER: NODE_PEDAGOGY_PLANNER,
            NODE_PLAN_ADVANCER: NODE_PLAN_ADVANCER, # Add path to the plan advancer
            NODE_CURRICULUM_NAVIGATOR: NODE_CURRICULUM_NAVIGATOR,
            NODE_SHOW_COURSE_MAP: NODE_SHOW_COURSE_MAP, # <<< ADD NEW CONDITIONAL EDGE
            END: END # Add mapping to the END node to break the recursion loop
        }
    )

    # After the detailed plan is created, we navigate to the first task.
    workflow.add_edge(NODE_PEDAGOGY_PLANNER, NODE_NAVIGATE_TO_TASK)

    # After the plan is advanced, we navigate to the next task.
    workflow.add_edge(NODE_PLAN_ADVANCER, NODE_NAVIGATE_TO_TASK)

    # The navigation node is a final step in this flow before exiting.
    workflow.add_edge(NODE_NAVIGATE_TO_TASK, END)
    
    # Fallback also ends the turn.
    workflow.add_edge(NODE_ROX_FALLBACK, END)

    # The new node is a terminal node in this flow, so it should exit the subgraph.
    workflow.add_edge(NODE_SHOW_COURSE_MAP, END) # <<< ADD EXIT PATH FOR NEW NODE

    return workflow.compile()