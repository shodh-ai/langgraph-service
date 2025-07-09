# graph/teaching_flow.py
import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState

logger = logging.getLogger(__name__)

# Import node functions
from agents.teaching_planner_node import teaching_planner_node
from agents.teaching_delivery_node import teaching_delivery_generator_node
from agents.teaching_plan_advancer_node import teaching_plan_advancer_node
# REMOVE the old QA handler import, we will replace it with the NLU node
# from agents.teaching_qa_handler_node import teaching_qa_handler_node
from agents.teaching_nlu_node import teaching_nlu_node # <<< IMPORT NEW NODE
from agents.teaching_RAG_document_node import teaching_RAG_document_node
from agents.teaching_output_formatter import teaching_output_formatter_node


# --- Node Names ---
NODE_TEACHING_RAG = "teaching_rag_document"
NODE_TEACHING_PLANNER = "teaching_planner"
NODE_TEACHING_DELIVERY_GENERATOR = "teaching_delivery_generator"
NODE_TEACHING_NLU = "teaching_nlu" # <<< RENAME a node
NODE_TEACHING_PLAN_ADVANCER = "teaching_plan_advancer"
NODE_CHECK_PLAN_COMPLETION = "check_plan_completion"
NODE_TEACHING_OUTPUT_FORMATTER = "teaching_output_formatter"

async def check_plan_completion_node(state: AgentGraphState) -> dict:
    """
    A placeholder node that now correctly preserves the ENTIRE session state
    (the plan AND the session context) for the next router.
    """
    logger.info("--- Checking if lesson plan is complete (Full State-Preserving) --- ")
    return {
        # The plan itself
        "pedagogical_plan": state.get("pedagogical_plan"),
        "current_plan_step_index": state.get("current_plan_step_index"),
        
        # The critical session context that must survive the whole lesson
        "lesson_id": state.get("lesson_id"),
        "Learning_Objective_Focus": state.get("Learning_Objective_Focus"),
        "STUDENT_PROFICIENCY": state.get("STUDENT_PROFICIENCY"),
        "STUDENT_AFFECTIVE_STATE": state.get("STUDENT_AFFECTIVE_STATE"),
    }

# --- MODIFIED ROUTERS ---
async def entry_router(state: AgentGraphState) -> str:
    """Routes based on the task and plan existence."""
    task_stage = state.get("current_context", {}).get("task_stage")

    logger.info(f"Teaching Subgraph: Routing for task '{task_stage}'.")

    if task_stage == "TEACHING_USER_REQUESTS_NEXT":
        logger.info("Teaching Subgraph: User clicked 'Next'. Routing to plan advancer.")
        # Go directly to the node that increments the step index
        return NODE_TEACHING_PLAN_ADVANCER

    # If the task is a follow-up turn from the user, go to our new NLU handler
    if task_stage == "TEACHING_PAGE_TURN": # <<< USE A NEW TASK_STAGE
        logger.info("Teaching Subgraph: Conversational turn. Routing to NLU handler.")
        return NODE_TEACHING_NLU

    # For initial tasks, check if a plan already exists
    if state.get("pedagogical_plan"):
        logger.info("Teaching Subgraph: Active plan found. Routing to check completion.")
        return NODE_CHECK_PLAN_COMPLETION
    else:
        logger.info("Teaching Subgraph: No plan found. Routing to RAG for planning.")
        return NODE_TEACHING_RAG

async def delivery_or_end_router(state: AgentGraphState) -> str:
    """Checks if the lesson plan is complete before delivering the next step."""
    plan = state.get("pedagogical_plan")
    current_index = state.get("current_plan_step_index", 0)

    if not plan or current_index >= len(plan):
        logger.info("Plan is complete or absent. Ending teaching module.")
        return END
    else:
        logger.info(f"Plan active at step {current_index + 1}. Routing to RAG for content.")
        return NODE_TEACHING_RAG

async def post_rag_router(state: AgentGraphState) -> str:
    """After RAG, routes to the planner if no plan exists, or delivery if one does."""
    if state.get("pedagogical_plan"):
        logger.info("Post-RAG: Plan exists. Routing to delivery generator.")
        return NODE_TEACHING_DELIVERY_GENERATOR
    else:
        logger.info("Post-RAG: No plan. Routing to planner.")
        return NODE_TEACHING_PLANNER

async def after_interaction_router(state: AgentGraphState) -> str:
    """Routes after student interaction (NLU). Decides whether to advance or re-explain."""
    # This router now works perfectly because teaching_nlu_node populates the correct key.
    intent = state.get("student_intent_for_lesson_turn", "CONFIRM_UNDERSTANDING")
    logger.info(f"Post-Interaction Router: Student intent is '{intent}'")
    
    if intent == "CONFIRM_UNDERSTANDING":
        return NODE_TEACHING_PLAN_ADVANCER
    else: # If student is confused or asks a question, re-run delivery for the same step.
        logger.info(f"Student intent is '{intent}'. Re-running RAG and delivery for the current step.")
        return NODE_TEACHING_RAG

# --- MODIFIED GRAPH DEFINITION ---
def create_teaching_subgraph():
    workflow = StateGraph(AgentGraphState)

    # Add Nodes
    workflow.add_node(NODE_TEACHING_PLANNER, teaching_planner_node)
    workflow.add_node(NODE_TEACHING_RAG, teaching_RAG_document_node)
    workflow.add_node(NODE_TEACHING_DELIVERY_GENERATOR, teaching_delivery_generator_node)
    workflow.add_node(NODE_TEACHING_OUTPUT_FORMATTER, teaching_output_formatter_node)
    workflow.add_node(NODE_TEACHING_NLU, teaching_nlu_node) # <<< ADD NEW NODE
    workflow.add_node(NODE_TEACHING_PLAN_ADVANCER, teaching_plan_advancer_node)
    workflow.add_node(NODE_CHECK_PLAN_COMPLETION, check_plan_completion_node)

    # 1. Entry Point: Plan, NLU, or Deliver?
    workflow.set_conditional_entry_point(
        entry_router,
        {
            NODE_TEACHING_RAG: NODE_TEACHING_RAG,
            NODE_CHECK_PLAN_COMPLETION: NODE_CHECK_PLAN_COMPLETION,
            NODE_TEACHING_NLU: NODE_TEACHING_NLU # <<< ADD NEW ENTRY PATH
        }
    )

    # 2. Post-RAG Routing (unchanged)
    workflow.add_conditional_edges(
        NODE_TEACHING_RAG,
        post_rag_router,
        {
            NODE_TEACHING_PLANNER: NODE_TEACHING_PLANNER,
            NODE_TEACHING_DELIVERY_GENERATOR: NODE_TEACHING_DELIVERY_GENERATOR
        }
    )

    # 3. Planning Flow (unchanged)
    workflow.add_edge(NODE_TEACHING_PLANNER, NODE_CHECK_PLAN_COMPLETION)

    # 4. Delivery Loop Entry (unchanged)
    workflow.add_conditional_edges(
        NODE_CHECK_PLAN_COMPLETION, 
        delivery_or_end_router,
        {
            NODE_TEACHING_RAG: NODE_TEACHING_RAG, # If not done, get content for next step
            END: END # If done, exit
        }
    )

    # 5. Delivery Content Generation -> Format then END (unchanged)
    workflow.add_edge(NODE_TEACHING_DELIVERY_GENERATOR, NODE_TEACHING_OUTPUT_FORMATTER)
    workflow.add_edge(NODE_TEACHING_OUTPUT_FORMATTER, END)

    # 6. Interaction Handling (The New Flow)
    workflow.add_conditional_edges(
        NODE_TEACHING_NLU, # After our new NLU node...
        after_interaction_router, # ...use the interaction router to decide what's next
        {
            NODE_TEACHING_PLAN_ADVANCER: NODE_TEACHING_PLAN_ADVANCER,
            NODE_TEACHING_RAG: NODE_TEACHING_RAG # Path for re-explaining
        }
    )

    # 7. Plan Advancer loops back to check completion
    workflow.add_edge(NODE_TEACHING_PLAN_ADVANCER, NODE_CHECK_PLAN_COMPLETION)

    return workflow.compile()