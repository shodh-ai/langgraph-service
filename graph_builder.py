# FINAL, PERFECTED graph/graph_builder.py

import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# --- 1. Import all your flows AND simple nodes ---
from graph.modeling_flow import create_modeling_subgraph
from graph.teaching_flow import create_teaching_subgraph
from graph.scaffolding_flow import create_scaffolding_subgraph
from graph.feedback_flow import create_feedback_subgraph
from graph.cowriting_flow import create_cowriting_subgraph
from graph.welcome_flow import create_welcome_subgraph
from graph.pedagogy_flow import create_pedagogy_subgraph

from agents import save_interaction_node
from agents.acknowledge_interrupt_node import acknowledge_interrupt_node
from agents.conversation_handler import conversation_handler_node

logger = logging.getLogger(__name__)

# --- 2. Define ALL node names ---
NODE_MODELING_MODULE = "modeling_module"
NODE_TEACHING_MODULE = "teaching_module"
NODE_SCAFFOLDING_MODULE = "scaffolding_module"
NODE_FEEDBACK_MODULE = "feedback_module"
NODE_COWRITING_MODULE = "cowriting_module"
NODE_PEDAGOGY_MODULE = "pedagogy_module"
NODE_HANDLE_WELCOME = "handle_welcome"
NODE_ACKNOWLEDGE_INTERRUPT = "acknowledge_interrupt"
NODE_ROUTER_ENTRY = "router_entry_point"
NODE_CONVERSATION_HANDLER = "conversation_handler"
NODE_SAVE_INTERACTION = "save_interaction"

# This node is a placeholder, it can be an empty function
async def router_entry_node(state: AgentGraphState) -> dict:
    logger.info("--- Entering Main Graph Router ---")
    return {}

async def initial_router_logic(state: AgentGraphState) -> str:
    task_name = state.get("task_name")
    if not task_name:
        context = state.get("current_context", {})
        task_name = context.get("task_stage")
    if not task_name:
        task_name = "handle_conversation"
    
    logger.info(f"INITIAL ROUTER: Evaluating route. Final determined Task Name: '{task_name}'")

    routing_map = {
        "handle_student_response": NODE_CONVERSATION_HANDLER,
        "user_wants_to_interrupt": NODE_ACKNOWLEDGE_INTERRUPT,
        "acknowledge_interruption": NODE_ACKNOWLEDGE_INTERRUPT,
        "handle_page_load": NODE_HANDLE_WELCOME,
        "start_modelling_activity": NODE_MODELING_MODULE,
        "request_teaching_lesson": NODE_TEACHING_MODULE,
        "scaffolding_needed": NODE_SCAFFOLDING_MODULE,
        "feedback_needed": NODE_FEEDBACK_MODULE,
        "initiate_cowriting": NODE_COWRITING_MODULE,
        "initiate_pedagogy": NODE_PEDAGOGY_MODULE,
        "handle_student_clarification_question": NODE_CONVERSATION_HANDLER,
    }
    route_destination = routing_map.get(task_name, NODE_CONVERSATION_HANDLER)
        
    logger.info(f"INITIAL ROUTER: Final decision. Routing to -> [{route_destination}]")
    return route_destination

def build_graph():
    logger.info("--- Building Main TOEFL Tutor Graph ---")
    workflow = StateGraph(AgentGraphState)

    # --- 3. Add ALL Nodes and Subgraphs ---
    workflow.add_node(NODE_MODELING_MODULE, create_modeling_subgraph())
    workflow.add_node(NODE_TEACHING_MODULE, create_teaching_subgraph())
    workflow.add_node(NODE_SCAFFOLDING_MODULE, create_scaffolding_subgraph())
    workflow.add_node(NODE_FEEDBACK_MODULE, create_feedback_subgraph())
    workflow.add_node(NODE_COWRITING_MODULE, create_cowriting_subgraph())
    workflow.add_node(NODE_HANDLE_WELCOME, create_welcome_subgraph())
    workflow.add_node(NODE_PEDAGOGY_MODULE, create_pedagogy_subgraph())

    # Add the simple nodes
    workflow.add_node(NODE_ROUTER_ENTRY, router_entry_node)
    workflow.add_node(NODE_CONVERSATION_HANDLER, conversation_handler_node)
    workflow.add_node(NODE_SAVE_INTERACTION, save_interaction_node)
    workflow.add_node(NODE_ACKNOWLEDGE_INTERRUPT, acknowledge_interrupt_node)

    # --- 4. Define the Graph's Flow ---
    workflow.set_entry_point(NODE_ROUTER_ENTRY)

    workflow.add_conditional_edges(
        NODE_ROUTER_ENTRY,
        initial_router_logic,
        {
            NODE_ACKNOWLEDGE_INTERRUPT: NODE_ACKNOWLEDGE_INTERRUPT,
            NODE_MODELING_MODULE: NODE_MODELING_MODULE,
            NODE_TEACHING_MODULE: NODE_TEACHING_MODULE,
            NODE_SCAFFOLDING_MODULE: NODE_SCAFFOLDING_MODULE,
            NODE_FEEDBACK_MODULE: NODE_FEEDBACK_MODULE,
            NODE_COWRITING_MODULE: NODE_COWRITING_MODULE,
            NODE_PEDAGOGY_MODULE: NODE_PEDAGOGY_MODULE,
            NODE_HANDLE_WELCOME: NODE_HANDLE_WELCOME,
            NODE_CONVERSATION_HANDLER: NODE_CONVERSATION_HANDLER,
        }
    )

    # --- 5. Define the Exit Paths ---
    all_flow_nodes = [
        NODE_MODELING_MODULE, NODE_TEACHING_MODULE, NODE_SCAFFOLDING_MODULE,
        NODE_FEEDBACK_MODULE, NODE_COWRITING_MODULE, NODE_PEDAGOGY_MODULE,
        NODE_HANDLE_WELCOME, NODE_CONVERSATION_HANDLER, NODE_ACKNOWLEDGE_INTERRUPT
    ]
    for node_name in all_flow_nodes:
        workflow.add_edge(node_name, NODE_SAVE_INTERACTION)

    workflow.add_edge(NODE_SAVE_INTERACTION, END)

    # --- Compile the Final Graph ---
    logger.info("--- Main Graph Compilation ---")
    compiled_graph = workflow.compile()
    logger.info("--- Main Graph Compiled Successfully ---")
    
    return compiled_graph
