# langgraph-service/graph_builder.py

import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# --- 1. Import all your subgraph creation functions ---
from graph.modeling_flow import create_modeling_subgraph
from graph.teaching_flow import create_teaching_subgraph
from graph.scaffolding_flow import create_scaffolding_subgraph
from graph.feedback_flow import create_feedback_subgraph
from graph.cowriting_flow import create_cowriting_subgraph
from graph.conversation_flow import create_conversation_subgraph
from graph.welcome_flow import create_welcome_subgraph
from graph.pedagogy_flow import create_pedagogy_subgraph


# --- Node Imports ---
from agents import (
    save_interaction_node
)

logger = logging.getLogger(__name__)

# --- Define clear node names for the main graph ---
# These are the names for your subgraphs when treated as nodes
NODE_MODELING_MODULE = "modeling_module"
NODE_TEACHING_MODULE = "teaching_module"
NODE_SCAFFOLDING_MODULE = "scaffolding_module"
NODE_FEEDBACK_MODULE = "feedback_module"
NODE_COWRITING_MODULE = "cowriting_module"
NODE_PEDAGOGY_MODULE = "pedagogy_module"

# Names for your top-level nodes
NODE_ROUTER_ENTRY = "router_entry_point" 
NODE_HANDLE_WELCOME = "handle_welcome"
NODE_CONVERSATION_HANDLER = "conversation_handler"
NODE_SAVE_INTERACTION = "save_interaction"


# This is the actual NODE. It must return a dictionary.
async def router_entry_node(state: AgentGraphState) -> dict:
    """
    An empty entry point node. Its only job is to be the source for the
    main conditional edge. It can also be used for pre-routing logging.
    """
    logger.info("--- Entering Main Graph Router ---")
    return {} 

async def initial_router_logic(state: AgentGraphState) -> str:
    """
    This is the master switchboard. It inspects the state and decides
    which specialized subgraph (or standalone node) to route to.
    """
    # Prioritize the explicit task_name from the InvokeAgentTask request
    task_name = state.get("task_name")
    
    # As a fallback, check the task_stage from the context dictionary
    if not task_name:
        context = state.get("current_context", {})
        task_name = context.get("task_stage")

    # If still nothing, default to conversation
    if not task_name:
        task_name = "handle_conversation"

    logger.info(f"INITIAL ROUTER: Evaluating route. Final determined Task Name: '{task_name}'")

    # The rest of your routing map is perfect.
    if task_name == "handle_page_load":
        route_destination = NODE_HANDLE_WELCOME
    elif task_name == "start_modelling_activity":
        route_destination = NODE_MODELING_MODULE
    elif task_name == "request_teaching_lesson":
        route_destination = NODE_TEACHING_MODULE
    elif task_name == "scaffolding_needed":
        route_destination = NODE_SCAFFOLDING_MODULE
    elif task_name == "feedback_needed":
        route_destination = NODE_FEEDBACK_MODULE
    elif task_name == "initiate_cowriting":
        route_destination = NODE_COWRITING_MODULE
    elif task_name == "initiate_pedagogy":
        route_destination = NODE_PEDAGOGY_MODULE
    else:
        route_destination = NODE_CONVERSATION_HANDLER
        
    logger.info(f"INITIAL ROUTER: Final decision. Routing to -> [{route_destination}]")
    return route_destination

def build_graph():
    """
    Builds the main LangGraph application by wiring together all the specialized subgraphs
    and top-level control nodes.
    """
    logger.info("--- Building Main TOEFL Tutor Graph ---")

    workflow = StateGraph(AgentGraphState)

    # --- 3. Instantiate and add all subgraphs 
    modeling_subgraph = create_modeling_subgraph()
    teaching_subgraph = create_teaching_subgraph()
    scaffolding_subgraph = create_scaffolding_subgraph()
    feedback_subgraph = create_feedback_subgraph()
    cowriting_subgraph = create_cowriting_subgraph()
    conversation_subgraph = create_conversation_subgraph()
    welcome_subgraph = create_welcome_subgraph()
    pedagogy_subgraph = create_pedagogy_subgraph()

    workflow.add_node(NODE_MODELING_MODULE, modeling_subgraph)
    workflow.add_node(NODE_TEACHING_MODULE, teaching_subgraph)
    workflow.add_node(NODE_SCAFFOLDING_MODULE, scaffolding_subgraph)
    workflow.add_node(NODE_FEEDBACK_MODULE, feedback_subgraph)
    workflow.add_node(NODE_COWRITING_MODULE, cowriting_subgraph)
    workflow.add_node(NODE_CONVERSATION_HANDLER, conversation_subgraph)
    workflow.add_node(NODE_HANDLE_WELCOME, welcome_subgraph)
    workflow.add_node(NODE_PEDAGOGY_MODULE, pedagogy_subgraph)

    workflow.add_node(NODE_ROUTER_ENTRY, router_entry_node)
    workflow.add_node(NODE_SAVE_INTERACTION, save_interaction_node)

    # --- 5. Define the Graph's Flow ---
    workflow.set_entry_point(NODE_ROUTER_ENTRY)

    # Add the conditional edges FROM the entry node, using your logic as the PATH function
    workflow.add_conditional_edges(
        NODE_ROUTER_ENTRY,
        initial_router_logic,
        {
            # Map the router's output string to the actual node name
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

    # --- 6. THE FINAL, SIMPLIFIED WIRING ---
    # Every single flow now connects DIRECTLY to the save node.
    # The formatting has already been done inside each flow.
    workflow.add_edge(NODE_MODELING_MODULE, NODE_SAVE_INTERACTION)
    workflow.add_edge(NODE_TEACHING_MODULE, NODE_SAVE_INTERACTION)
    workflow.add_edge(NODE_SCAFFOLDING_MODULE, NODE_SAVE_INTERACTION)
    workflow.add_edge(NODE_FEEDBACK_MODULE, NODE_SAVE_INTERACTION)
    workflow.add_edge(NODE_COWRITING_MODULE, NODE_SAVE_INTERACTION)
    workflow.add_edge(NODE_PEDAGOGY_MODULE, NODE_SAVE_INTERACTION)
    workflow.add_edge(NODE_CONVERSATION_HANDLER, NODE_SAVE_INTERACTION)
    workflow.add_edge(NODE_HANDLE_WELCOME, NODE_SAVE_INTERACTION)

    # After saving, the graph ends.
    workflow.add_edge(NODE_SAVE_INTERACTION, END)

    # --- 7. Compile the Final Graph ---
    logger.info("--- Main Graph Compilation ---")
    # compiled_graph = workflow.compile(checkpointer=checkpointer)
    compiled_graph = workflow.compile() # Compile without checkpointer for now
    logger.info("--- Main Graph Compiled Successfully ---")
    
    return compiled_graph
