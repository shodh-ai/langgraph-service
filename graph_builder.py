import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# Import the new async agent node functions
from agents import (
    generate_test_button_feedback_stub_node, # New import
    load_student_context_node,
    save_interaction_summary_node,
    diagnose_speaking_stub_node,
    generate_feedback_stub_node
)

logger = logging.getLogger(__name__)

# Define node names for clarity
NODE_LOAD_STUDENT_CONTEXT = "load_student_context"
NODE_DIAGNOSE_SPEAKING = "diagnose_speaking"
NODE_GENERATE_FEEDBACK = "generate_feedback"
NODE_SAVE_INTERACTION = "save_interaction"
NODE_FEEDBACK_FOR_TEST_BUTTON = "feedback_for_test_button_node" # New node constant

def build_graph():
    """Builds and compiles the LangGraph application with the new structure."""
    logger.info("Building LangGraph with AgentGraphState...")
    workflow = StateGraph(AgentGraphState)

    # Add nodes to the graph
    # These are async functions, LangGraph handles their invocation.
    workflow.add_node(NODE_LOAD_STUDENT_CONTEXT, load_student_context_node)
    workflow.add_node(NODE_DIAGNOSE_SPEAKING, diagnose_speaking_stub_node)
    workflow.add_node(NODE_GENERATE_FEEDBACK, generate_feedback_stub_node)
    workflow.add_node(NODE_SAVE_INTERACTION, save_interaction_summary_node)
    workflow.add_node(NODE_FEEDBACK_FOR_TEST_BUTTON, generate_test_button_feedback_stub_node) # Add new node

    # Set the entry point for the graph
    workflow.set_entry_point(NODE_LOAD_STUDENT_CONTEXT)

    # Define edges
    # workflow.add_edge(NODE_LOAD_STUDENT_CONTEXT, NODE_DIAGNOSE_SPEAKING) # Original edge
    # Instead of direct edge, let's route after loading context, or after diagnosis if preferred.
    # For this test, routing after NODE_DIAGNOSE_SPEAKING as per user plan.
    workflow.add_edge(NODE_LOAD_STUDENT_CONTEXT, NODE_DIAGNOSE_SPEAKING)

    # Define the router function based on task_stage
    async def route_based_on_task_stage(state: AgentGraphState) -> str:
        context = state.get("current_context")
        if not context:
            logger.warning("LangGraph Router: current_context is missing in state. Routing to default feedback.")
            return NODE_GENERATE_FEEDBACK

        task_stage = getattr(context, 'task_stage', None)
        user_id = state.get('user_id', 'unknown_user')

        logger.info(f"LangGraph Router: User '{user_id}', Current task_stage is '{task_stage}'")
        if task_stage == "testing_specific_context_from_button":
            logger.info(f"LangGraph Router: User '{user_id}', Routing to feedback_for_test_button_node.")
            return NODE_FEEDBACK_FOR_TEST_BUTTON
        else:
            logger.info(f"LangGraph Router: User '{user_id}', Routing to default_feedback_node ({NODE_GENERATE_FEEDBACK}).")
            return NODE_GENERATE_FEEDBACK

    # Add conditional edge from NODE_DIAGNOSE_SPEAKING (or NODE_LOAD_STUDENT_CONTEXT)
    workflow.add_conditional_edges(
        NODE_DIAGNOSE_SPEAKING, 
        route_based_on_task_stage,
        {
            NODE_FEEDBACK_FOR_TEST_BUTTON: NODE_FEEDBACK_FOR_TEST_BUTTON,
            NODE_GENERATE_FEEDBACK: NODE_GENERATE_FEEDBACK
        }
    )

    # Edges from feedback nodes to save_interaction
    workflow.add_edge(NODE_GENERATE_FEEDBACK, NODE_SAVE_INTERACTION)
    workflow.add_edge(NODE_FEEDBACK_FOR_TEST_BUTTON, NODE_SAVE_INTERACTION) # Ensure this path also leads to save
    workflow.add_edge(NODE_SAVE_INTERACTION, END) # End of the graph after saving

    # Compile the graph
    app_graph = workflow.compile()
    logger.info("LangGraph (AgentGraphState) built and compiled successfully.")
    return app_graph

# To test the graph building process (optional, can be run directly)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        graph = build_graph()
        logger.info("Graph compiled successfully in __main__.")
        # Example: Visualizing the graph (requires graphviz and python-pygraphviz or similar)
        # try:
        #     from PIL import Image
        #     img_bytes = graph.get_graph().draw_mermaid_png()
        #     with open("new_graph_visualization.png", "wb") as f:
        #         f.write(img_bytes)
        #     logger.info("Graph visualization saved to new_graph_visualization.png")
        # except Exception as e_viz:
        #     logger.warning(f"Could not generate graph visualization: {e_viz}")
    except Exception as e_build:
        logger.error(f"Error building graph in __main__: {e_build}", exc_info=True)
