import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# Import the new async agent node functions
from agents import (
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

    # Set the entry point for the graph
    workflow.set_entry_point(NODE_LOAD_STUDENT_CONTEXT)

    # Define edges
    workflow.add_edge(NODE_LOAD_STUDENT_CONTEXT, NODE_DIAGNOSE_SPEAKING)
    
    # Conditional edge after diagnosis
    async def after_diagnosis_router(state: AgentGraphState) -> str:
        """Router function to decide path after diagnosis."""
        logger.info(f"Router: Checking diagnosis result: {state.get('diagnosis_result')}")
        if state.get("diagnosis_result", {}).get("errors") or state.get("diagnosis_result", {}).get("needs_assistance"):
            logger.info("Router: Errors or assistance needed, routing to generate_feedback.")
            return "generate_feedback"
        logger.info("Router: No errors/assistance needed, routing to save_interaction (skipping feedback).")
        return "save_interaction" # Skip feedback if no errors (for this simple flow)

    workflow.add_conditional_edges(
        NODE_DIAGNOSE_SPEAKING, # Source node
        after_diagnosis_router,    # Async conditional function
        {
            "generate_feedback": NODE_GENERATE_FEEDBACK,
            "save_interaction": NODE_SAVE_INTERACTION 
        }
    )
    
    workflow.add_edge(NODE_GENERATE_FEEDBACK, NODE_SAVE_INTERACTION)
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
