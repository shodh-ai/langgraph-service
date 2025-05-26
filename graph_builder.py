import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# Import all the agent node functions
from agents import (
    # Student model nodes
    load_student_data_node,
    save_interaction_node,
    
    # Conversational and curriculum management nodes
    handle_home_greeting_node,
    determine_next_pedagogical_step_stub_node,
    
    # Diagnostic nodes
    process_speaking_submission_node,
    diagnose_speaking_stub_node,
    
    # Feedback and output nodes
    generate_speaking_feedback_stub_node,
    compile_session_notes_stub_node,
    format_final_output_node,
    
    # Legacy nodes (kept for backward compatibility)
    generate_feedback_stub_node,
    generate_test_button_feedback_stub_node
)

logger = logging.getLogger(__name__)

# Define node names for clarity
# Student model nodes
NODE_LOAD_STUDENT_DATA = "load_student_data"
NODE_SAVE_INTERACTION = "save_interaction"

# Conversational and curriculum management nodes
NODE_HOME_GREETING = "home_greeting"
NODE_CURRICULUM_NAVIGATOR = "curriculum_navigator"

# Diagnostic nodes
NODE_PROCESS_SPEAKING_SUBMISSION = "process_speaking_submission"
NODE_DIAGNOSE_SPEAKING = "diagnose_speaking"

# Feedback and output nodes
NODE_GENERATE_SPEAKING_FEEDBACK = "generate_speaking_feedback"
NODE_COMPILE_SESSION_NOTES = "compile_session_notes"
NODE_FORMAT_FINAL_OUTPUT = "format_final_output"

# Legacy nodes (kept for backward compatibility)
NODE_GENERATE_FEEDBACK = "generate_feedback"
NODE_FEEDBACK_FOR_TEST_BUTTON = "feedback_for_test_button"

def build_graph():
    """Builds and compiles the LangGraph application with the P1 and P2 submission flows."""
    logger.info("Building LangGraph with AgentGraphState...")
    workflow = StateGraph(AgentGraphState)

    # Add all nodes to the graph
    # Student model nodes
    workflow.add_node(NODE_LOAD_STUDENT_DATA, load_student_data_node)
    workflow.add_node(NODE_SAVE_INTERACTION, save_interaction_node)
    
    # Conversational and curriculum management nodes
    workflow.add_node(NODE_HOME_GREETING, handle_home_greeting_node)
    workflow.add_node(NODE_CURRICULUM_NAVIGATOR, determine_next_pedagogical_step_stub_node)
    
    # Diagnostic nodes
    workflow.add_node(NODE_PROCESS_SPEAKING_SUBMISSION, process_speaking_submission_node)
    workflow.add_node(NODE_DIAGNOSE_SPEAKING, diagnose_speaking_stub_node)
    
    # Feedback and output nodes
    workflow.add_node(NODE_GENERATE_SPEAKING_FEEDBACK, generate_speaking_feedback_stub_node)
    workflow.add_node(NODE_COMPILE_SESSION_NOTES, compile_session_notes_stub_node)
    workflow.add_node(NODE_FORMAT_FINAL_OUTPUT, format_final_output_node)
    
    # Legacy nodes (kept for backward compatibility)
    workflow.add_node(NODE_GENERATE_FEEDBACK, generate_feedback_stub_node)
    workflow.add_node(NODE_FEEDBACK_FOR_TEST_BUTTON, generate_test_button_feedback_stub_node)

    # Define a router node (empty function that doesn't modify state)
    async def router_node(state: AgentGraphState) -> dict:
        logger.info(f"Router node entry point activated for user {state.get('user_id', 'unknown_user')}")
        return {}
        
    # Add the router node to the graph
    NODE_ROUTER = "router"
    workflow.add_node(NODE_ROUTER, router_node)
    
    # Define the initial router function based on task_stage
    async def initial_router_logic(state: AgentGraphState) -> str:
        context = state.get("current_context")
        user_id = state.get('user_id', 'unknown_user')
        
        if not context:
            logger.warning(f"Router: User '{user_id}', missing current_context. Defaulting to load_student_data.")
            return NODE_LOAD_STUDENT_DATA

        task_stage = getattr(context, 'task_stage', None)
        logger.info(f"Router: User '{user_id}', Current task_stage is '{task_stage}'")
        
        # Route based on task_stage
        if task_stage == "session_start_home":
            logger.info(f"Router: User '{user_id}', Routing to load_student_data for home screen.")
            return NODE_LOAD_STUDENT_DATA
        elif task_stage == "speaking_task_submitted":
            logger.info(f"Router: User '{user_id}', Routing to process_speaking_submission.")
            return NODE_PROCESS_SPEAKING_SUBMISSION
        # For backward compatibility
        elif task_stage == "testing_specific_context_from_button":
            logger.info(f"Router: User '{user_id}', Routing to legacy test button feedback.")
            return NODE_FEEDBACK_FOR_TEST_BUTTON
        else:
            logger.info(f"Router: User '{user_id}', Unknown task_stage. Defaulting to load_student_data.")
            return NODE_LOAD_STUDENT_DATA

    # Set the entry point to the router node
    workflow.set_entry_point(NODE_ROUTER)
    
    # Add conditional edges from the router node to the appropriate starting nodes
    workflow.add_conditional_edges(
        NODE_ROUTER,
        initial_router_logic,
        {
            NODE_LOAD_STUDENT_DATA: NODE_LOAD_STUDENT_DATA,
            NODE_PROCESS_SPEAKING_SUBMISSION: NODE_PROCESS_SPEAKING_SUBMISSION,
            NODE_FEEDBACK_FOR_TEST_BUTTON: NODE_FEEDBACK_FOR_TEST_BUTTON
        }
    )

    # Define P1 Flow Edges (Home screen flow)
    workflow.add_edge(NODE_LOAD_STUDENT_DATA, NODE_HOME_GREETING)
    workflow.add_edge(NODE_HOME_GREETING, NODE_CURRICULUM_NAVIGATOR)
    workflow.add_edge(NODE_CURRICULUM_NAVIGATOR, NODE_FORMAT_FINAL_OUTPUT)
    workflow.add_edge(NODE_FORMAT_FINAL_OUTPUT, NODE_SAVE_INTERACTION)
    
    # Define P2 Submission Flow Edges (Speaking task submission flow)
    workflow.add_edge(NODE_PROCESS_SPEAKING_SUBMISSION, NODE_DIAGNOSE_SPEAKING)
    workflow.add_edge(NODE_DIAGNOSE_SPEAKING, NODE_GENERATE_SPEAKING_FEEDBACK)
    workflow.add_edge(NODE_GENERATE_SPEAKING_FEEDBACK, NODE_COMPILE_SESSION_NOTES)
    workflow.add_edge(NODE_COMPILE_SESSION_NOTES, NODE_FORMAT_FINAL_OUTPUT) 
    # Note: NODE_FORMAT_FINAL_OUTPUT -> NODE_SAVE_INTERACTION is already defined above
    
    # Legacy edges for backward compatibility
    workflow.add_edge(NODE_FEEDBACK_FOR_TEST_BUTTON, NODE_SAVE_INTERACTION)
    workflow.add_edge(NODE_GENERATE_FEEDBACK, NODE_SAVE_INTERACTION)
    
    # All paths end after saving the interaction
    workflow.add_edge(NODE_SAVE_INTERACTION, END)

    # Compile the graph
    app_graph = workflow.compile()
    logger.info("LangGraph with P1 and P2 flows built and compiled successfully.")
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
