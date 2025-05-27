import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# Import the existing stub node functions
from agents import (
    diagnose_speaking_stub_node,
    generate_feedback_stub_node,
    generate_test_button_feedback_stub_node
)

# Import the new agent node functions
from agents.student_model_node import (
    load_or_initialize_student_profile,
    update_student_skills_after_diagnosis,
    log_interaction_to_memory,
    save_generated_notes_to_memory,
    get_student_affective_state
)
from agents.knowledge_node import (
    get_task_prompt_node,
    get_teaching_material_node,
    get_model_answer_node,
    get_skill_drill_content_node,
    get_rubric_details_node
)
from agents.perspective_shaper_node import apply_teacher_persona_node

logger = logging.getLogger(__name__)

# Define node names for clarity
# Student Model nodes
NODE_LOAD_STUDENT_PROFILE = "load_student_profile"
NODE_UPDATE_STUDENT_SKILLS = "update_student_skills"
NODE_LOG_INTERACTION = "log_interaction"
NODE_SAVE_NOTES = "save_notes"
NODE_GET_AFFECTIVE_STATE = "get_affective_state"

# Knowledge nodes
NODE_GET_TASK_PROMPT = "get_task_prompt"
NODE_GET_TEACHING_MATERIAL = "get_teaching_material"
NODE_GET_MODEL_ANSWER = "get_model_answer"
NODE_GET_SKILL_DRILL = "get_skill_drill"
NODE_GET_RUBRIC = "get_rubric"

# Perspective Shaper node
NODE_APPLY_TEACHER_PERSONA = "apply_teacher_persona"

# Existing nodes (for compatibility)
NODE_DIAGNOSE_SPEAKING = "diagnose_speaking"
NODE_GENERATE_FEEDBACK = "generate_feedback"
NODE_FEEDBACK_FOR_TEST_BUTTON = "feedback_for_test_button_node"

def build_graph():
    """Builds and compiles the LangGraph application with the new structure."""
    logger.info("Building LangGraph with AgentGraphState...")
    workflow = StateGraph(AgentGraphState)

    # Add Student Model nodes
    workflow.add_node(NODE_LOAD_STUDENT_PROFILE, load_or_initialize_student_profile)
    workflow.add_node(NODE_UPDATE_STUDENT_SKILLS, update_student_skills_after_diagnosis)
    workflow.add_node(NODE_LOG_INTERACTION, log_interaction_to_memory)
    workflow.add_node(NODE_SAVE_NOTES, save_generated_notes_to_memory)
    workflow.add_node(NODE_GET_AFFECTIVE_STATE, get_student_affective_state)
    
    # Add Knowledge nodes
    workflow.add_node(NODE_GET_TASK_PROMPT, get_task_prompt_node)
    workflow.add_node(NODE_GET_TEACHING_MATERIAL, get_teaching_material_node)
    workflow.add_node(NODE_GET_MODEL_ANSWER, get_model_answer_node)
    workflow.add_node(NODE_GET_SKILL_DRILL, get_skill_drill_content_node)
    workflow.add_node(NODE_GET_RUBRIC, get_rubric_details_node)
    
    # Add Perspective Shaper node
    workflow.add_node(NODE_APPLY_TEACHER_PERSONA, apply_teacher_persona_node)
    
    # Add existing nodes for compatibility
    workflow.add_node(NODE_DIAGNOSE_SPEAKING, diagnose_speaking_stub_node)
    workflow.add_node(NODE_GENERATE_FEEDBACK, generate_feedback_stub_node)
    workflow.add_node(NODE_FEEDBACK_FOR_TEST_BUTTON, generate_test_button_feedback_stub_node)

    # Set the entry point for the graph
    workflow.set_entry_point(NODE_LOAD_STUDENT_PROFILE)

    # Define the main flow
    # After loading student profile, get task prompt and apply teacher persona
    workflow.add_edge(NODE_LOAD_STUDENT_PROFILE, NODE_GET_TASK_PROMPT)
    workflow.add_edge(NODE_GET_TASK_PROMPT, NODE_GET_RUBRIC)
    workflow.add_edge(NODE_GET_RUBRIC, NODE_GET_AFFECTIVE_STATE)
    workflow.add_edge(NODE_GET_AFFECTIVE_STATE, NODE_APPLY_TEACHER_PERSONA)
    workflow.add_edge(NODE_APPLY_TEACHER_PERSONA, NODE_DIAGNOSE_SPEAKING)

    # Define the router function based on task_stage (keeping existing logic)
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

    # Add conditional edge from NODE_DIAGNOSE_SPEAKING
    workflow.add_conditional_edges(
        NODE_DIAGNOSE_SPEAKING, 
        route_based_on_task_stage,
        {
            NODE_FEEDBACK_FOR_TEST_BUTTON: NODE_FEEDBACK_FOR_TEST_BUTTON,
            NODE_GENERATE_FEEDBACK: NODE_GENERATE_FEEDBACK
        }
    )

    # Connect feedback nodes to update student skills
    workflow.add_edge(NODE_GENERATE_FEEDBACK, NODE_UPDATE_STUDENT_SKILLS)
    workflow.add_edge(NODE_FEEDBACK_FOR_TEST_BUTTON, NODE_UPDATE_STUDENT_SKILLS)
    
    # Connect to interaction logging and note saving
    workflow.add_edge(NODE_UPDATE_STUDENT_SKILLS, NODE_LOG_INTERACTION)
    workflow.add_edge(NODE_LOG_INTERACTION, NODE_SAVE_NOTES)
    workflow.add_edge(NODE_SAVE_NOTES, END)

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
