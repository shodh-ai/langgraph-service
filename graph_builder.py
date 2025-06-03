import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# Import the existing stub node functions
from agents import (
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
from agents.diagnostic_nodes import (
    diagnose_submitted_speaking_response_node,
    diagnose_submitted_writing_response_node,
    analyze_live_writing_chunk_node
)
from agents.feedback_generator_node import generate_feedback_for_task_node
from agents.socratic_questioning_node import generate_socratic_question_node
from agents.scaffolding_provider_node import provide_scaffolding_node
from agents.motivational_speaker_node import generate_motivational_message_node
from agents.practice_selector_node import select_next_practice_or_drill_node
from agents.curriculum_navigator_node import determine_next_pedagogical_step_node
from agents.teaching_delivery_node import (
    deliver_teaching_module_node,
    manage_skill_drill_node
)
from agents.ai_modeling_node import (
    prepare_speaking_model_response_node,
    prepare_writing_model_response_node
)
from agents.session_notes_node import compile_session_notes_node
from agents.conversational_turn_manager_node import process_conversational_turn_node, generate_welcome_greeting_node
from agents.output_formatter_node import format_final_output_for_client_node
from agents.default_fallback_node import handle_unmatched_interaction_node
from agents.initial_router_node import route_initial_request_node, start_graph_node

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

# Diagnostic nodes
NODE_DIAGNOSE_SPEAKING = "diagnose_speaking"
NODE_DIAGNOSE_WRITING = "diagnose_writing"
NODE_ANALYZE_LIVE_WRITING = "analyze_live_writing"

# Feedback and guidance nodes
NODE_GENERATE_FEEDBACK = "generate_feedback"
NODE_GENERATE_SOCRATIC_QUESTIONS = "generate_socratic_questions"
NODE_PROVIDE_SCAFFOLDING = "provide_scaffolding"
NODE_GENERATE_MOTIVATION = "generate_motivation"

# Practice and curriculum navigation nodes
NODE_SELECT_PRACTICE = "select_practice"
NODE_DETERMINE_NEXT_STEP = "determine_next_step"

# Teaching delivery nodes
NODE_DELIVER_TEACHING = "deliver_teaching"
NODE_MANAGE_SKILL_DRILL = "manage_skill_drill"

# AI modeling nodes
NODE_PREPARE_SPEAKING_MODEL = "prepare_speaking_model"
NODE_PREPARE_WRITING_MODEL = "prepare_writing_model"

# Session notes node
NODE_COMPILE_SESSION_NOTES = "compile_session_notes"

# Conversational turn manager node
NODE_PROCESS_CONVERSATIONAL_TURN = "process_conversational_turn"
NODE_GENERATE_WELCOME_GREETING = "generate_welcome_greeting"

# Output formatter node
NODE_FORMAT_FINAL_OUTPUT = "format_final_output"

# Default fallback node
NODE_HANDLE_UNMATCHED_INTERACTION = "handle_unmatched_interaction"

# Initial router node
NODE_ROUTE_INITIAL_REQUEST = "route_initial_request"

# Legacy nodes (for compatibility)
NODE_FEEDBACK_FOR_TEST_BUTTON = "feedback_for_test_button_node"

def build_graph():
    """Builds and compiles the LangGraph application with the new structure."""
    logger.info("Building LangGraph with AgentGraphState...")
    workflow = StateGraph(AgentGraphState)

    # Add all nodes to the graph
    # Define all nodes in a dictionary first
    all_nodes = {
        NODE_LOAD_STUDENT_PROFILE: load_or_initialize_student_profile,
        NODE_UPDATE_STUDENT_SKILLS: update_student_skills_after_diagnosis,
        NODE_LOG_INTERACTION: log_interaction_to_memory,
        NODE_SAVE_NOTES: save_generated_notes_to_memory,
        NODE_GET_AFFECTIVE_STATE: get_student_affective_state,
        NODE_GET_TASK_PROMPT: get_task_prompt_node,
        NODE_GET_TEACHING_MATERIAL: get_teaching_material_node,
        NODE_GET_MODEL_ANSWER: get_model_answer_node,
        NODE_GET_SKILL_DRILL: get_skill_drill_content_node,
        NODE_GET_RUBRIC: get_rubric_details_node,
        NODE_APPLY_TEACHER_PERSONA: apply_teacher_persona_node,
        NODE_DIAGNOSE_SPEAKING: diagnose_submitted_speaking_response_node,
        NODE_DIAGNOSE_WRITING: diagnose_submitted_writing_response_node,
        NODE_ANALYZE_LIVE_WRITING: analyze_live_writing_chunk_node,
        NODE_GENERATE_FEEDBACK: generate_feedback_for_task_node,
        NODE_GENERATE_SOCRATIC_QUESTIONS: generate_socratic_question_node,
        NODE_PROVIDE_SCAFFOLDING: provide_scaffolding_node,
        NODE_GENERATE_MOTIVATION: generate_motivational_message_node,
        NODE_SELECT_PRACTICE: select_next_practice_or_drill_node,
        NODE_DETERMINE_NEXT_STEP: determine_next_pedagogical_step_node,
        NODE_DELIVER_TEACHING: deliver_teaching_module_node,
        NODE_MANAGE_SKILL_DRILL: manage_skill_drill_node,
        NODE_PREPARE_SPEAKING_MODEL: prepare_speaking_model_response_node,
        NODE_PREPARE_WRITING_MODEL: prepare_writing_model_response_node,
        NODE_COMPILE_SESSION_NOTES: compile_session_notes_node,
        NODE_PROCESS_CONVERSATIONAL_TURN: process_conversational_turn_node,
        NODE_GENERATE_WELCOME_GREETING: generate_welcome_greeting_node, # Added for welcome flow
        NODE_FORMAT_FINAL_OUTPUT: format_final_output_for_client_node,
        NODE_HANDLE_UNMATCHED_INTERACTION: handle_unmatched_interaction_node,
        NODE_ROUTE_INITIAL_REQUEST: start_graph_node, # Changed to start_graph_node for entry point
        NODE_FEEDBACK_FOR_TEST_BUTTON: generate_test_button_feedback_stub_node,
    }

    # Add all nodes to the graph by iterating through the dictionary
    for node_name, node_action in all_nodes.items():
        workflow.add_node(node_name, node_action)
    logger.info(f"Added {len(all_nodes)} nodes to the graph.")

    # Set the entry point for the graph
    workflow.set_entry_point(NODE_ROUTE_INITIAL_REQUEST)

    # Conditional routing after initial request processing
    initial_router_path_map = {
        "load_or_initialize_student_profile": NODE_LOAD_STUDENT_PROFILE,
        "process_speaking_submission_node": NODE_DIAGNOSE_SPEAKING, 
        "process_conversational_turn_node": NODE_PROCESS_CONVERSATIONAL_TURN,
        "handle_unmatched_interaction_node": NODE_HANDLE_UNMATCHED_INTERACTION,
    }
    workflow.add_conditional_edges(
        NODE_ROUTE_INITIAL_REQUEST, 
        route_initial_request_node, # The actual router function from initial_router_node.py
        initial_router_path_map
    )

    # --- ROX WELCOME FLOW --- 
    # Initial Router -> NODE_LOAD_STUDENT_PROFILE (handled by conditional edge above)
    # For ROX_WELCOME_INIT, the flow is mostly linear after initial routing.
    # Note: NODE_GET_AFFECTIVE_STATE is skipped in this specific welcome flow for simplicity,
    # persona node can still operate with default/empty affective state.
    workflow.add_edge(NODE_LOAD_STUDENT_PROFILE, NODE_APPLY_TEACHER_PERSONA)
    # NODE_APPLY_TEACHER_PERSONA will then conditionally route to NODE_GENERATE_WELCOME_GREETING (see below)

    workflow.add_edge(NODE_GENERATE_WELCOME_GREETING, NODE_DETERMINE_NEXT_STEP)
    # For ROX_WELCOME_INIT, determine_next_step_node suggests a task and outputs to task_suggestion_tts_intermediate etc.
    # Then it should go to format_final_output.
    workflow.add_edge(NODE_DETERMINE_NEXT_STEP, NODE_FORMAT_FINAL_OUTPUT)
    # NODE_FORMAT_FINAL_OUTPUT for ROX_WELCOME_INIT combines greeting and task suggestion.
    # The rest of the path from NODE_FORMAT_FINAL_OUTPUT to END is common (logging, etc.)

    # --- GENERAL FLOW CONNECTIONS --- 
    # Connect student profile loading to affective state (for non-welcome flows or if needed by persona)
    # This path is taken if NODE_LOAD_STUDENT_PROFILE is reached by other means or if welcome flow needs it later.
    # For now, welcome flow goes NODE_LOAD_STUDENT_PROFILE -> NODE_APPLY_TEACHER_PERSONA directly.
    # If NODE_GET_AFFECTIVE_STATE is essential before NODE_APPLY_TEACHER_PERSONA for all flows including welcome,
    # then the welcome flow edge should be: NODE_LOAD_STUDENT_PROFILE -> NODE_GET_AFFECTIVE_STATE
    # and then NODE_GET_AFFECTIVE_STATE -> NODE_APPLY_TEACHER_PERSONA.
    # For now, keeping welcome flow simpler as per summary (Load Profile -> Apply Persona).
    # General flows might still use: workflow.add_edge(NODE_LOAD_STUDENT_PROFILE, NODE_GET_AFFECTIVE_STATE)
    # workflow.add_edge(NODE_GET_AFFECTIVE_STATE, NODE_APPLY_TEACHER_PERSONA) # This is the general path

    # Define the router function based on task_stage and section
    async def route_after_persona(state: AgentGraphState) -> str:
        """Routes based on the current context after persona has been applied."""
        logger.info(f"Routing after persona. Current state keys: {state.keys()}")
        current_context = state.get("current_context")
        
        if not current_context:
            logger.warning("route_after_persona: current_context is None. Defaulting to fallback.")
            return "to_fallback"

        task_stage = getattr(current_context, "task_stage", None)
        logger.info(f"route_after_persona: task_stage='{task_stage}'")

        if task_stage == "ROX_WELCOME_INIT":
            logger.info("Routing after persona for ROX_WELCOME_INIT to welcome greeting.")
            return "to_welcome_greeting"
        elif task_stage == "ROX_CONVERSATION_TURN":
            logger.info("Routing to conversational turn processing.")
            return "to_conversational_turn"
        elif task_stage == "SPEAKING_TESTING_SUBMITTED":
            logger.info("Routing to speaking diagnosis after persona.")
            return "to_speaking_diagnosis"
        
        logger.warning(f"route_after_persona: Unhandled task_stage '{task_stage}'. Defaulting to fallback.")
        return "to_fallback"

    # Conditional routing after persona application (handles ROX_WELCOME_INIT via modified route_after_persona)
    path_map_after_persona = {
        "to_welcome_greeting": NODE_GENERATE_WELCOME_GREETING, # New path for welcome flow
        "to_conversational_turn": NODE_PROCESS_CONVERSATIONAL_TURN,
        "to_speaking_diagnosis": NODE_DIAGNOSE_SPEAKING, 
        "to_fallback": NODE_HANDLE_UNMATCHED_INTERACTION
    }
    workflow.add_conditional_edges(
        NODE_APPLY_TEACHER_PERSONA,
        route_after_persona, # This function is now modified to handle ROX_WELCOME_INIT
        path_map_after_persona
    )

    # Connect task prompt fetching to various diagnostic/processing nodes (General Flow)
    workflow.add_edge(NODE_GET_TASK_PROMPT, NODE_DIAGNOSE_SPEAKING)
    workflow.add_edge(NODE_GET_TASK_PROMPT, NODE_DIAGNOSE_WRITING)
    workflow.add_edge(NODE_GET_TASK_PROMPT, NODE_PROCESS_CONVERSATIONAL_TURN)

    # Connect diagnostic nodes to feedback generation (General Flow)
    workflow.add_edge(NODE_DIAGNOSE_SPEAKING, NODE_GENERATE_SOCRATIC_QUESTIONS)
    workflow.add_edge(NODE_DIAGNOSE_WRITING, NODE_GENERATE_SOCRATIC_QUESTIONS)
    workflow.add_edge(NODE_GENERATE_SOCRATIC_QUESTIONS, NODE_GENERATE_FEEDBACK)
    
    workflow.add_edge(NODE_ANALYZE_LIVE_WRITING, NODE_GENERATE_FEEDBACK)
    workflow.add_edge(NODE_PROVIDE_SCAFFOLDING, NODE_GENERATE_FEEDBACK)
    workflow.add_edge(NODE_SELECT_PRACTICE, NODE_GENERATE_FEEDBACK)
    # workflow.add_edge(NODE_DETERMINE_NEXT_STEP, NODE_GENERATE_FEEDBACK) # This is now conditional. For welcome, it goes to FORMAT_FINAL_OUTPUT.
                                                                        # For other flows, determine_next_step might lead to feedback if it's part of a different loop.
                                                                        # For now, removing this generic edge. Specific flows from determine_next_step should be explicit.

    workflow.add_edge(NODE_DELIVER_TEACHING, NODE_GENERATE_FEEDBACK)
    workflow.add_edge(NODE_MANAGE_SKILL_DRILL, NODE_GENERATE_FEEDBACK)
    workflow.add_edge(NODE_PREPARE_SPEAKING_MODEL, NODE_GENERATE_FEEDBACK)
    workflow.add_edge(NODE_PREPARE_WRITING_MODEL, NODE_GENERATE_FEEDBACK)
    workflow.add_edge(NODE_PROCESS_CONVERSATIONAL_TURN, NODE_GENERATE_FEEDBACK)
    workflow.add_edge(NODE_HANDLE_UNMATCHED_INTERACTION, NODE_GENERATE_FEEDBACK)
    
    workflow.add_edge(NODE_GENERATE_FEEDBACK, NODE_GENERATE_MOTIVATION)
    workflow.add_edge(NODE_FEEDBACK_FOR_TEST_BUTTON, NODE_GENERATE_MOTIVATION) # Legacy button flow
    
    workflow.add_edge(NODE_GENERATE_MOTIVATION, NODE_FORMAT_FINAL_OUTPUT)
    
    # Common tail: Connect to student model update, logging, and end
    workflow.add_edge(NODE_FORMAT_FINAL_OUTPUT, NODE_UPDATE_STUDENT_SKILLS)
    workflow.add_edge(NODE_UPDATE_STUDENT_SKILLS, NODE_COMPILE_SESSION_NOTES)
    workflow.add_edge(NODE_COMPILE_SESSION_NOTES, NODE_LOG_INTERACTION)
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
