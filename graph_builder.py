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
from agents.conversational_turn_manager_node import process_conversational_turn_node
from agents.output_formatter_node import format_final_output_for_client_node
from agents.default_fallback_node import handle_unmatched_interaction_node
from agents.initial_router_node import route_initial_request_node

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
    
    # Add Diagnostic nodes
    workflow.add_node(NODE_DIAGNOSE_SPEAKING, diagnose_submitted_speaking_response_node)
    workflow.add_node(NODE_DIAGNOSE_WRITING, diagnose_submitted_writing_response_node)
    workflow.add_node(NODE_ANALYZE_LIVE_WRITING, analyze_live_writing_chunk_node)
    
    # Add Feedback and guidance nodes
    workflow.add_node(NODE_GENERATE_FEEDBACK, generate_feedback_for_task_node)
    workflow.add_node(NODE_GENERATE_SOCRATIC_QUESTIONS, generate_socratic_question_node)
    workflow.add_node(NODE_PROVIDE_SCAFFOLDING, provide_scaffolding_node)
    workflow.add_node(NODE_GENERATE_MOTIVATION, generate_motivational_message_node)
    
    # Add practice and curriculum navigation nodes
    workflow.add_node(NODE_SELECT_PRACTICE, select_next_practice_or_drill_node)
    workflow.add_node(NODE_DETERMINE_NEXT_STEP, determine_next_pedagogical_step_node)
    
    # Add teaching delivery nodes
    workflow.add_node(NODE_DELIVER_TEACHING, deliver_teaching_module_node)
    workflow.add_node(NODE_MANAGE_SKILL_DRILL, manage_skill_drill_node)
    
    # Add AI modeling nodes
    workflow.add_node(NODE_PREPARE_SPEAKING_MODEL, prepare_speaking_model_response_node)
    workflow.add_node(NODE_PREPARE_WRITING_MODEL, prepare_writing_model_response_node)
    
    # Add session notes node
    workflow.add_node(NODE_COMPILE_SESSION_NOTES, compile_session_notes_node)
    
    # Add conversational turn manager node
    workflow.add_node(NODE_PROCESS_CONVERSATIONAL_TURN, process_conversational_turn_node)
    
    # Add output formatter node
    workflow.add_node(NODE_FORMAT_FINAL_OUTPUT, format_final_output_for_client_node)
    
    # Add default fallback node
    workflow.add_node(NODE_HANDLE_UNMATCHED_INTERACTION, handle_unmatched_interaction_node)
    
    # Add initial router node
    workflow.add_node(NODE_ROUTE_INITIAL_REQUEST, route_initial_request_node)
    
    # Add legacy nodes for compatibility
    workflow.add_node(NODE_FEEDBACK_FOR_TEST_BUTTON, generate_test_button_feedback_stub_node)

    # Set the entry point for the graph
    workflow.set_entry_point(NODE_LOAD_STUDENT_PROFILE)

    # Define the main flow
    # After loading student profile, get task prompt and apply teacher persona
    workflow.add_edge(NODE_LOAD_STUDENT_PROFILE, NODE_GET_TASK_PROMPT)
    workflow.add_edge(NODE_GET_TASK_PROMPT, NODE_GET_RUBRIC)
    workflow.add_edge(NODE_GET_RUBRIC, NODE_GET_AFFECTIVE_STATE)
    workflow.add_edge(NODE_GET_AFFECTIVE_STATE, NODE_APPLY_TEACHER_PERSONA)

    # Define the router function based on task_stage and section
    async def route_after_persona(state: AgentGraphState) -> str:
        context = state.get("current_context")
        if not context:
            logger.warning("LangGraph Router: current_context is missing in state. Routing to default feedback.")
            return NODE_GENERATE_FEEDBACK

        task_stage = getattr(context, 'task_stage', None)
        toefl_section = getattr(context, 'toefl_section', None)
        user_id = state.get('user_id', 'unknown_user')

        logger.info(f"LangGraph Router: User '{user_id}', Current task_stage is '{task_stage}', Section: '{toefl_section}'")
        
        # Route based on task stage and section
        if task_stage == "active_response_speaking":
            return NODE_DIAGNOSE_SPEAKING
        elif task_stage == "active_response_writing":
            return NODE_DIAGNOSE_WRITING
        elif task_stage == "live_writing_analysis":
            return NODE_ANALYZE_LIVE_WRITING
        elif task_stage == "viewing_prompt":
            return NODE_PROVIDE_SCAFFOLDING
        elif task_stage == "practice_selection":
            return NODE_SELECT_PRACTICE
        elif task_stage == "curriculum_navigation":
            return NODE_DETERMINE_NEXT_STEP
        elif task_stage == "teaching_delivery":
            return NODE_DELIVER_TEACHING
        elif task_stage == "skill_drill_management":
            return NODE_MANAGE_SKILL_DRILL
        elif task_stage == "model_speaking":
            return NODE_PREPARE_SPEAKING_MODEL
        elif task_stage == "model_writing":
            return NODE_PREPARE_WRITING_MODEL
        elif task_stage == "conversational_interaction":
            return NODE_PROCESS_CONVERSATIONAL_TURN
        elif task_stage == "unmatched_interaction":
            return NODE_HANDLE_UNMATCHED_INTERACTION
        elif task_stage == "testing_specific_context_from_button":
            return NODE_FEEDBACK_FOR_TEST_BUTTON
        else:
            # Default to generate feedback
            return NODE_GENERATE_FEEDBACK

    # Add conditional edges from persona node
    workflow.add_conditional_edges(
        NODE_APPLY_TEACHER_PERSONA, 
        route_after_persona,
        {
            NODE_DIAGNOSE_SPEAKING: NODE_DIAGNOSE_SPEAKING,
            NODE_DIAGNOSE_WRITING: NODE_DIAGNOSE_WRITING,
            NODE_ANALYZE_LIVE_WRITING: NODE_ANALYZE_LIVE_WRITING,
            NODE_PROVIDE_SCAFFOLDING: NODE_PROVIDE_SCAFFOLDING,
            NODE_SELECT_PRACTICE: NODE_SELECT_PRACTICE,
            NODE_DETERMINE_NEXT_STEP: NODE_DETERMINE_NEXT_STEP,
            NODE_DELIVER_TEACHING: NODE_DELIVER_TEACHING,
            NODE_MANAGE_SKILL_DRILL: NODE_MANAGE_SKILL_DRILL,
            NODE_PREPARE_SPEAKING_MODEL: NODE_PREPARE_SPEAKING_MODEL,
            NODE_PREPARE_WRITING_MODEL: NODE_PREPARE_WRITING_MODEL,
            NODE_PROCESS_CONVERSATIONAL_TURN: NODE_PROCESS_CONVERSATIONAL_TURN,
            NODE_HANDLE_UNMATCHED_INTERACTION: NODE_HANDLE_UNMATCHED_INTERACTION,
            NODE_FEEDBACK_FOR_TEST_BUTTON: NODE_FEEDBACK_FOR_TEST_BUTTON,
            NODE_GENERATE_FEEDBACK: NODE_GENERATE_FEEDBACK
        }
    )

    # Connect diagnostic nodes to feedback generation
    workflow.add_edge(NODE_DIAGNOSE_SPEAKING, NODE_GENERATE_SOCRATIC_QUESTIONS)
    workflow.add_edge(NODE_DIAGNOSE_WRITING, NODE_GENERATE_SOCRATIC_QUESTIONS)
    workflow.add_edge(NODE_GENERATE_SOCRATIC_QUESTIONS, NODE_GENERATE_FEEDBACK)
    
    # Live writing analysis goes directly to feedback
    workflow.add_edge(NODE_ANALYZE_LIVE_WRITING, NODE_GENERATE_FEEDBACK)
    
    # Connect scaffolding to feedback
    workflow.add_edge(NODE_PROVIDE_SCAFFOLDING, NODE_GENERATE_FEEDBACK)
    
    # Connect practice selector to feedback
    workflow.add_edge(NODE_SELECT_PRACTICE, NODE_GENERATE_FEEDBACK)
    
    # Connect curriculum navigator to feedback
    workflow.add_edge(NODE_DETERMINE_NEXT_STEP, NODE_GENERATE_FEEDBACK)
    
    # Connect teaching delivery nodes to feedback
    workflow.add_edge(NODE_DELIVER_TEACHING, NODE_GENERATE_FEEDBACK)
    workflow.add_edge(NODE_MANAGE_SKILL_DRILL, NODE_GENERATE_FEEDBACK)
    
    # Connect AI modeling nodes to feedback
    workflow.add_edge(NODE_PREPARE_SPEAKING_MODEL, NODE_GENERATE_FEEDBACK)
    workflow.add_edge(NODE_PREPARE_WRITING_MODEL, NODE_GENERATE_FEEDBACK)
    
    # Connect conversational turn manager to feedback
    workflow.add_edge(NODE_PROCESS_CONVERSATIONAL_TURN, NODE_GENERATE_FEEDBACK)
    
    # Connect default fallback to feedback
    workflow.add_edge(NODE_HANDLE_UNMATCHED_INTERACTION, NODE_GENERATE_FEEDBACK)
    
    # After feedback, generate motivation
    workflow.add_edge(NODE_GENERATE_FEEDBACK, NODE_GENERATE_MOTIVATION)
    workflow.add_edge(NODE_FEEDBACK_FOR_TEST_BUTTON, NODE_GENERATE_MOTIVATION)
    
    # After motivation, format the final output
    workflow.add_edge(NODE_GENERATE_MOTIVATION, NODE_FORMAT_FINAL_OUTPUT)
    
    # Connect to student model update and logging
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
