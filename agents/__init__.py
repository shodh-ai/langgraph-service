# agents/__init__.py

# Student model nodes
from .student_model_node import load_student_data_node, save_interaction_node

# Conversational and curriculum management nodes
from .conversational_manager_node import handle_home_greeting_node
from .curriculum_navigator_node import determine_next_pedagogical_step_stub_node

# Diagnostic nodes
from .diagnostic_nodes import (
    process_speaking_submission_node,
    diagnose_speaking_stub_node,
)

# Feedback and output nodes
from .feedback_generator_node2 import generate_speaking_feedback_stub_node
from .session_notes_node import compile_session_notes_stub_node
from .output_formatter_node import format_final_output_node

# Kept for compatibility (to be deprecated)
from .feedback_node import generate_feedback_stub_node
from .special_feedback_node import generate_test_button_feedback_stub_node

from .handle_welcome import handle_welcome_node
from .student_data import student_data_node
from .welcome_prompt import welcome_prompt_node
from .motivational_support_node import motivational_support_node
from .progress_reporter_node import progress_reporter_node
from .inactivity_prompt_node import inactivity_prompt_node # Added inactivity_prompt_node
from .tech_support_acknowledger_node import tech_support_acknowledger_node # Added tech support node
from .prepare_navigation_node import prepare_navigation_node # Added prepare navigation node
from .session_wrap_up_node import session_wrap_up_node
from .finalize_session_in_mem0_node import finalize_session_in_mem0_node # Added session wrap up node

from .conversation_handler import conversation_handler_node
from .error_generator import error_generator_node
from .feedback_student_data import feedback_student_data_node
from .query_document import query_document_node
from .RAG_document import RAG_document_node
from .feedback_palnner import feedback_planner_node
from .feedback_generator import feedback_generator_node
from .initial_report_generation import initial_report_generation_node
from .pedagogy_generator import pedagogy_generator_node

__all__ = [
    # Student model nodes
    "load_student_data_node",
    "save_interaction_node",
    # Conversational and curriculum management nodes
    "handle_home_greeting_node",
    "determine_next_pedagogical_step_stub_node",
    # Diagnostic nodes
    "process_speaking_submission_node",
    "diagnose_speaking_stub_node",
    # Feedback and output nodes
    "generate_speaking_feedback_stub_node",
    "compile_session_notes_stub_node",
    "format_final_output_node",
    # Legacy nodes (to be deprecated)
    "generate_feedback_stub_node",
    "generate_test_button_feedback_stub_node",
    "inactivity_prompt_node", # Added inactivity_prompt_node
    "tech_support_acknowledger_node", # Added tech support node
    "prepare_navigation_node", # Added prepare navigation node
    "session_wrap_up_node",
    "finalize_session_in_mem0_node", # Added session wrap up node

    # Welcome node
    "handle_welcome_node",
    "welcome_prompt_node",
    "student_data_node",
    "conversation_handler_node",
    "error_generator_node",
    "feedback_student_data_node",
    "query_document_node",
    "RAG_document_node",
    "feedback_planner_node",
    "feedback_generator_node",
    "initial_report_generation_node",
    "pedagogy_generator_node",
]
