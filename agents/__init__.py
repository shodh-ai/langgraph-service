# agents/__init__.py

# Student model nodes
from .student_model_node import load_student_data_node, save_interaction_node

# Conversational and curriculum management nodes
from .conversational_manager_node import handle_home_greeting_node, process_conversational_turn_node
from .curriculum_navigator_node import determine_next_pedagogical_step_stub_node
from .practice_selector_node import select_next_practice_or_drill_node
from .scaffolding_provider_node import provide_scaffolding_node

# Teaching and skill drill nodes
from .teaching_delivery_node import deliver_teaching_module_node, manage_skill_drill_node

# AI modeling nodes
from .ai_modeling_node import generate_speaking_model_node, generate_writing_model_node

# Diagnostic nodes
from .diagnostic_nodes import process_speaking_submission_node, diagnose_speaking_stub_node

# Feedback and output nodes
from .feedback_generator_node import generate_speaking_feedback_stub_node
from .session_notes_node import compile_session_notes_stub_node, compile_session_notes_node
from .output_formatter_node import format_final_output_node

# Kept for compatibility (to be deprecated)
from .feedback_node import generate_feedback_stub_node
from .special_feedback_node import generate_test_button_feedback_stub_node

from .handle_welcome import handle_welcome_node
from .student_data import student_data_node
from .welcome_prompt import welcome_prompt_node

from .conversation_handler import conversation_handler_node

__all__ = [
    # Student model nodes
    "load_student_data_node",
    "save_interaction_node",
    
    # Conversational and curriculum management nodes
    "handle_home_greeting_node",
    "process_conversational_turn_node",
    "determine_next_pedagogical_step_stub_node",
    "select_next_practice_or_drill_node",
    "provide_scaffolding_node",
    
    # Teaching and skill drill nodes
    "deliver_teaching_module_node", 
    "manage_skill_drill_node",
    
    # AI modeling nodes
    "generate_speaking_model_node",
    "generate_writing_model_node",
    
    # Diagnostic nodes
    "process_speaking_submission_node",
    "diagnose_speaking_stub_node",
    
    # Feedback and output nodes
    "generate_speaking_feedback_stub_node",
    "compile_session_notes_node",
    "compile_session_notes_stub_node",
    "format_final_output_node",
    
    # Legacy nodes (to be deprecated)
    "generate_feedback_stub_node",
    "generate_test_button_feedback_stub_node",

    # Welcome node
    "handle_welcome_node",
    "welcome_prompt_node",
    "student_data_node",
    "conversation_handler_node"
]
