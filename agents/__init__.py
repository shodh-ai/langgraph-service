# agents/__init__.py

# Student model nodes
from .student_model_node import load_student_data_node, save_interaction_node

# Conversational and curriculum management nodes
from .conversational_manager_node import handle_home_greeting_node
from .curriculum_navigator_node import determine_next_pedagogical_step_stub_node

# Diagnostic nodes
from .diagnostic_nodes import process_speaking_submission_node, diagnose_speaking_stub_node

# Feedback and output nodes
from .feedback_generator_node import generate_speaking_feedback_stub_node
from .session_notes_node import compile_session_notes_stub_node
from .output_formatter_node import format_final_output_node

# Kept for compatibility (to be deprecated)
from .feedback_node import generate_feedback_stub_node
from .special_feedback_node import generate_test_button_feedback_stub_node

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
    "generate_test_button_feedback_stub_node"
]
