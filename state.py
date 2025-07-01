from typing import TypedDict, List, Dict, Any, Optional

# This is the single source of truth for the entire graph's memory.
# Every key that any node reads or writes MUST be defined here.

AgentGraphState = TypedDict('AgentGraphState', {
    # === Core Identifiers ===
    'user_id': str,
    'session_id': str,
    'task_name': str,
    'current_context': Dict[str, Any],
    'chat_history': List[Dict[str, Any]],
    'user_token': str,

    # === Universal Input/Output Keys ===
    'transcript': Optional[str],
    'final_text_for_tts': Optional[str],
    'final_ui_actions': List[Dict[str, Any]],
    
    # === RAG & Intermediate Payloads ===
    'rag_document_data': List[Dict[str, Any]],
    'intermediate_modelling_payload': Optional[Dict[str, Any]],
    'intermediate_teaching_payload': Optional[Dict[str, Any]],
    'intermediate_scaffolding_payload': Optional[Dict[str, Any]],
    'intermediate_feedback_payload': Optional[Dict[str, Any]],
    'intermediate_cowriting_payload': Optional[Dict[str, Any]],
    'intermediate_pedagogy_payload': Optional[Dict[str, Any]],

    # Keys from the new Initial Report -> Pedagogy Flow
    'initial_report_content': Optional[Dict[str, Any]], # From initial_report_generation_node
    'conversational_tts': Optional[str], # From initial_report_generation_node
    'task_suggestion_llm_output': Optional[Dict[str, Any]], # From pedagogy_generator_node

    # === Flow-Specific Context Keys (from app.py) ===
    # These are the keys that were being dropped. Now they are official.
    
    # -- Modelling --
    'example_prompt_text': Optional[str],
    'student_struggle_context': Optional[str],
    'english_comfort_level': Optional[str],
    'student_goal_context': Optional[str],
    'student_confidence_context': Optional[str],
    'teacher_initial_impression': Optional[str],

    # -- Teaching --
    'Learning_Objective_Focus': Optional[str], # Also used by Cowriting
    'STUDENT_PROFICIENCY': Optional[str],
    'STUDENT_AFFECTIVE_STATE': Optional[str],

    # -- Scaffolding --
    'Learning_Objective_Task': Optional[str],
    'Specific_Struggle_Point': Optional[str],
    'Student_Attitude_Context': Optional[str],
    
    # -- Feedback --
    'Task': Optional[str],
    'Proficiency': Optional[str],
    'Error': Optional[str],
    'Behavior Factor': Optional[str],
    'diagnosed_error_type': Optional[str],

    # -- Cowriting --
    'Student_Written_Input_Chunk': Optional[str],
    'Immediate_Assessment_of_Input': Optional[str],
    'Student_Articulated_Thought': Optional[str],

    # -- Pedagogy --
    'Answer One': Optional[str],
    'Answer Two': Optional[str],
    'Answer Three': Optional[str],
    'Initial Impression': Optional[str],
    'Speaking Strengths': Optional[str],
    
    # === Error Handling ===
    'error_message': Optional[str],
    'route_to_error_handler': bool,
    
    # === Welcome Flow ===
    'welcome_text': Optional[str],  # Welcome message text for TTS
    'welcome_ui_actions': Optional[List[Dict[str, Any]]]  # UI actions for welcome flow navigation
})
