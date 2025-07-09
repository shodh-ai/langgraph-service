# state.py (The Corrected, Final Version)

from typing import TypedDict, List, Dict, Any, Optional

# This is the single source of truth for the entire graph's memory.
# Every key that any node reads or writes MUST be defined here.

AgentGraphState = TypedDict('AgentGraphState', {
    # === Core Identifiers ===
    'user_id': str,
    'session_id': str,
    'task_name': str,
    'current_context': Dict[str, Any],
    'incoming_context': Optional[Dict[str, Any]],
    'chat_history': List[Dict[str, Any]],
    'user_token': str,
    'rag_query_config': Optional[Dict[str, Any]],
    'rag_retrieved_documents': Optional[List[Dict[str, Any]]],

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
    'initial_report_content': Optional[Dict[str, Any]],
    'conversational_tts': Optional[str],
    'task_suggestion_llm_output': Optional[Dict[str, Any]],

    # === Flow-Specific Context Keys (from app.py) ===
    'example_prompt_text': Optional[str],
    'student_struggle_context': Optional[str],
    'english_comfort_level': Optional[str],
    'student_goal_context': Optional[str],
    'student_confidence_context': Optional[str],
    'teacher_initial_impression': Optional[str],

    'Learning_Objective_Focus': Optional[str],
    'STUDENT_PROFICIENCY': Optional[str],
    'STUDENT_AFFECTIVE_STATE': Optional[str],

    'Learning_Objective_Task': Optional[str],
    'Specific_Struggle_Point': Optional[str],
    'Student_Attitude_Context': Optional[str],
    
    'Task': Optional[str],
    'Proficiency': Optional[str],
    'Error': Optional[str],
    'Behavior Factor': Optional[str],
    'diagnosed_error_type': Optional[str],

    'Student_Written_Input_Chunk': Optional[str],
    'Immediate_Assessment_of_Input': Optional[str],
    'Student_Articulated_Thought': Optional[str],

    'Answer One': Optional[str],
    'Answer Two': Optional[str],
    'Answer Three': Optional[str],
    'Initial Impression': Optional[str],
    'Speaking Strengths': Optional[str],
    
    # --- Hierarchical Planning State ---
    # This is now the SINGLE source of truth for the lesson plan.
    'student_memory_context': Optional[Dict[str, Any]],
    'current_lo_to_address': Optional[Dict[str, Any]],
    'pedagogical_plan': Optional[List[Dict[str, Any]]], # DEFINED ONCE
    'current_plan_step_index': Optional[int],          # DEFINED ONCE
    'course_map_data': Optional[Dict[str, Any]],
    'lesson_plan_graph_data': Optional[Dict[str, Any]],
    'current_plan_active': bool,                       # Kept for compatibility
    'last_ai_action': Optional[str],
    'last_action_was': Optional[str],
    'student_intent_for_rox_turn': Optional[str],
    'student_intent_for_lesson_turn': Optional[str],   # Added for teaching flow
    'classified_student_intent': Optional[str],
    'classified_student_entities': Optional[Dict[str, Any]],
    'is_simple_greeting': Optional[bool],

    # === Error Handling ===
    'error_message': Optional[str],
    'route_to_error_handler': bool
})
