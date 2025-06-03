# backend_ai_service_langgraph/state.py
from typing import TypedDict, List, Optional, Dict, Any
from models import InteractionRequestContext, ReactUIAction # Assuming ReactUIAction is your Pydantic model for UI actions

class AgentGraphState(TypedDict):
    # === Core Identifiers & Inputs ===
    user_id: str
    session_id: str # Also used as thread_id for LangSmith
    
    transcript: Optional[str] # Current user utterance for conversational turns
    full_submitted_transcript: Optional[str] # For P2 Speaking/P3 Writing submissions
    current_context: InteractionRequestContext # Rich context from frontend
    chat_history: List[Dict[str, str]] # Default to empty list if not provided initially

    # === Data From/For StudentModelNode & Memory ===
    student_memory_context: Optional[Dict[str, Any]] # Profile, history, skills, affect from Mem0
    # Example nested structure for student_memory_context:
    # {
    #     "profile": {"name": "Jane", "level": "Beginner", "native_language": "Spanish"},
    #     "skills": {"speaking_fluency": 60, "grammar_sva": 50},
    #     "affective_state": "neutral",
    #     "learning_goals": ["improve_speaking_q1"],
    #     "last_interaction_summary": "Discussed S-V agreement.",
    #     "feedback_agenda": Optional[List[Dict[str,Any]]] # For P6 Feedback page
    #     "feedback_agenda_status": Optional[Dict[str,str]] # e.g. {"err1":"discussed"}
    # }
    interaction_summary_for_memory: Optional[Dict[str, Any]] # Prepared by a node to be saved by StudentModelNode
    notes_to_save_in_memory: Optional[Dict[str, Any]] # Prepared by SessionNotesNode

    # === Data From/For PerspectiveShaperNode ===
    active_persona_details: Optional[Dict[str, Any]] # e.g., {"name": "Structuralist", "style_directives": "..."}

    # === Data From/For CurriculumNavigatorNode ===
    # This is the primary output of CurriculumNavigatorNode, to be used by OutputFormatterNode
    # and also by app.py to include in InteractionResponse
    next_task_details_for_client: Optional[Dict[str, Any]] 
    # Example: {"page_name": "P2_SpeakingExercise", "type": "SPEAKING", "question_type": "Q1", "prompt_id": "SPK_Q1_001", "title": "Favorite City"}

    # === Data From/For DiagnosticNode(s) ===
    diagnosis_result: Optional[Dict[str, Any]] # Structured output from a diagnostic node
    # Example: {"summary": "...", "errors": [{"type":"fluency", ...}], "strengths": [...], "data_for_p6_page": {...}}

    # === Intermediate Outputs from Specialized Nodes (used by OutputFormatterNode or other nodes) ===
    # For P1 Welcome Sequence
    greeting_tts_intermediate: Optional[str]
    greeting_ui_actions_intermediate: Optional[List[Dict[str, Any]]] # List of ReactUIAction-like dicts
    
    status_update_tts_intermediate: Optional[str]
    status_update_ui_actions_intermediate: Optional[List[Dict[str, Any]]]
    
    suggestion_tts_intermediate: Optional[str] # From CurriculumNavigatorNode
    suggestion_ui_actions_intermediate: Optional[List[Dict[str, Any]]] # From CurriculumNavigatorNode

    # For Feedback Generation (P6) - one feedback item at a time
    current_feedback_focus_id: Optional[str] # e.g., "err1_fluency"
    current_feedback_item_details: Optional[Dict[str, Any]] # Details of the error/point to give feedback on
    # The FeedbackGeneratorNode will produce text and UI actions for this *single* item.

    # For Teaching Delivery (P7)
    current_lesson_id: Optional[str]
    current_lesson_step_id: Optional[str]
    teaching_material_for_step: Optional[Dict[str, Any]] # JSON content for the current teaching step

    # For AI Modeling (P8)
    current_model_task_id: Optional[str]
    current_modeling_step_id: Optional[str]
    model_script_chunk_tts: Optional[str]
    model_think_aloud_tts: Optional[str]
    model_ui_actions_for_display: Optional[List[Dict[str, Any]]]
    
    # For general conversational turns (output from ConversationalTurnManagerNode)
    # This node might directly populate 'output_content' or set these intermediate fields
    # if another node is meant to further process/combine its output.
    # Let's assume it will populate output_content for now as per your example.

    # === Final Output Assembly (Populated by OutputFormatterNode or last acting node) ===
    # This 'output_content' is what app.py will primarily look for to build the API response.
    # It should contain the fully assembled TTS text and UI actions for the current interaction.
    output_content: Optional[Dict[str, Any]] # Structure: {"text_for_tts": str, "ui_actions": List[ReactUIAction-like dicts]}
    
    # These are specific fields that app.py can also look for in the final_state
    # to populate distinct parts of the InteractionResponse.
    # OutputFormatterNode should aim to populate these from output_content or other state fields.
    final_response_text_for_api: Optional[str]
    final_ui_actions_for_api: Optional[List[Dict[str, Any]]] # List of ReactUIAction-like dicts
    # final_next_task_info_for_api is already covered by 'next_task_details_for_client'
    final_navigation_instruction_for_api: Optional[Dict[str, Any]] # e.g., {"page_name": "P2", "data": {...}}

    # === Control Flow & Error Handling ===
    # next_node_override: Optional[str] # If a node wants to force the next step
    error_message_for_user: Optional[str] # If an error occurs that needs to be relayed
    graph_execution_status: Optional[str] # e.g., "success", "error_in_node_X"