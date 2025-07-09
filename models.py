from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- Request Models ---


class InteractionRequestContext(BaseModel):
    """
    Detailed context of the student's current state on the frontend.
    Sent by the client with (almost) every interaction.
    """

    user_id: str  # Should always be present
    toefl_section: Optional[str] = None  # e.g., "Speaking", "Writing"
    question_type: Optional[str] = None  # e.g., "Q1_Independent", "Integrated_Essay"
    task_stage: (
        str  # CRITICAL: What the student is currently doing/what page/mode they are in.
    )
    # e.g., "ROX_WELCOME_INIT", "SPEAKING_TESTING_SUBMITTED", "SPEAKING_FEEDBACK_QA"
    question_stage: Optional[str] = (
        None  # CRITICAL: What the student is currently doing/what page/mode they are in.
    )
    student_name: Optional[str] = None  # Student's name, if available
    goal: Optional[str] = None  # User's stated goal
    feeling: Optional[str] = None  # User's feeling about their skills
    confidence: Optional[str] = None  # User's confidence level
    current_prompt_id: Optional[str] = None
    ui_element_in_focus: Optional[str] = None
    timer_value_seconds: Optional[int] = None
    selected_tools_or_options: Optional[Dict[str, Any]] = None
    # For specific UI events if not covered by task_stage
    feedback_content: Optional[Dict[str, Any]] = None
    modelling_document_data: Optional[Dict[str, Any]] = None

    # Keys from OutputFormatterNode for final client response
    final_text_for_tts: Optional[str] = None
    final_ui_actions: Optional[List[Any]] = None  # Assuming it can be a list of dicts or ReactUIActionModels
    final_next_task_info: Optional[Dict[str, Any]] = None
    final_navigation_instruction: Optional[Dict[str, Any]] = None
    raw_modelling_output: Optional[Dict[str, Any]] = None
    # Flattened keys from modelling_output_content
    think_aloud_sequence: Optional[List[Dict[str, Any]]] = None
    pre_modelling_setup_script: Optional[str] = None
    post_modelling_summary_and_key_takeaways: Optional[str] = None
    comprehension_check_or_reflection_prompt_for_student: Optional[str] = None
    adaptation_notes: Optional[str] = None  # e.g., data from a form field on P9 drill

    # Modelling System Context Fields
    example_prompt_text: Optional[str] = None
    student_goal_context: Optional[str] = None
    student_confidence_context: Optional[str] = None
    teacher_initial_impression: Optional[str] = None
    student_struggle_context: Optional[str] = None
    english_comfort_level: Optional[str] = None

    # Fields for error or raw data logging if needed during RPC/context transfer
    error_during_context_preparation: Optional[str] = None
    raw_frontend_event_data: Optional[Any] = None

    # Student assessment fields (from previous local feedback-system state)
    speaking_strengths: Optional[str] = None  # Positive aspects of student's speaking
    fluency: Optional[str] = None  # Assessment of speech flow and naturalness
    grammar: Optional[str] = None  # Grammar issues and areas for improvement
    vocabulary: Optional[str] = None  # Assessment of vocabulary usage
    goal: Optional[str] = None  # Student's response to first question
    feeling: Optional[str] = None  # Student's response to second question
    confidence: Optional[str] = None  # Student's response to third question

    # Teaching System Context Fields (from origin/feedback-system)
    teacher_persona: Optional[str] = None
    learning_objective_id: Optional[str] = None # Corresponds to LEARNING_OBJECTIVE in CSV for RAG
    student_proficiency_level: Optional[str] = None # Corresponds to STUDENT_PROFICIENCY in CSV for RAG
    current_student_affective_state: Optional[str] = None # Corresponds to STUDENT_AFFECTIVE_STATE in CSV for RAG
    current_lesson_step_number: Optional[int] = None # 1-indexed step number for multi-step content


class InteractionRequest(BaseModel):
    """
    The main request object sent from the LiveKit Agent Service (originating from frontend RPC)
    to the FastAPI/LangGraph backend.
    """

    transcript: Optional[str] = None  # Student's spoken/typed input for this turn
    current_context: InteractionRequestContext  # This should ideally be non-optional
    session_id: str  # LiveKit room SID or a persistent custom session ID; CRITICAL for LangSmith thread_id
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default_factory=list
    )  # e.g. [{"role": "user", "content": "..."}, {"role": "ai", "content": "..."}]

    user_id: Optional[str] = None
    user_token: Optional[str] = None


class InvokeTaskRequest(BaseModel):
    """
    A generic request to invoke a task in the LangGraph backend.
    """
    task_name: str
    json_payload: str


# --- Response Models (and UI Action sub-models) ---


class ReactUIAction(BaseModel):  # RENAMED from DomAction
    """
    Represents a single instruction for the frontend React UI to perform.
    Generated by backend agent nodes.
    """

    action_type: str  # RENAMED from 'action'. String that matches frontend ClientUIActionType enum members
    # e.g., "SHOW_ALERT", "UPDATE_TEXT_CONTENT", "NAVIGATE_TO_PAGE"
    target_element_id: Optional[str] = (
        None  # ID of the HTML element for actions like UPDATE_TEXT_CONTENT
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict
    )  # RESTRUCTURED from 'payload'.
    # Flexible parameters for the action.
    # e.g., for SHOW_ALERT: {"message": "Hello!", "alert_type": "info"}
    # e.g., for UPDATE_TEXT_CONTENT: {"text": "New content", "append": false}
    # e.g., for NAVIGATE_TO_PAGE: {"page_name": "P6_Feedback", "data_for_page": {...}}


from pydantic import BaseModel, Field, Extra  # Ensure Extra is imported

class InteractionResponse(BaseModel):
    raw_initial_report_output: Optional[Dict[str, Any]] = Field(default=None, description="The full, raw JSON output from the initial report generation node.")
    """
    The main response object sent from the FastAPI/LangGraph backend
    back to the LiveKit Agent Service (and then relayed to the frontend).
    """

    response: str  # RENAMED from 'response_for_tts'. Main text for AI to speak (TTS).
    ui_actions: Optional[List[ReactUIAction]] = Field(
        default_factory=list
    )  # RENAMED from 'dom_actions'. Uses the new ReactUIAction model.

    # Optional fields for guiding frontend flow or providing additional info
    next_task_info: Optional[Dict[str, Any]] = (
        None  # e.g., {"type": "SPEAKING", "prompt_id": "SPK_Q1_CITY", "title": "Your Favorite City"}
    )
    navigation_instruction: Optional[Dict[str, Any]] = (
        None  # More structured than just a string
    )
    # e.g., {"target_page": "P6_Feedback",
    #        "data_for_page_load": {"submission_id": "xyz"}}
    # This might be superseded by a specific ui_action of type NAVIGATE_TO_PAGE

    model_config = {"extra": "allow"}
