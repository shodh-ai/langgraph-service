from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class InteractionRequestContext(BaseModel):
    user_id: str
    toefl_section: Optional[str] = None  # e.g., "Speaking", "Writing"
    question_type: Optional[str] = None  # e.g., "Q1_Independent", "Integrated_Essay"
    task_stage: Optional[str] = None  # e.g., "viewing_prompt", "prep_time", "active_response_speaking", "active_response_writing", "reviewing_feedback", "skill_drill_X"
    current_prompt_id: Optional[str] = None  # Unique ID for the current question/prompt
    ui_element_in_focus: Optional[str] = None  # e.g., "writing_paragraph_2", "speaking_point_1_notes"
    timer_value_seconds: Optional[int] = None  # If a timer is active
    selected_tools_or_options: Optional[Dict[str, Any]] = None  # e.g., {"highlighter_color": "yellow"}
    error: Optional[str] = None  # To capture any error messages, e.g., from custom_data parsing
    original_custom_data: Optional[Any] = None  # To capture the raw custom_data if parsing failed or for logging
    # ... any other relevant state from your UI ...


class InteractionRequest(BaseModel):
    transcript: Optional[str] = None
    current_context: Optional[InteractionRequestContext] = None
    session_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None  # e.g. [{"role": "user", "content": "..."}, {"role": "ai", "content": "..."}]
    usertoken: Optional[str] = None
    user_id: Optional[str] = None

class DomAction(BaseModel):
    action: str  # e.g., "update_text", "highlight_element", "play_audio"
    payload: Dict[str, Any]  # e.g., {"element_id": "paragraph1", "text": "New content"} or {"element_id": "sentence2", "color": "yellow"}


class InteractionResponse(BaseModel):
    response_for_tts: str
    frontend_rpc_calls: Optional[List[Dict[str, Any]]] = None # Optional list of RPC calls for the UI to perform
