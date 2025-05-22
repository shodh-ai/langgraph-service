from typing import TypedDict, List, Optional, Dict, Any
from models import InteractionRequestContext # Import your Pydantic model

class AgentGraphState(TypedDict):
    user_id: str
    session_id: str
    transcript: Optional[str]
    current_context: InteractionRequestContext # Use the Pydantic model for type hinting
    chat_history: Optional[List[Dict[str, str]]]
    
    # Placeholder for data retrieved from memory stub
    student_memory_context: Optional[Dict[str, Any]]

    diagnosis_result: Optional[Dict[str, Any]]
    feedback_content: Optional[Dict[str, Any]] # To hold {"text": "...", "dom_actions": [...]}

    # No need for final_response_for_user or final_dom_actions here,
    # the node that produces the final output will populate feedback_content
    # or a similar field that the FastAPI route handler then uses.
    # Or, a dedicated "final_output" field could be used.
    # Let's use feedback_content for now as the source of final output.
