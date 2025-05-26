from typing import TypedDict, List, Optional, Dict, Any, Union
from models import InteractionRequestContext # Import your Pydantic model

class AgentGraphState(TypedDict):
    # User and session identifiers
    user_id: str
    session_id: str
    
    # Input data from the current interaction
    transcript: Optional[str]  # The most recent transcript from the user
    full_submitted_transcript: Optional[str]  # Complete transcript when a task is submitted (P2->P5)
    current_context: InteractionRequestContext  # Current UI state and context information
    chat_history: Optional[List[Dict[str, str]]]  # Previous conversation turns
    
    # Student model and memory data
    student_memory_context: Optional[Dict[str, Any]]  # Data retrieved from memory system
    
    # Task management
    next_task_details: Optional[Dict[str, Any]]  # Information about the next task to present
    
    # Processing results
    diagnosis_result: Optional[Dict[str, Any]]  # Results from diagnostic analysis
    
    # Output data to be returned to the frontend
    output_content: Optional[Dict[str, Any]]  # Contains text_for_tts, ui_actions, and frontend_rpc_calls
    
    # For backward compatibility during transition
    feedback_content: Optional[Dict[str, Any]]  # To be deprecated in favor of output_content
