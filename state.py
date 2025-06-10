from typing import TypedDict, List, Optional, Dict, Any, Union
from models import InteractionRequestContext  # Import your Pydantic model


class AgentGraphState(TypedDict):
    # User and session identifiers
    user_id: str
    user_token: str
    session_id: str

    # Input data from the current interaction
    transcript: Optional[str]  # The most recent transcript from the user
    full_submitted_transcript: Optional[
        str
    ]  # Complete transcript when a task is submitted (P2->P5)
    current_context: (
        InteractionRequestContext  # Current UI state and context information
    )
    chat_history: Optional[List[Dict[str, str]]]  # Previous conversation turns
    question_stage: Optional[str]  # Current question stage

    # Student model and memory data
    student_memory_context: Optional[
        Dict[str, Any]
    ]  # Data retrieved from memory system

    # Task management
    next_task_details: Optional[
        Dict[str, Any]
    ]  # Information about the next task to present

    # Processing results
    diagnosis_result: Optional[Dict[str, Any]]  # Results from diagnostic analysis

    # Data from specific welcome flow nodes
    greeting_data: Optional[Dict[str, str]]  # Expected: {"greeting_tts": "..."}
    task_suggestion_llm_output: Optional[
        Dict[str, str]
    ]  # Expected: {"task_suggestion_tts": "..."}

    # Output data to be returned to the frontend
    output_content: Optional[
        Dict[str, Any]
    ]  # Contains text_for_tts, ui_actions, and frontend_rpc_calls

    # For backward compatibility during transition
    feedback_content: Optional[
        Dict[str, Any]
    ]  # To be deprecated in favor of output_content

    # Scaffolding-specific fields
    primary_struggle: Optional[str]  # The primary struggle identified for the student
    secondary_struggles: Optional[List[str]]  # Additional struggles identified
    learning_objective_id: Optional[str]  # ID of the learning objective to focus on
    scaffolding_strategies: Optional[List[Dict[str, Any]]]  # Retrieved scaffolding strategies
    selected_scaffold_type: Optional[str]  # Type of scaffold selected by planner
    scaffold_adaptation_plan: Optional[str]  # Plan for adapting the scaffold
    scaffold_content_type: Optional[str]  # Type of content (template, bank, etc.)
    scaffold_content_name: Optional[str]  # Name of the scaffold content
    scaffold_content: Optional[Dict[str, Any]]  # Actual scaffold content structure
    scaffolding_output: Optional[Dict[str, Any]]  # Final scaffolding output with TTS and UI components
    
    llm_instruction: Optional[
        str
    ]  # Instruction for the LLM, typically from a prompt node
    user_data: Optional[Dict[str, Any]]  # User data, typically from the frontend

    primary_error: Optional[str]  # Primary error from the LLM
    explanation: Optional[str]  # Explanation for the primary error
    document_data: Optional[List[Dict]]

    prioritized_issue: Optional[str]
    chosen_pedagogical_strategy: Optional[str]
