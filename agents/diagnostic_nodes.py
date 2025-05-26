import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def process_speaking_submission_node(state: AgentGraphState) -> dict:
    """
    Process a speaking submission, logging the full submitted transcript and context.
    In a full implementation, this would prepare the data for diagnosis.
    
    Args:
        state: The current agent graph state
        
    Returns:
        Dict with no updates in this stub implementation
    """
    full_transcript = state.get("full_submitted_transcript", "")
    context = state.get("current_context")
    
    logger.info(f"DiagnosticNode: Processing speaking submission for user_id: {state['user_id']}")
    logger.info(f"DiagnosticNode: Full submitted transcript: '{full_transcript}'")
    logger.info(f"DiagnosticNode: Submission context: {context}")
    
    # In Phase 1, this node just acknowledges receipt
    # No state updates in this stub implementation
    return {}

async def diagnose_speaking_stub_node(state: AgentGraphState) -> dict:
    """
    Stub implementation for diagnosing a speaking submission.
    Creates a hardcoded diagnosis result with basic feedback areas.
    
    Args:
        state: The current agent graph state
        
    Returns:
        Dict with diagnosis_result update
    """
    transcript = state.get("full_submitted_transcript") or state.get("transcript", "")
    logger.info(f"DiagnosticNode: Diagnosing speaking submission for user_id: {state['user_id']}")
    
    # Hardcoded diagnosis result
    diagnosis = {
        "summary": "Stub: Speaking needs general practice.",
        "errors": [{"type": "stub_fluency_general", "details": "Needs more practice with fluency."}],
        "strengths": ["stub_content_relevance"],
        "improvement_areas": ["stub_pronunciation", "stub_grammar"],
        "score": {
            "overall": 3,
            "delivery": 3,
            "language_use": 3,
            "topic_development": 3
        }
    }
    
    logger.info(f"DiagnosticNode: Generated diagnosis result: {diagnosis}")
    
    return {"diagnosis_result": diagnosis}
