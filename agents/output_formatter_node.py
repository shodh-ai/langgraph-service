import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def format_final_output_node(state: AgentGraphState) -> dict:
    """
    Ensures the output_content exists and is properly formatted.
    For Phase 1, this is simple but will be more important in complex flows.
    
    Args:
        state: The current agent graph state
        
    Returns:
        Dict with output_content update if needed
    """
    logger.info(f"OutputFormatterNode: Checking and formatting final output")
    
    # Check if output_content exists
    if not state.get("output_content"):
        # If no output_content has been set by previous nodes, create a default
        logger.warning("OutputFormatterNode: No output_content found in state, creating default")
        default_output = {
            "text_for_tts": "I'm ready to assist with your TOEFL speaking practice.",
            "ui_actions": None
        }
        return {"output_content": default_output}
    
    # In Phase 1, we just log the existing output_content
    output_content = state.get("output_content", {})
    text_for_tts = output_content.get("text_for_tts", "")
    ui_actions = output_content.get("ui_actions", [])
    
    logger.info(f"OutputFormatterNode: Final text_for_tts: '{text_for_tts}'")
    logger.info(f"OutputFormatterNode: Final ui_actions: {ui_actions}")
    
    # For backward compatibility, also update feedback_content
    if "feedback_content" not in state:
        logger.info("OutputFormatterNode: Also setting feedback_content for backward compatibility")
        return {"feedback_content": output_content}
    
    # No changes needed if output_content already exists
    return {}
