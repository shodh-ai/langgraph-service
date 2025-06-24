# refactored agents/output_formatter_node.py

import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def format_final_output_for_client_node(state: AgentGraphState) -> dict:
    """
    A simple, universal exit point for all flows. It takes the standardized
    'final_flow_output' and prepares it for the client.
    """
    logger.info("---UNIFIED EXIT POINT: Formatting final output for client---")
    
    # This node's only job is to find the standardized output key.
    flow_output = state.get("final_flow_output")
    
    if not flow_output:
        logger.warning("Final formatter did not find 'final_flow_output' in state. Returning default error message.")
        flow_output = {
            "text_for_tts": "I'm sorry, an unexpected error occurred.",
            "ui_actions": []
        }
        
    # We can add other top-level keys if needed, like navigation instructions
    # final_response = {**flow_output} # Start with TTS and UI actions
    # final_response["final_navigation_instruction"] = state.get("navigation_instruction")
    
    logger.info(f"Unified Exit Point: Passing through final output.")

    # The state key for this node's output can be the same as its input,
    # or a new one like `client_ready_output`. Let's stick with the final keys.
    return {
        "final_text_for_tts": flow_output.get("text_for_tts"),
        "final_ui_actions": flow_output.get("ui_actions")
    }

