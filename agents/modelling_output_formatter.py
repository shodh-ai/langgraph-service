# langgraph-service/agents/modelling_output_formatter.py
import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def modelling_output_formatter_node(state: AgentGraphState) -> dict:
    """
    Formats the output from the modelling generator into a final, user-facing response.
    This node is responsible for creating the final `final_text_for_tts` and
    `final_ui_actions` keys that the SSE streamer will yield to the client.
    """
    logger.info("---Executing Modelling Output Formatter---")
    
    payload = state.get("intermediate_modelling_payload")

    # --- Graceful Handling of Missing or Empty Payload ---
    if not payload:
        logger.warning("Modelling Formatter: Received an empty or missing payload.")
        return {
            "final_text_for_tts": "I seem to have had a problem preparing that example. Let's try something else.",
            "final_ui_actions": []
        }

    # --- Handling Structured Errors from the Generator ---
    if 'error' in payload:
        error_message = payload.get('error_message', 'An unknown error occurred.')
        logger.error(f"Modelling Formatter: Received error from generator: {error_message}")
        return {
            "final_text_for_tts": error_message,
            "final_ui_actions": []
        }

    # --- Formatting the Successful Payload ---
    try:
        title = payload.get("model_title", "Here's an example")
        summary = payload.get("model_summary", "")
        steps = payload.get("model_steps", [])

        # 1. Format the text for text-to-speech
        tts_parts = [
            f"Let's walk through an example: {title}.",
            "I'll break it down step-by-step.",
            *steps,  # Unpack the list of steps into the tts parts
            f"The key takeaway is: {summary}"
        ]
        text_for_tts = " ".join(filter(None, tts_parts))

        # 2. Create UI actions to display the model clearly
        ui_actions = [
            {
                "action_type": "SHOW_MODEL_EXAMPLE",
                "parameters": {
                    "title": title,
                    "steps": steps,  # Pass the list of steps directly
                    "summary": summary
                }
            }
        ]

        # 3. Return the final, client-ready keys directly
        return {
            "final_text_for_tts": text_for_tts,
            "final_ui_actions": ui_actions
        }
    except Exception as e:
        logger.error(f"Modelling Formatter: Failed to format payload. Error: {e}", exc_info=True)
        return {
            "final_text_for_tts": "I had trouble formatting the example content. My apologies.",
            "final_ui_actions": []
        }
