# langgraph-service/agents/modelling_output_formatter.py
import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def modelling_output_formatter_node(state: AgentGraphState) -> dict:
    """
    Formats the output from the modelling generator into a final, user-facing response.
    """
    logger.info("---Executing Modelling Output Formatter---")
    
    payload = state.get("intermediate_modelling_payload", {})
    if not payload:
        return {"final_flow_output": {"text_for_tts": "I had a problem preparing the example.", "ui_actions": []}}

    # 1. Format the text for text-to-speech
    title = payload.get("model_title", "Here's an example")
    summary = payload.get("model_summary", "")
    steps = payload.get("model_steps", [])

    tts_parts = [
        f"Let's walk through an example: {title}.",
        "I'll break it down step-by-step.",
        *steps, # Unpack the list of steps into the tts parts
        f"The key takeaway is: {summary}"
    ]
    text_for_tts = " ".join(filter(None, tts_parts))

    # 2. Create UI actions to display the model clearly
    ui_actions = [
        {
            "action_type": "SHOW_MODEL_EXAMPLE",
            "parameters": {
                "title": title,
                "steps": steps, # Pass the list of steps directly
                "summary": summary
            }
        }
    ]

    # 3. Return the final, client-ready keys directly
    return {
        "final_text_for_tts": text_for_tts,
        "final_ui_actions": ui_actions
    }
