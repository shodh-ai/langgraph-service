# langgraph-service/agents/cowriting_output_formatter.py
import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def cowriting_output_formatter_node(state: AgentGraphState) -> dict:
    """
    Formats the output from the co-writing generator into a final, user-facing response.
    """
    logger.info("---Executing Co-Writing Output Formatter---")
    
    payload = state.get("intermediate_cowriting_payload", {})
    if not payload:
        return {
            "final_text_for_tts": "I had a problem coming up with suggestions.",
            "final_ui_actions": []
        }

    # 1. Format the text for text-to-speech
    suggestion1_text = payload.get("suggestion_1_text", "")
    suggestion1_exp = payload.get("suggestion_1_explanation", "")
    suggestion2_text = payload.get("suggestion_2_text", "")
    suggestion2_exp = payload.get("suggestion_2_explanation", "")

    tts_parts = [
        "Here are a couple of ideas to keep you going.",
        f"First, you could try this: {suggestion1_text}.",
        f"{suggestion1_exp}",
        "Or, how about this?",
        f"{suggestion2_text}. {suggestion2_exp}"
    ]
    text_for_tts = " ".join(filter(None, tts_parts))

    # 2. Create UI actions to show the suggestions as clickable buttons or cards
    ui_actions = [
        {
            "action_type": "SHOW_SUGGESTION_CARD",
            "parameters": {
                "title": "Suggestion 1",
                "text": suggestion1_text,
                "explanation": suggestion1_exp
            }
        },
        {
            "action_type": "SHOW_SUGGESTION_CARD",
            "parameters": {
                "title": "Suggestion 2",
                "text": suggestion2_text,
                "explanation": suggestion2_exp
            }
        }
    ]

    # 3. Return the final, client-ready keys directly
    return {
        "final_text_for_tts": text_for_tts,
        "final_ui_actions": ui_actions
    }
