# new graph/teaching_output_formatter.py
import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

import copy

async def teaching_output_formatter_node(state: AgentGraphState) -> dict:
    payload = state.get("intermediate_teaching_payload", {})
    sequence = payload.get("sequence", [])
    
    # For now, let's just take the first two steps of the sequence
    first_tts = sequence[0].get("content") if len(sequence) > 0 else "Let's begin."
    second_tts = sequence[1].get("content") if len(sequence) > 1 else ""

    return {
        "final_text_for_tts": f"{first_tts} {second_tts}",
        "final_ui_actions": [{
            "action_type": "PROMPT_FOR_USER_INPUT",
            "parameters": {"timeout_ms": 7000}
        }]
    }