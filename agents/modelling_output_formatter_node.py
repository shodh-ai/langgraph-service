# modelling_output_formatter_node.py (NEW STREAMING VERSION)

import logging
import json
from state import AgentGraphState

logger = logging.getLogger(__name__)

# This node now becomes a Python "generator function" using 'yield'
async def modelling_output_formatter_node(state: AgentGraphState):
    """
    Formats the raw sequence from the generator node into a STREAM of individual
    TTS chunks and UI actions.
    """
    logger.info("---Executing Streaming Modelling Output Formatter Node---")

    intermediate_payload = state.get("intermediate_modelling_payload", {})
    if not intermediate_payload:
        logger.warning("Modelling output formatter received an empty payload.")
        return

    think_aloud_sequence = intermediate_payload.get("generated_modeling_and_think_aloud_sequence_json", [])
    
    # Process the pre-setup script first
    if pre_setup_script := intermediate_payload.get("generated_pre_modeling_setup_script"):
        yield {"streaming_text_chunk": pre_setup_script}

    # Process the main sequence item by item
    if isinstance(think_aloud_sequence, list):
        for item in think_aloud_sequence:
            if not isinstance(item, dict):
                continue
            
            item_type = item.get("type")
            
            # If it's a text chunk (either thought or essay), yield it for TTS
            if item_type in ["think_aloud_text", "essay_text_chunk"]:
                if content := item.get("content"):
                    yield {"streaming_text_chunk": content}

            # If it's an essay chunk, ALSO yield a UI action to append it
            if item_type == "essay_text_chunk":
                if content := item.get("content"):
                    yield {
                        "ui_action": {
                            "action_type": "APPEND_TEXT_TO_EDITOR_REALTIME",
                            "parameters": {"text_chunk": content}
                        }
                    }

            # If it's an explicit UI action instruction, yield it directly
            elif item_type == "ui_action_instruction":
                yield {"ui_action": item}

    # Process the post-summary script last
    if post_summary := intermediate_payload.get("generated_post_modeling_summary_and_key_takeaways"):
        yield {"streaming_text_chunk": post_summary}
        
    # And the reflection prompt
    if reflection_prompt := intermediate_payload.get("generated_comprehension_check_or_reflection_prompt_for_student"):
        yield {"streaming_text_chunk": reflection_prompt}

    # This node doesn't need to return a final state, as it has yielded everything.
    # To satisfy LangGraph, we can return a final summary object if needed, but the stream is the main output.
    logger.info("---Finished Streaming from Formatter Node---")
    return
