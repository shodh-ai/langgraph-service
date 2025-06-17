import logging
from state import AgentGraphState
import json

logger = logging.getLogger(__name__)

async def modelling_output_formatter_node(state: AgentGraphState) -> dict:
    """
    Formats the raw output from the modelling_generator_node into a structured
    payload suitable for the client-side UI.

    This node is responsible for:
    1. Extracting the different content pieces (pre-script, sequence, summary, reflection) from the generator's output.
    2. Consolidating all textual components into a single `text_for_tts` string for text-to-speech playback.
    3. Generating a `ui_actions` list, for example, to append text to the UI for a 'live typing' effect.
    4. Returning a dictionary under the `modelling_output_content` key in the agent state, which contains the consolidated text, UI actions, and individual components for reference.

    Args:
        state (AgentGraphState): The current state of the graph.

    Returns:
        dict: A dictionary with the key `modelling_output_content` containing the formatted output.
    """
    logger.info("---Executing Modelling Output Formatter Node---")

    # Retrieve the payload from the generator node
    intermediate_payload = state.get("intermediate_modelling_payload", {})
    if not intermediate_payload:
        logger.warning("Modelling output formatter node received an empty intermediate payload.")
        # Return a default structure to avoid breaking the graph
        return {
            "modelling_output_content": {
                "text_for_tts": "",
                "pre_modeling_setup_script": "",
                "post_modeling_summary": "",
                "reflection_prompt": "",
                "ui_actions": [],
            }
        }

    # Extract individual fields from the payload
    pre_setup_script = intermediate_payload.get("generated_pre_modeling_setup_script", "")
    think_aloud_sequence = intermediate_payload.get("generated_modeling_and_think_aloud_sequence_json", [])
    post_summary = intermediate_payload.get("generated_post_modeling_summary_and_key_takeaways", "")
    reflection_prompt = intermediate_payload.get("generated_post_modeling_reflection_prompt", "")
    
    # Consolidate all text for TTS and generate UI actions
    text_for_tts_parts = [pre_setup_script]
    ui_actions = []

    if isinstance(think_aloud_sequence, list):
        for item in think_aloud_sequence:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in ["essay_text_chunk", "think_aloud_text"] and "content" in item:
                    content = item.get("content", "")
                    text_for_tts_parts.append(content)
                    if item_type == "essay_text_chunk":
                        ui_actions.append({
                            "action_type": "APPEND_TEXT_TO_EDITOR_REALTIME",
                            "parameters": {
                                "text_chunk": content
                            }
                        })
                elif item_type == "ui_action_instruction":
                    action_type = item.get("action_type")
                    if action_type == "HIGHLIGHT_TEXT_RANGES":
                        ui_actions.append({
                            "action_type": "HIGHLIGHT_TEXT_RANGES",
                            "parameters": item.get("parameters", {})
                        })
    
    text_for_tts_parts.append(post_summary)
    text_for_tts_parts.append(reflection_prompt)

    text_for_tts = " ".join(filter(None, text_for_tts_parts))

    output_content = {
        "text_for_tts": text_for_tts,
        "pre_modeling_setup_script": pre_setup_script,
        "post_modeling_summary": post_summary,
        "reflection_prompt": reflection_prompt,
        "ui_actions": ui_actions, # List of specific UI actions to be executed
    }

    logger.debug(f"Formatted modelling output: {json.dumps(output_content, indent=2)}")

    return {"modelling_output_content": output_content}
