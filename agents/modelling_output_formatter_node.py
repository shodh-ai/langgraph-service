import logging
from state import AgentGraphState
import json

logger = logging.getLogger(__name__)

# Keys for the outputs from the modelling_generator_node
GENERATOR_OUTPUT_KEYS = [
    "generated_pre_modeling_setup_script",
    "generated_modeling_and_think_aloud_sequence_json",
    "generated_post_modeling_summary_and_key_takeaways",
    "generated_comprehension_check_or_reflection_prompt_for_student",
    "generated_adaptation_for_student_profile_notes"
]

async def modelling_output_formatter_node(state: AgentGraphState) -> dict:
    logger.info(f"ModellingOutputFormatterNode: Current state keys: {list(state.keys())}")
    """
    Formats the outputs from the modelling_generator_node into a structured
    payload for the client interface.
    """
    logger.info(
        f"ModellingOutputFormatterNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )

    intermediate_payload = state.get("intermediate_modelling_payload", {})

    # Retrieve generated content from the state
    pre_setup_script = intermediate_payload.get(GENERATOR_OUTPUT_KEYS[0], "")
    think_aloud_json_str = intermediate_payload.get(GENERATOR_OUTPUT_KEYS[1], "[]")
    post_summary = intermediate_payload.get(GENERATOR_OUTPUT_KEYS[2], "")
    reflection_prompt = intermediate_payload.get(GENERATOR_OUTPUT_KEYS[3], "")
    adaptation_notes = intermediate_payload.get(GENERATOR_OUTPUT_KEYS[4], "")

    # Combine textual parts for a primary TTS/display script
    # Ensuring each part ends with a space if it's not empty, for better concatenation.
    tts_script_parts = []
    if pre_setup_script and isinstance(pre_setup_script, str):
        tts_script_parts.append(pre_setup_script.strip())
    if post_summary and isinstance(post_summary, str):
        tts_script_parts.append(post_summary.strip())
    if reflection_prompt and isinstance(reflection_prompt, str):
        tts_script_parts.append(reflection_prompt.strip())
    
    # Join with a double newline for clear separation if spoken or displayed as a block
    combined_tts_script = "\n\n".join(filter(None, tts_script_parts))

    # Validate and ensure think_aloud_sequence is a list (it should be JSON from the generator)
    # The generator node already attempts to ensure it's a list or defaults to [].
    # Here, we ensure it's a list for the final output.
    think_aloud_sequence = []
    ui_actions_from_modelling = []
    logger.info(f"ModellingOutputFormatterNode: Type of think_aloud_json_str before parsing: {type(think_aloud_json_str)}")
    logger.debug(f"ModellingOutputFormatterNode: Content of think_aloud_json_str (first 300 chars): {str(think_aloud_json_str)[:300]}")

    try:
        # Determine parsed_sequence
        if isinstance(think_aloud_json_str, list):
            parsed_sequence = think_aloud_json_str
            logger.info("ModellingOutputFormatterNode: think_aloud_json_str is already a list, using directly.")
        elif isinstance(think_aloud_json_str, str):
            parsed_sequence = json.loads(think_aloud_json_str)
            logger.info("ModellingOutputFormatterNode: Successfully parsed think_aloud_json_str string.")
        else:
            logger.warning(f"ModellingOutputFormatterNode: think_aloud_json_str is neither a list nor a string. Type: {type(think_aloud_json_str)}. Defaulting to empty list for parsed_sequence.")
            parsed_sequence = []

        # Process parsed_sequence if it's a list
        if isinstance(parsed_sequence, list):
            think_aloud_sequence = parsed_sequence
            # Extract UI actions with robust logic
            for item in think_aloud_sequence:
                if isinstance(item, dict):
                    hints = item.get("ui_action_hints")
                    if hints is None:
                        hints = item.get("ui_actions")
                    
                    if isinstance(hints, list) and hints:
                        ui_actions_from_modelling.extend(h for h in hints if isinstance(h, dict))
                    elif isinstance(hints, dict) and hints:
                         ui_actions_from_modelling.append(hints)
        else:
            logger.warning(f"ModellingOutputFormatterNode: Parsed sequence is not a list (Type: {type(parsed_sequence)}), UI actions will be empty.")
            think_aloud_sequence = [] # Ensure it's an empty list if parsing failed or type was wrong

    except json.JSONDecodeError as e_json:
        logger.error(
            f"ModellingOutputFormatterNode: Failed to decode think_aloud_sequence JSON. String was: '{think_aloud_json_str}'. Error: {e_json}",
            exc_info=True
        )
        think_aloud_sequence = [] # Default to empty list on error
    except Exception as e_general:
        logger.error(
            f"ModellingOutputFormatterNode: An unexpected error occurred processing think_aloud_sequence. Data was: '{think_aloud_json_str}'. Error: {e_general}",
            exc_info=True
        )
        think_aloud_sequence = [] # Default to empty list on error

    # Prepare the final output structure
    # UI actions are extracted from the think_aloud_sequence.
    all_ui_actions = []
    if isinstance(think_aloud_sequence, list):
        for item in think_aloud_sequence:
            if isinstance(item, dict) and "ui_action_hints" in item and isinstance(item["ui_action_hints"], list):
                all_ui_actions.extend(item["ui_action_hints"])
            elif isinstance(item, dict) and "ui_actions" in item and isinstance(item["ui_actions"], list): # Also check for 'ui_actions' for robustness
                 all_ui_actions.extend(item["ui_actions"])

    final_modelling_output = {
        "text_for_tts": combined_tts_script, # Changed key from tts_script
        "think_aloud_sequence": think_aloud_sequence, # This is the structured JSON array
        "pre_modeling_setup_script": pre_setup_script if isinstance(pre_setup_script, str) else str(pre_setup_script),
        "post_modeling_summary_and_key_takeaways": post_summary if isinstance(post_summary, str) else str(post_summary),
        "comprehension_check_or_reflection_prompt_for_student": reflection_prompt if isinstance(reflection_prompt, str) else str(reflection_prompt),
        "adaptation_notes": adaptation_notes if isinstance(adaptation_notes, str) else str(adaptation_notes),
        "ui_actions": all_ui_actions # Populated from think_aloud_sequence
    }

    logger.info(
        f"ModellingOutputFormatterNode: Successfully formatted modelling output. "
        f"TTS script length: {len(combined_tts_script)}, Think-aloud items: {len(think_aloud_sequence)}, "
        f"UI Actions extracted: {len(ui_actions_from_modelling)}"
    )

    return {"modelling_output_content": final_modelling_output}


# Example usage (for local testing if needed)
async def main_test():
    class MockAgentGraphState(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    state1 = MockAgentGraphState({
        "user_id": "test_user_format_1",
        GENERATOR_OUTPUT_KEYS[0]: "Welcome to the modeling session! We'll focus on structuring your paragraph.",
        GENERATOR_OUTPUT_KEYS[1]: [
            {"type": "think_aloud_text", "content": "First, I need a topic sentence."},
            {"type": "essay_text_chunk", "content": "The main idea is crucial for clarity."},
            {"type": "ui_action_instruction", "action_type": "HIGHLIGHT_TEXT_RANGES", "target_element_id": "modelDisplay", "parameters": {"ranges": [{"start":0, "end":10}]}}
        ],
        GENERATOR_OUTPUT_KEYS[2]: "So, we covered topic sentences and supporting details. Key takeaway: always start strong!",
        GENERATOR_OUTPUT_KEYS[3]: "How can you apply this to your next writing task?",
        GENERATOR_OUTPUT_KEYS[4]: "Adapted for student's need for clear structure. Used simple examples."
    })

    print("\n--- Test Case: Modelling Output Formatter ---")
    result = await modelling_output_formatter_node(state1)
    
    if result.get("modelling_output_content"):
        print("Successfully formatted modelling output:")
        formatted_output = result["modelling_output_content"]
        print(f"  TTS Script: {formatted_output['tts_script']}")
        print(f"  Think Aloud Sequence (items): {len(formatted_output['think_aloud_sequence'])}")
        print(f"  Think Aloud Sequence (first item): {formatted_output['think_aloud_sequence'][0] if formatted_output['think_aloud_sequence'] else 'N/A'}")
        print(f"  Adaptation Notes: {formatted_output['adaptation_notes']}")
        print(f"  UI Actions: {formatted_output['ui_actions']}")
        # Also print the individual components for verification
        print(f"  Raw Pre-setup: {formatted_output['pre_modeling_setup_script']}")
        print(f"  Raw Post-summary: {formatted_output['post_modeling_summary_and_key_takeaways']}")
        print(f"  Raw Reflection: {formatted_output['comprehension_check_or_reflection_prompt_for_student']}")
    else:
        print("Error or unexpected result during formatting.")

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main_test())
