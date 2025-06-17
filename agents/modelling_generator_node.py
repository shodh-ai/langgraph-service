import logging
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

# Define the keys for the five outputs we expect the LLM to generate
OUTPUT_KEYS = [
    "generated_pre_modeling_setup_script",
    "generated_modeling_and_think_aloud_sequence_json",
    "generated_post_modeling_summary_and_key_takeaways",
    "generated_comprehension_check_or_reflection_prompt_for_student",
    "generated_adaptation_for_student_profile_notes"
]

# Columns from RAG data to use as examples in the prompt
EXAMPLE_DATA_COLUMNS = [
    'pre_modeling_setup_script',
    'modeling_and_think_aloud_sequence_json',
    'post_modeling_summary_and_key_takeaways',
    'comprehension_check_or_reflection_prompt_for_student',
    'adaptation_for_student_profile_notes'
]

def format_rag_examples_for_prompt(rag_data_list: list) -> str:
    """Formats the RAG retrieved data into a string for the LLM prompt."""
    if not rag_data_list:
        return "No specific expert examples retrieved. Rely on general knowledge."
    
    example_strings = []
    for i, doc in enumerate(rag_data_list[:2]): # Use top 1-2 examples
        example_str = f"--- Expert Example {i+1} ---"
        for col in EXAMPLE_DATA_COLUMNS:
            col_value = doc.get(col, 'N/A')
            # Special handling for JSON string if it's indeed a string representation of JSON
            if col == 'modeling_and_think_aloud_sequence_json':
                try:
                    # Attempt to parse and pretty-print if it's a valid JSON string
                    parsed_json = json.loads(col_value)
                    col_value_formatted = json.dumps(parsed_json, indent=2)
                except (json.JSONDecodeError, TypeError):
                    col_value_formatted = str(col_value) # Keep as string if not valid JSON
                example_str += f"\n  {col}:\n{col_value_formatted}"
            else:
                example_str += f"\n  {col}: {str(col_value)}"
        example_strings.append(example_str + "\n-----------------------")
    return "\n\n".join(example_strings)

async def modelling_generator_node(state: AgentGraphState) -> dict:
    """
    Generates the full modelling script and associated content using an LLM,
    guided by student context and RAG-retrieved examples.
    """
    logger.info(
        f"ModellingGeneratorNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )

    # Retrieve RAG data and student context from state
    rag_document_data = state.get("modelling_document_data", [])
    example_prompt_text = state.get("example_prompt_text", "Default prompt: Describe a memorable event.") # Student's current task prompt
    student_goal_context = state.get("student_goal_context", "N/A")
    student_confidence_context = state.get("student_confidence_context", "N/A")
    teacher_initial_impression = state.get("teacher_initial_impression", "N/A")
    student_struggle_context = state.get("student_struggle_context", "N/A")
    english_comfort_level = state.get("english_comfort_level", "N/A")

    # Format RAG examples for the prompt
    expert_examples_formatted = format_rag_examples_for_prompt(rag_document_data)

    # Construct the main LLM prompt
    # This prompt needs to be carefully engineered to request all five outputs in a structured way.
    # The user's example prompt is primarily for the 'modeling_and_think_aloud_sequence_json' part.
    llm_prompt = f"""
You are 'The Structuralist', an expert AI TOEFL Tutor. Your current task is to MODEL how to approach a TOEFL task for a student.

**Student Context:**
- Task Prompt for Student: "{example_prompt_text}"
- Student's Goal: {student_goal_context}
- Student's Confidence: {student_confidence_context}
- Student's English Comfort Level: {english_comfort_level}
- Teacher's Initial Impression of Student: {teacher_initial_impression}
- Student's Primary Struggle: {student_struggle_context}

**Expert Examples & Guidance (from similar past scenarios):**
{expert_examples_formatted}

**Your Task:**
Based on the student's context and the expert examples, generate a comprehensive modelling session. You MUST provide ALL FIVE of the following components, structured in a single JSON object with the specified keys:

1.  `generated_pre_modeling_setup_script`: (String) A script to set the stage before the main modeling. Explain what you'll do and why, tailored to the student's context and the task prompt.

2.  `generated_modeling_and_think_aloud_sequence_json`: (JSON Array) The core of the modeling. Demonstrate how to tackle the student's task prompt ('{example_prompt_text}').
    - Focus on demonstrating the application of relevant principles (e.g., coherence, PREP structure, essay structure, etc., as appropriate for the task and student struggle).
    - The sequence MUST be a JSON array of objects. Each object must have a `type` field.
    - Objects can be of three types:
        1. `{{"type": "essay_text_chunk", "content": "Text the student would write/say..."}}`
        2. `{{"type": "think_aloud_text", "content": "Your explanation of your thought process..."}}`
        3. `{{"type": "ui_action_instruction", "action_type": "HIGHLIGHT_TEXT_RANGES", "parameters": {{"ranges": [{{"start": <integer>, "end": <integer>, "style_class": "ai_emphasis_point", "remark_id": "M_R<number>"}}]}}}}`
    - Interleave these types logically. For example, an `essay_text_chunk` might be followed by a `think_aloud_text` explaining it, and then a `ui_action_instruction` to highlight a part of that `essay_text_chunk`.
    - For `HIGHLIGHT_TEXT_RANGES`:
        - `parameters.ranges`: An array containing one or more range objects.
        - `range.start` and `range.end`: Character offsets for the highlight. These offsets MUST correspond to the text within the *most recent* `essay_text_chunk` you provided. Be precise.
        - `range.style_class`: Must be `"ai_emphasis_point"`.
        - `range.remark_id`: A unique ID (e.g., `"M_R1"`, `"M_R2"`) that you should also reference in the corresponding `think_aloud_text` that explains *why* this part is highlighted. This creates a link between the visual highlight and your explanation.
    Example of a sequence fragment:
    `[...,`
    `  {{"type": "essay_text_chunk", "content": "Effective topic sentences are crucial for clarity."}},`
    `  {{"type": "think_aloud_text", "content": "Notice the phrase 'crucial for clarity' (M_R1). This emphasizes the importance of this aspect."}},`
    `  {{"type": "ui_action_instruction", "action_type": "HIGHLIGHT_TEXT_RANGES", "parameters": {{"ranges": [{{"start": 29, "end": 48, "style_class": "ai_emphasis_point", "remark_id": "M_R1"}}]}}}},`
    `... ]`

3.  `generated_post_modeling_summary_and_key_takeaways`: (String) After the modeling sequence, summarize what was demonstrated and highlight the key learning points for the student.

4.  `generated_comprehension_check_or_reflection_prompt_for_student`: (String) A question or prompt to encourage the student to reflect on the modeling or to check their understanding.

5.  `generated_adaptation_for_student_profile_notes`: (String) Notes on how this specific modeling session was (or could be further) adapted for the student's profile (goal, confidence, struggle, comfort level). Explain your pedagogical choices.

**Output Format:**
Return a SINGLE JSON object with exactly these five keys: "generated_pre_modeling_setup_script", "generated_modeling_and_think_aloud_sequence_json", "generated_post_modeling_summary_and_key_takeaways", "generated_comprehension_check_or_reflection_prompt_for_student", and "generated_adaptation_for_student_profile_notes".
Ensure the value for `generated_modeling_and_think_aloud_sequence_json` is a valid JSON array as described.
"""

    # Configure Google Generative AI
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("ModellingGeneratorNode: GOOGLE_API_KEY environment variable is not set.")
        return {"error": "GOOGLE_API_KEY not set", **{key: None for key in OUTPUT_KEYS}}
    genai.configure(api_key=api_key)

    try:
        logger.info(f"ModellingGeneratorNode: Sending prompt to LLM (first 500 chars):\n{llm_prompt[:500]}...")
        model = genai.GenerativeModel(
            # Using a capable model, e.g., gemini-2.0-flash or gemini-2.0-flash-exp if available and needed for complexity
            "gemini-2.0-flash", 
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(llm_prompt) # Use async version if available
        
        logger.info(f"ModellingGeneratorNode: Received response from LLM.")
        # Ensure response.text is valid JSON before parsing
        if not response.text or not response.text.strip().startswith('{'):
            logger.error(f"ModellingGeneratorNode: LLM response is not valid JSON or is empty. Response text: {response.text}")
            # Try to get error details from the response object if available
            error_details = "Unknown LLM error or non-JSON response."
            try:
                # Accessing prompt_feedback might give more insight, or candidate.finish_reason
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    error_details = f"LLM blocked prompt: {response.prompt_feedback.block_reason_message}"
                elif response.candidates and response.candidates[0].finish_reason != 'STOP':
                    error_details = f"LLM generation finished unexpectedly: {response.candidates[0].finish_reason.name}"
            except Exception as feedback_error:
                logger.warning(f"ModellingGeneratorNode: Could not retrieve detailed feedback from LLM response: {feedback_error}")
            return {"error": error_details, **{key: None for key in OUTPUT_KEYS}}

        logger.info(f"ModellingGeneratorNode: Raw LLM response text (first 1000 chars): {response.text[:1000]}")
        response_json = json.loads(response.text)
        logger.info("ModellingGeneratorNode: LLM response parsed successfully.")

        # Log the content of each expected key from the parsed JSON
        for key_to_log in OUTPUT_KEYS:
            value_to_log = response_json.get(key_to_log)
            if isinstance(value_to_log, str):
                logger.info(f"ModellingGeneratorNode: Parsed content for '{key_to_log}': '{str(value_to_log)[:200]}...' (Length: {len(str(value_to_log))})")
            elif isinstance(value_to_log, list):
                logger.info(f"ModellingGeneratorNode: Parsed content for '{key_to_log}': List with {len(value_to_log)} items. First item (if any): {str(value_to_log[0])[:150] if value_to_log else 'N/A'}...")
            else:
                logger.info(f"ModellingGeneratorNode: Parsed content for '{key_to_log}': {type(value_to_log)} - {str(value_to_log)[:200]}")

        # Validate that all expected keys are in the response
        output_payload = {}
        all_keys_present = True
        for key in OUTPUT_KEYS:
            if key not in response_json:
                logger.warning(f"ModellingGeneratorNode: Key '{key}' missing in LLM JSON response.")
                output_payload[key] = None # Or some default value
                all_keys_present = False
            else:
                output_payload[key] = response_json[key]
        
        if not all_keys_present:
            logger.warning("ModellingGeneratorNode: Not all expected keys were present in LLM response. Some outputs might be missing.")
            output_payload["warning"] = "LLM did not return all requested output components."

        # Specific validation for generated_modeling_and_think_aloud_sequence_json
        think_aloud_seq = output_payload.get("generated_modeling_and_think_aloud_sequence_json")
        if not isinstance(think_aloud_seq, list):
            logger.warning(f"ModellingGeneratorNode: 'generated_modeling_and_think_aloud_sequence_json' is not a list as expected. Type: {type(think_aloud_seq)}")
            output_payload["generated_modeling_and_think_aloud_sequence_json"] = [] # Default to empty list
            if "warning" not in output_payload:
                 output_payload["warning"] = "'generated_modeling_and_think_aloud_sequence_json' was not a list."
            else:
                 output_payload["warning"] += " 'generated_modeling_and_think_aloud_sequence_json' was not a list."

        logger.info(f"ModellingGeneratorNode: Successfully processed LLM output. First output key content (pre_setup): {str(output_payload.get(OUTPUT_KEYS[0]))[:100]}...")
        return {"intermediate_modelling_payload": output_payload}

    except json.JSONDecodeError as e:
        logger.error(f"ModellingGeneratorNode: Error decoding JSON from LLM response: {e}. Response text: {response.text}", exc_info=True)
        return {"error": f"JSONDecodeError: {e}", **{key: None for key in OUTPUT_KEYS}}
    except Exception as e:
        logger.error(f"ModellingGeneratorNode: Error during LLM call or processing: {e}", exc_info=True)
        return {"error": f"General error in LLM generation: {str(e)}", **{key: None for key in OUTPUT_KEYS}}


# Example usage (for local testing if needed)
async def main_test():
    class MockAgentGraphState(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    # Sample RAG data (normally from modelling_RAG_document_node)
    sample_rag_data = [
        {
            'Example_Prompt_Text': 'Describe your favorite book.', 
            'Student_Goal_Context': 'Improve speaking.', 
            'Student_Confidence_Context': 'Low.', 
            'English_Comfort_Level': 'Beginner', 
            'Teacher_Initial_Impression': 'Struggles with structure.', 
            'Student_Struggle_Context': 'Difficulty starting tasks.',
            'pre_modeling_setup_script': 'Expert: We will use PREP for your favorite book.', 
            'modeling_and_think_aloud_sequence_json': '[{"type":"think_aloud", "content":"First, state your point..."}, {"type":"essay_text_chunk", "content":"My favorite book is..."}]',
            'post_modeling_summary_and_key_takeaways': 'Expert: PREP helps structure answers.',
            'comprehension_check_or_reflection_prompt_for_student': 'Expert: How can PREP help you?',
            'adaptation_for_student_profile_notes': 'Expert: PREP is good for beginners who struggle to start.'
        }
    ]

    # Student's current task context
    state1 = MockAgentGraphState({
        "user_id": "test_user_gen_1",
        "modelling_document_data": sample_rag_data,
        "example_prompt_text": "Describe what you did last weekend.",
        "student_goal_context": "Speak more fluently for daily conversation.",
        "student_confidence_context": "I get nervous and forget words.",
        "teacher_initial_impression": "Seems hesitant to speak.",
        "student_struggle_context": "Fluency and sentence construction.",
        "english_comfort_level": "Intermediate"
    })
    
    print("\n--- Test Case: Modelling Generation ---")
    # Ensure GOOGLE_API_KEY is set as an environment variable
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not set. Cannot run test.")
        return

    result = await modelling_generator_node(state1)
    
    if result.get("error"):
        print(f"Error in generation: {result.get('error')}")
    else:
        print("Successfully generated modelling content (first 100 chars of each):")
        for key in OUTPUT_KEYS:
            content = result.get(key)
            if isinstance(content, (list, dict)):
                print(f"  {key}: {json.dumps(content)[:100]}...")
            else:
                print(f"  {key}: {str(content)[:100]}...")
        if result.get("warning"):
            print(f"Warning: {result.get('warning')}")

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    # IMPORTANT: Set GOOGLE_API_KEY environment variable before running this test.
    # e.g., export GOOGLE_API_KEY='your_api_key_here'
    asyncio.run(main_test())
