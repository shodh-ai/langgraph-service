# agents/modelling_generator.py
import logging
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

def format_rag_for_prompt(rag_data: list) -> str:
    """Helper to format RAG results for the prompt."""
    if not rag_data:
        return "No expert examples were retrieved. Rely on general principles."
    # We'll just use the first, most relevant example to keep the prompt clean
    # and give the LLM a single, strong pattern to follow.
    try:
        example = rag_data[0]
        # We only need the 'modeling_and_think_aloud_sequence_json' as the example
        sequence_example_str = example.get('modeling_and_think_aloud_sequence_json', '{}')
        # Pretty-print the JSON so the LLM can easily read the structure
        sequence_example_json = json.dumps(json.loads(sequence_example_str), indent=2)
        return f"Follow this example structure for the sequence:\n{sequence_example_json}"
    except Exception as e:
        logger.warning(f"Could not format RAG example for prompt: {e}")
        return "No valid expert examples were retrieved."

async def modelling_generator_node(state: AgentGraphState) -> dict:
    """
    Generates a rich, sequential script for a modelling session.
    """
    logger.info("---Executing Modelling Generator Node (Advanced UI Version)---")
    
    try:
        prompt_to_model = state.get("example_prompt_text")
        if not prompt_to_model:
            raise ValueError("'example_prompt_text' is missing from the state.")
            
        rag_data = state.get("rag_document_data", [])
        expert_example_str = format_rag_for_prompt(rag_data)

        # This is a highly detailed prompt that instructs the LLM to produce
        # a script with all the actions you need.
        llm_prompt = f"""
You are 'The Structuralist', an expert AI TOEFL Tutor. Your task is to generate a step-by-step script for a modelling session to demonstrate how to answer the following prompt.

**Student's Task Prompt:** "{prompt_to_model}"

**Your Task:**
Generate a JSON object containing a "sequence" of actions. This sequence will be executed one by one to create an interactive modelling experience. Each object in the sequence array must have a "type" and a "payload".

Here are the valid types and their payloads:

1.  **`"type": "update_prompt_display"`**
    -   `"payload": {{"text": "The prompt you are modeling."}}`
    -   Use this ONCE at the very beginning.

2.  **`"type": "think_aloud"`**
    -   `"payload": {{"text": "Your meta-commentary, explaining your thought process."}}`
    -   Use this to explain WHAT you are doing and WHY.

3.  **`"type": "ai_writing_chunk"`**
    -   `"payload": {{"text_chunk": "A piece of the essay text you are writing."}}`
    -   Use this to progressively "type out" the model answer.

4.  **`"type": "highlight_writing"`**
    -   `"payload": {{"start": <int>, "end": <int>, "remark_id": "M_R1"}}`
    -   Use this to highlight a section of the text you just wrote. `start` and `end` are character offsets relative to the full essay text so far.

5.  **`"type": "display_remark"`**
    -   `"payload": {{"remark_id": "M_R1", "text": "A detailed explanation for why you highlighted this part."}}`
    -   Use this immediately after a `highlight_writing` action. The `remark_id` MUST match.

6.  **`"type": "self_correction"`**
    -   `"payload": {{"start": <int>, "end": <int>, "new_text": "a better phrase"}}`
    -   Use this to demonstrate editing and improving your own writing.

**Example of a sequence:**
{expert_example_str}

**Instructions:**
- Start with `update_prompt_display`.
- Intersperse `think_aloud` steps to explain your reasoning.
- Use `ai_writing_chunk` multiple times to build the essay piece by piece.
- Use `highlight_writing` and `display_remark` to draw attention to key parts.
- Optionally, include a `self_correction` to show the editing process.
- Ensure all character offsets for `highlight_writing` and `self_correction` are accurate.

Generate the JSON object with the "sequence" array now.
"""

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("GOOGLE_API_KEY is not set.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash", # A more advanced model is needed for this complex task
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(llm_prompt)
        response_json = json.loads(response.text)
        
        logger.info("Modelling generator successfully created a rich action sequence.")
        
        return {"intermediate_modelling_payload": response_json}

    except Exception as e:
        logger.error(f"ModellingGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"intermediate_modelling_payload": {"error": True, "error_message": str(e)}}