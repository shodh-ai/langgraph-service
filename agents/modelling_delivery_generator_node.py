# agents/modelling_delivery_generator_node.py
import logging
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

# Configure Gemini client
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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

# THIS NODE IS NOW THE UPGRADED, PLAN-DRIVEN, AND STATE-PRESERVING GENERATOR
async def modelling_delivery_generator_node(state: AgentGraphState) -> dict:
    """
    Generates a rich, sequential script for the CURRENT step of a modeling plan
    while preserving all critical state keys.
    """
    logger.info("---Executing State-Preserving, Plan-Driven Modelling Delivery Generator---")

    try:
        # 1. Read all necessary data from the state, including the plan and context.
        plan = state.get("modelling_plan")
        current_index = state.get("current_plan_step_index", 0)
        rag_data = state.get("rag_document_data", [])
        current_context = state.get("current_context") # Must be preserved

        if not plan or current_index >= len(plan):
            raise ValueError(f"Invalid plan or step index. Plan: {plan}, Index: {current_index}")

        # 2. Get the prompt for the current step from the plan.
        current_step = plan[current_index]
        # Assuming the planner creates steps with a 'step_prompt' key.
        prompt_to_model = current_step.get("step_prompt")
        if not prompt_to_model:
            raise ValueError(f"The current plan step (index {current_index}) is missing 'step_prompt'.")

        expert_example_str = format_rag_for_prompt(rag_data)

        # 3. Construct the LLM prompt using the current step's details.
        llm_prompt = f"""
You are 'The Structuralist', an expert AI TOEFL Tutor. Your task is to generate a step-by-step script for a modelling session to demonstrate how to accomplish the following sub-task.

**Current Sub-Task:** "{prompt_to_model}"

**Your Task:**
Generate a JSON object containing a "sequence" of actions. This sequence will be executed one by one to create an interactive modelling experience. Each object in the sequence array must have a "type" and a "payload".

Here are the valid types and their payloads:

1.  **`"type": "think_aloud"`**
    -   `"payload": {{"text": "Your meta-commentary, explaining your thought process."}}`

2.  **`"type": "ai_writing_chunk"`**
    -   `"payload": {{"text_chunk": "A piece of the essay text you are writing."}}`

3.  **`"type": "highlight_writing"`**
    -   `"payload": {{"start": <int>, "end": <int>, "remark_id": "M_R1"}}`

4.  **`"type": "display_remark"`**
    -   `"payload": {{"remark_id": "M_R1", "text": "A detailed explanation for why you highlighted this part."}}`

5.  **`"type": "self_correction"`**
    -   `"payload": {{"start": <int>, "end": <int>, "new_text": "a better phrase"}}`

**Example of a sequence:**
{expert_example_str}

**Instructions:**
- Focus ONLY on the current sub-task: "{prompt_to_model}".
- Intersperse `think_aloud` steps to explain your reasoning.
- Use `ai_writing_chunk` to build the relevant part of the essay.
- Use `highlight_writing` and `display_remark` to draw attention to key parts.
- Ensure all character offsets for `highlight_writing` and `self_correction` are accurate relative to the full essay text composed so far.

Generate the JSON object with the "sequence" array now.
"""
        # 4. Call the LLM to generate the action sequence.
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(llm_prompt)
        response_json = json.loads(response.text)

        logger.info(f"Modelling generator created action sequence for step {current_index + 1}.")

        # 5. Return the new payload AND preserve all other critical state.
        return {
            "intermediate_modelling_payload": response_json,
            "modelling_plan": plan, # Preserve plan
            "current_plan_step_index": current_index, # Preserve index
            "current_context": current_context, # Preserve context
            "rag_document_data": None  # Clear RAG data after use
        }

    except Exception as e:
        logger.error(f"ModellingDeliveryGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": f"Failed to generate modelling script: {e}", "route_to_error_handler": True}
