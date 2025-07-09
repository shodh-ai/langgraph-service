# agents/modelling_delivery_node.py

import logging
import json
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

# Configure Gemini client
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def modelling_delivery_generator_node(state: AgentGraphState) -> dict:
    """
    Generates the rich modelling payload for the current step (e.g., feedback or a model answer)
    and preserves the session state for the formatter node.
    """
    logger.info("--- Executing Modelling Delivery Generator ---")
    try:
        # Get all the state we need to use AND preserve
        rag_documents = state.get("rag_document_data")
        plan = state.get("pedagogical_plan")
        current_index = state.get("current_plan_step_index", 0)
        current_step = plan[current_index]
        step_type = current_step.get("step_type")
        step_focus = current_step.get("focus")
        student_submission = state.get("current_context", {}).get('student_task_submission')

        rag_context_examples = json.dumps(rag_documents, indent=2) if rag_documents else "[]"

        # --- Define the JSON output structure --- 
        json_output_example = """
{
  "text_for_tts": "A spoken summary of the content being delivered.",
  "ui_actions": [
    {"component": "display_feedback", "text": "Detailed written feedback for the student."},
    {"component": "display_model_answer", "text": "A complete model answer for comparison."}
  ]
}
"""

        # --- Construct the prompt based on the step type ---
        llm_prompt = f"""
        You are an expert AI TOEFL Tutor.
        You are delivering one step of a modelling session plan.

        **Student's Original Submission:** {student_submission}
        **Current Step Type:** "{step_type}"
        **Focus for this step:** "{step_focus}"

        **Expert Examples from Knowledge Base:**
        {rag_context_examples}

        **Your Task:**
        Generate a JSON payload for this specific step: **\"{step_focus}\"**.
        - If the step_type is 'DELIVER_FEEDBACK', focus on providing constructive feedback.
        - If the step_type is 'SHOW_MODEL_ANSWER', focus on presenting a high-quality model answer.
        - The `text_for_tts` should be a short, spoken introduction to what you are presenting.
        - The `ui_actions` should contain the detailed text for the UI.

        Return ONLY a SINGLE JSON object with the exact structure shown in this example:
        {json_output_example}
        """

        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(llm_prompt)
        response_json = json.loads(response.text)

        logger.info(f"Generated modelling payload for step {current_index + 1} ('{step_type}').")

        # Return the new payload, clearing the RAG data for this turn.
        return {
            "intermediate_modelling_payload": response_json,
            "rag_document_data": None, 
        }

    except Exception as e:
        logger.error(f"ModellingDeliveryGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": f"Failed to deliver modelling step: {e}"}
