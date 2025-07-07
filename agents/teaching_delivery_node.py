# agents/teaching_delivery_node.py (The New, Definitive Version)

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

async def teaching_delivery_generator_node(state: AgentGraphState) -> dict:
    """
    Executes a SINGLE step from the plan, generating a rich payload with
    both conversational text (for an interactive sequence) AND a visual aid suggestion.
    """
    logger.info("--- Executing Definitive Teaching Delivery Generator Node ---")
    try:
        # --- Get context from state ---
        rag_documents = state.get("rag_document_data")
        plan = state.get("pedagogical_plan")
        current_index = state.get("current_plan_step_index", 0)
        current_step = plan[current_index]
        step_focus = current_step.get("focus")
        active_persona = state.get("active_persona", "The Structuralist")
        student_profile = state.get("student_memory_context", {})
        
        rag_context_examples = json.dumps(rag_documents, indent=2) if rag_documents else "[]"

        # --- Define the comprehensive JSON output structure we want ---
        json_output_example = """
{
  "core_explanation": "A clear and concise explanation of the key concept for this lesson step.",
  "key_examples": "One or two clear and relevant examples illustrating the core explanation.",
  "sequence": [
    {"type": "tts", "content": "The opening sentence of the explanation."},
    {"type": "tts", "content": "The main point or example."},
    {"type": "listen", "expected_intent": "CONFIRMATION", "prompt_if_silent": "Does that make sense so far?", "timeout_ms": 5000}
  ],
  "visual_aid_suggestion": {
    "lessonTitle": "Visual Aid Title (e.g., The Pythagorean Theorem)",
    "canvasDimensions": { "width": 1000, "height": 750 },
    "steps": [
      {
        "command": "write", "id": "title-text",
        "payload": { "text": "...", "position": {"x": 50, "y": 50}, "varaOptions": {...}}
      },
      {
        "command": "drawShape", "id": "some-shape",
        "payload": { "shapeType": "line", "points": [...], "roughOptions": {...}}
      }
    ]
  },
  "comprehension_check_question": "A short, open-ended question to check the student's understanding."
}
"""
        # --- Construct the merged, powerful prompt ---
        llm_prompt = f"""
        You are an expert AI TOEFL Tutor with the persona of '{active_persona}'.
        You are delivering one specific part of a larger lesson plan.

        **Student Profile:** {json.dumps(student_profile, indent=2)}
        **Current Lesson Step to Deliver (Focus):** "{step_focus}"

        **Expert Examples from Knowledge Base:**
        {rag_context_examples}

        **Your Task:**
        Generate a comprehensive teaching payload for the current step: **"{step_focus}"**.
        Your output MUST be a single JSON object with the following keys:
        1.  `core_explanation`: The main text explanation.
        2.  `key_examples`: Text-based examples.
        3.  `sequence`: An interactive sequence for the explanation, ending with a 'listen' action. Use the core_explanation and key_examples to create the content for the 'tts' steps.
        4.  `visual_aid_suggestion`: A JSON object detailing commands for a visual aid. If no visual is needed for this step, provide an empty JSON object: {{}}.
        5.  `comprehension_check_question`: A question to ask the student.

        Return ONLY a SINGLE JSON object with the exact structure shown in this example:
        {json_output_example}
        """

        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(llm_prompt)
        response_json = json.loads(response.text)
        
        logger.info(f"Generated comprehensive payload for step {current_index + 1}.")

        # This payload now contains everything the formatter might need.
        return {
            "intermediate_teaching_payload": response_json,
            "rag_document_data": None
        }

    except Exception as e:
        logger.error(f"TeachingDeliveryGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": f"Failed to deliver lesson step: {e}", "route_to_error_handler": True}