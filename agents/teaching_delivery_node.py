# agents/teaching_delivery_node.py (The Final, State-Preserving, Corrected Version)

import logging
import json
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def teaching_delivery_generator_node(state: AgentGraphState) -> dict:
    """
    Generates a rich teaching payload for the current step, ensuring it
    receives the plan from the previous state and passes all critical
    state keys forward to the next node.
    """
    logger.info("--- Executing State-Preserving Assertive Delivery Generator ---")
    
    try:
        # --- Retrieve all critical state keys that must be preserved ---
        plan = state.get("pedagogical_plan")
        current_index = state.get("current_plan_step_index", 0)
        rag_documents = state.get("rag_document_data", [])
        active_persona = state.get("active_persona", "The Structuralist")
        student_profile = state.get("student_memory_context", {})
        
        # --- THIS IS THE CRITICAL FIX: Check for the plan's existence ---
        if not plan or not isinstance(plan, list) or current_index >= len(plan):
            error_message = f"TeachingDeliveryGenerator: Invalid or missing pedagogical plan. Index: {current_index}, Plan: {plan}"
            logger.error(error_message)
            # Return an empty payload but preserve the rest of the state
            return {
                "error_message": error_message,
                "intermediate_teaching_payload": {},
                **state  # Pass the entire state forward on error to aid debugging
            }
        
        # --- Proceed with generation now that we know the plan is valid ---
        current_step = plan[current_index]
        step_focus = current_step.get("focus")
        rag_context_examples = json.dumps(rag_documents, indent=2)

        json_output_example = """
        {
  "core_explanation": "A clear, text-based explanation of the concept.",
  "key_examples": "One or two clear, text-based examples.",
  "comprehension_check_question": "An open-ended question for the student.",
  "visual_aid_suggestion": {
    "lessonTitle": "Visual Aid Title",
    "canvasDimensions": { "width": 800, "height": 600 },
    "steps": [
      {
        "command": "write",
        "id": "example-title-1",
        "payload": {
          "text": "This is the Title",
          "position": {"x": 50, "y": 50},
          "varaOptions": { "fontSize": 36, "color": "#000000", "duration": 1500 }
        }
      },
      {
        "command": "drawShape",
        "id": "divider-line",
        "payload": {
            "shapeType": "line",
            "points": [
                { "x": 50, "y": 120 },
                { "x": 750, "y": 120 }
            ],
            "isRough": true,
            "roughOptions": {
                "stroke": "black",
                "strokeWidth": 2,
                "roughness": 1.5
            }
        }
      }
    ]
  },
  "sequence": [
    { "type": "tts", "content": "This is the main explanation." },
    { "type": "tts", "content": "Now, look at the visual I've prepared for you." },
    { "type": "listen" }
  ]
}
        """

        llm_prompt = f"""
        You are an expert AI TOEFL Tutor. Your task is to generate a teaching payload for the lesson step: **"{step_focus}"**.
        You MUST return a single valid JSON object.
CRITICAL INSTRUCTIONS:
1.  The array of visual commands MUST be named `steps`.
2.  For EVERY command object in the `steps` array, all of its data MUST be nested inside a single object named `payload`.
3.  Specifically for a "drawShape" command, the `payload` object MUST contain a `shapeType` string, a `points` array of objects, `isRough` boolean, and `roughOptions` object. DO NOT forget the `points` array.

        {json_output_example}
        """

        model = genai.GenerativeModel("gemini-2.0-flash", generation_config=GenerationConfig(response_mime_type="application/json"))
        response = await model.generate_content_async(llm_prompt)
        response_json = json.loads(response.text)

        logger.info(f"Generated explicit flat-structure payload for step {current_index + 1}.")

        # --- Return the new payload AND preserve all other state keys ---
        return {
            "intermediate_teaching_payload": response_json,
            "rag_document_data": None,  # Clear used data
            "pedagogical_plan": plan,
            "current_plan_step_index": current_index,
            "active_persona": active_persona,
            "student_memory_context": student_profile,
            # Pass along any other keys that future nodes might need
            "lesson_id": state.get("lesson_id"),
            "Learning_Objective_Focus": state.get("Learning_Objective_Focus"),
            "STUDENT_PROFICIENCY": state.get("STUDENT_PROFICIENCY"),
            "STUDENT_AFFECTIVE_STATE": state.get("STUDENT_AFFECTIVE_STATE"),
        }

    except Exception as e:
        logger.error(f"ExplicitDeliveryGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {
            "error_message": f"Failed to deliver lesson step: {e}",
            "intermediate_teaching_payload": {},
            **state
        }

