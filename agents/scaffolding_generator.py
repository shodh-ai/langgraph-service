# In agents/scaffolding_generator.py
import json
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def scaffolding_generator_node(state: AgentGraphState) -> dict:
    logger.info("---Executing Scaffolding Generator Node (Refactored)---")
    try:
        # Get the necessary data from the state, using clear variable names
        scaffolding_strategies = state.get("rag_document_data")
        if not scaffolding_strategies:
            raise ValueError("Scaffolding generator received no RAG documents.")

        primary_struggle = state.get("Specific_Struggle_Point", "Not specified")
        learning_objective_id = state.get("Learning_Objective_Task", "Not specified")
        
        # Placeholder for user_data; this can be expanded later
        user_data = {
            "proficiency": state.get("STUDENT_PROFICIENCY", "N/A"),
            "affective_state": state.get("STUDENT_AFFECTIVE_STATE", "N/A")
        }

        # Define the complex JSON example outside the f-string
        json_output_example = """
{
    "text_for_tts": "The spoken introduction to the scaffolding (friendly, encouraging, brief)",
    "ui_components": [
        {"type": "text", "content": "Any explanatory text to display"},
        {"type": "scaffold", "scaffold_type": "template", "content": {}},
        {"type": "guidance", "content": "Instructions on using the scaffold"}
    ]
}
"""
        prompt = f"""
You are 'The Encouraging Nurturer' AI Tutor.

Student Profile: {user_data}
Primary Struggle: {primary_struggle}
Learning Objective ID: {learning_objective_id}
Expert Examples: {json.dumps(scaffolding_strategies, indent=2)}

Your Task:
Create a complete, personalized scaffolding response.

Return a SINGLE JSON object with this exact structure:
{json_output_example}
"""
        # Call the LLM
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(prompt)
        response_json = json.loads(response.text)
        logger.info(f"LLM Response for scaffolding: {response_json}")

        # --- RETURN THE STANDARDIZED PAYLOAD ---
        return {"intermediate_scaffolding_payload": response_json}

    except Exception as e:
        logger.error(f"ScaffoldingGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": str(e), "route_to_error_handler": True}