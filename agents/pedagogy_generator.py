# agents/pedagogy_generator.py

import logging
import json
import os
import httpx
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

# These are the keys for the student's context, which are still needed for the prompt.
query_columns = [
    "Goal",
    "Feeling",
    "Confidence",
    "Estimated Overall English Comfort Level",
    "Initial Impression",
    "Speaking Strengths",
    "Fluency",
    "Grammar",
    "Vocabulary",
]

async def pedagogy_generator_node(state: AgentGraphState) -> dict:
    """
    Generates a personalized pedagogy plan.
    This node is now data-agnostic and expects RAG data from the state.
    """
    logger.info("---Executing Pedagogy Generator Node (Fully Refactored)---")
    try:
        # 1. Get RAG data from state (consistent with other generators)
        rag_data = state.get("rag_document_data")
        if not rag_data:
            raise ValueError("Pedagogy generator received no RAG documents from state.")

        # 2. Get student context from state for the prompt
        student_context = {col: state.get(col, 'Not specified') for col in query_columns}

        # 3. Define the JSON output structure
        json_output_example = """
{
    "task_suggestion_tts": "A friendly, user-facing message that introduces the learning plan.",
    "reasoning": "The overall reasoning behind the pedagogy, explaining the choices made.",
    "steps": [
        {
            "type": "Module type (Teaching, Modelling, Scaffolding, Cowriting, or Test)",
            "task": "The task type (speaking or writing)",
            "topic": "The specific topic for the module.",
            "level": "The difficulty level (Basic, Intermediate, or Advanced)"
        }
    ]
}
"""
        # 4. Construct the prompt
        prompt = f"""
        You are an expert pedagogue designing an English learning plan for a TOEFL student.

        **Student Context:**
        {json.dumps(student_context, indent=2)}

        **Examples from Similar Cases (for inspiration on structure and tone):**
        {json.dumps(rag_data, indent=2)}

        **Your Task:**
        Design a personalized pedagogy for this student. Return a SINGLE JSON object with the exact structure below:
        {json_output_example}
        """

        # 5. Call the LLM
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
        logger.info(f"LLM Response for pedagogy: {response_json}")

        # 6. Save to backend (this logic remains)
        user_id = state.get("user_id")
        auth_token = state.get("user_token")

        if user_id and auth_token:
            pronity_backend_url = os.getenv("PRONITY_BACKEND_URL", "http://localhost:8000")
            api_endpoint = f"{pronity_backend_url}/user/save-flow"
            payload = {
                "analysis": response_json.get("reasoning", ""),
                "flowElements": response_json.get("steps", []),
            }
            headers = {
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient() as client:
                logger.info(f"Sending pedagogy flow to Pronity backend for user {user_id}")
                api_response = await client.post(api_endpoint, json=payload, headers=headers)
                api_response.raise_for_status()
                logger.info(f"Successfully saved pedagogy flow for user {user_id}.")
        else:
            logger.warning("User ID or Auth Token not found. Skipping pedagogy save.")

        # 7. Return the standardized payload
        return {"intermediate_pedagogy_payload": response_json}

    except Exception as e:
        logger.error(f"PedagogyGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": str(e), "route_to_error_handler": True}
