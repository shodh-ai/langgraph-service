# graph/feedback_generator_node.py
import json
import logging
from state import AgentGraphState
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)

async def feedback_generator_node(state: AgentGraphState) -> dict:
    logger.info("---Executing Feedback Generator Node (Refactored)---")
    try:
        rag_data = state.get("rag_document_data")
        if not rag_data:
            raise ValueError("Feedback generator received no RAG documents.")

        diagnosed_error = state.get('diagnosed_error_type', 'the issue')

        # Define the complex JSON example outside the f-string for clarity
        json_output_example = """
{
    "acknowledgement": "A brief, encouraging acknowledgement of the student's effort.",
    "explanation": "A clear, concise explanation of the error without being discouraging.",
    "corrected_example": "A corrected version of what the student was trying to do.",
    "follow_up_task": "A short, actionable follow-up task to practice the skill."
}
"""

        llm_prompt = f"""
You are 'The Structuralist', an expert teacher giving feedback.

A student made this error: {diagnosed_error}
Based on these expert examples of giving feedback: {json.dumps(rag_data, indent=2)}

Your Task:
Generate a feedback response that:
1.  Acknowledges the student's effort.
2.  Clearly explains the error without being discouraging.
3.  Provides a corrected example.
4.  Gives a short, actionable follow-up task.

Return a SINGLE JSON object with this exact structure:
{json_output_example}
"""
        
        # --- Call the LLM ---
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(llm_prompt)
        response_json = json.loads(response.text)
        logger.info(f"LLM Response for feedback: {response_json}")

        # --- RETURN THE STANDARDIZED PAYLOAD ---
        return {"intermediate_feedback_payload": response_json}

    except Exception as e:
        logger.error(f"FeedbackGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": str(e), "route_to_error_handler": True}