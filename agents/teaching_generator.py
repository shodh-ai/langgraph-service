# new graph/teaching_generator_node.py

import logging
import json
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def teaching_generator_node(state: AgentGraphState) -> dict:
    """
    Generates the raw, structured content for a teaching segment using an LLM,
    guided by student context and RAG-retrieved examples.
    """
    logger.info("---Executing Teaching Generator Node (Refactored)---")
    try:
        rag_data = state.get("rag_document_data")
        if not rag_data:
            raise ValueError("Teaching generator received no RAG documents.")

        # Define the complex JSON example outside the f-string for clarity
        json_output_example = """
{
    "opening_hook": "A short, engaging sentence to start this lesson segment.",
    "core_explanation": "A clear, concise explanation of the main concept, tailored to the student's context.",
    "key_examples": "One or two clear examples that illustrate the core explanation.",
    "comprehension_check": "A single, open-ended question to check for understanding.",
    "visual_aid_instructions": "A description of a visual aid if highly effective, otherwise null."
}
"""

        llm_prompt = f"""
You are an expert AI TOEFL Tutor with the persona of 'The Structuralist'.

**Student & Lesson Context:**
- Learning Objective: {state.get('Learning_Objective_Focus', 'Not specified')}
- Student's Proficiency: {state.get('STUDENT_PROFICIENCY', 'Not specified')}
- Student's Affective State: {state.get('STUDENT_AFFECTIVE_STATE', 'Neutral')}

**Expert Examples from Knowledge Base (for inspiration on structure and tone):**
{json.dumps(rag_data, indent=2)}

**Your Task:**
Generate the fundamental building blocks for this teaching moment. DO NOT write the final script. Instead, provide the raw content for each part.

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
        logger.info(f"LLM Response for teaching: {response_json}")

        # --- RETURN THE STANDARDIZED PAYLOAD ---
        return {"intermediate_teaching_payload": response_json}

    except Exception as e:
        logger.error(f"TeachingGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": str(e), "route_to_error_handler": True}