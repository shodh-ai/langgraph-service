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
    Generates a layered teaching payload including a main explanation, simplified version,
    pre-written clarifications, and an interactive TTS/listen sequence.
    """
    logger.info("---Executing Teaching Generator Node (Layered Content Refactor)---")
    try:
        rag_data = state.get("rag_document_data")
        if not rag_data:
            raise ValueError("Teaching generator received no RAG documents.")

        # Define the new, richer JSON structure example for the LLM
        json_output_example = """
{
  "main_explanation": "In English grammar, subject-verb agreement is the correspondence of a verb with its subject in person and number. In simple terms, this means if you have a singular subject, you need a singular verb.",
  "simplified_explanation": "Think of it like a puzzle piece. A singular subject (like 'he' or 'the cat') needs a singular verb (like 'is' or 'sits'). A plural subject (like 'they' or 'the cats') needs a plural verb (like 'are' or 'sit'). They have to match!",
  "clarifications": {
    "what_is_a_subject": "The subject is the 'who' or 'what' that performs the action or is described in a sentence. For example, in 'The dog barks,' the subject is 'The dog'.",
    "what_is_a_verb": "The verb is the action word or state of being in a sentence. In 'The dog barks,' the verb is 'barks'.",
    "example_of_agreement": "Correct: 'She writes every day.' Incorrect: 'She write every day.' because 'She' is singular and needs the singular verb 'writes'."
  },
  "sequence": [
    {"type": "tts", "content": "Okay, let's talk about a core concept: Subject-Verb Agreement."},
    {"type": "tts", "content": "In simple terms, this means if you have a singular subject, you need a singular verb."},
    {"type": "listen", "expected_intent": "CONFIRMATION", "prompt_if_silent": "Does that basic idea make sense?", "timeout_ms": 4000}
  ]
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
Generate a "Layered Content" payload for a teaching module. This payload must be a single JSON object containing four keys:
1.  `main_explanation`: A clear, concise, and technically accurate explanation of the concept.
2.  `simplified_explanation`: A very simple analogy or rephrasing for students who might be struggling.
3.  `clarifications`: A JSON object containing pre-written answers to 2-3 common, simple clarification questions a student might ask. The keys should be short, snake_case identifiers for the questions.
4.  `sequence`: A JSON array for an interactive presentation. Each object must have a "type" ('tts' or 'listen'). Use this to present the core idea interactively.

Return a SINGLE JSON object with the exact structure shown in this example:
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
        logger.info(f"LLM Response for layered teaching content: {response_json}")

        # --- RETURN THE STANDARDIZED PAYLOAD ---
        return {"intermediate_teaching_payload": response_json}

    except Exception as e:
        logger.error(f"TeachingGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": str(e), "route_to_error_handler": True}