# In agents/scaffolding_generator.py
import json
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def scaffolding_generator_node(state: AgentGraphState) -> dict:
    logger.info("---Executing Scaffolding Generator Node (Layered Content Version)---")
    try:
        scaffolding_strategies = state.get("rag_document_data")
        if not scaffolding_strategies:
            raise ValueError("Scaffolding generator received no RAG documents.")

        primary_struggle = state.get("specific_struggle_point", "Not specified")
        learning_objective_id = state.get("learning_objective_task", "Not specified")
        user_transcript = state.get("transcript", "")
        
        user_data = {
            "proficiency": state.get("student_proficiency", "N/A"),
            "affective_state": state.get("student_affective_state", "N/A")
        }

        json_output_example = """
{
    "main_explanation": "A clear, technically accurate explanation of the concept needed to overcome the struggle.",
    "simplified_explanation": "A very simple analogy or rephrasing of the main explanation for students who are still confused.",
    "clarifications": {
        "Why did we do X?": "A pre-written answer to a common, simple clarification question.",
        "What does Y mean?": "Another pre-written answer."
    },
    "sequence": [
        {"type": "tts", "content": "First, let's try a hint to get you started."},
        {"type": "tts", "content": "Think about the first step we took in the last example. How might that apply here?"},
        {"type": "listen", "expected_intent": "student_attempts_scaffolded_task", "prompt_if_silent": "Feel free to ask for another hint!"}
    ]
}
"""
        prompt = f"""
You are 'The Encouraging Nurturer' AI Tutor, an expert in providing helpful hints and scaffolding.

**Student Context:**
- User Profile: {user_data}
- Learning Objective ID: {learning_objective_id}
- Their specific struggle: {primary_struggle}
- What they just said: "{user_transcript}"
- Expert Scaffolding Strategies (for inspiration): {json.dumps(scaffolding_strategies, indent=2)}

**Your Task:**
Generate a "Layered Content" payload to provide a helpful scaffold for the student. This is NOT a full lesson, but a targeted hint or tool to help them overcome their specific struggle.

The payload must be a single JSON object with exactly these four keys:
1.  `main_explanation`: A clear, concise explanation of the underlying concept they are missing. (Max 1-2 sentences)
2.  `simplified_explanation`: A simple analogy or rephrasing of the main explanation.
3.  `clarifications`: A JSON object containing 2-3 pre-written answers to likely, simple clarification questions about your explanation.
4.  `sequence`: An interactive sequence of `tts` and `listen` steps. Start with a gentle hint, then prompt the user to try again. The final step MUST be a `listen` step.

**Example JSON Output:**
{json_output_example}

**Instructions:**
- Be encouraging and supportive.
- The `sequence` should guide the user, not give them the direct answer.
- The `expected_intent` in the `listen` step should be specific to the action you want the user to take.

Generate the JSON payload now.
"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(prompt)
        response_json = json.loads(response.text)
        logger.info(f"LLM Response for scaffolding: {response_json}")

        return {"intermediate_scaffolding_payload": response_json}

    except Exception as e:
        logger.error(f"ScaffoldingGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"intermediate_scaffolding_payload": {"error": True, "error_message": str(e)}}