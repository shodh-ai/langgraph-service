# graph/feedback_generator_node.py
import json
import logging
from state import AgentGraphState
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)

async def feedback_generator_node(state: AgentGraphState) -> dict:
    logger.info("---Executing Feedback Generator Node (Layered Content Version)---")
    try:
        rag_data = state.get("rag_document_data")
        if not rag_data:
            raise ValueError("Feedback generator received no RAG documents.")

        diagnosed_error = state.get('diagnosed_error', 'the issue')
        user_transcript = state.get("transcript", "")
        user_data = {
            "proficiency": state.get("student_proficiency", "N/A"),
            "affective_state": state.get("student_affective_state", "N/A")
        }

        json_output_example = """
{
    "main_explanation": "A clear, technically accurate explanation of why the student's attempt was incorrect, focusing on the core misconception.",
    "simplified_explanation": "A very simple analogy for the error, like 'It's like trying to put your shoes on before your socks.'",
    "clarifications": {
        "Why is my way wrong?": "A pre-written, encouraging answer to this common question.",
        "Can you show me the first step?": "A pre-written answer that gives a small hint."
    },
    "sequence": [
        {"type": "tts", "content": "That's a great attempt! You're very close to the right idea."},
        {"type": "tts", "content": "Let's look at one part specifically. Here is a corrected version of that line..."},
        {"type": "tts", "content": "Do you see the key difference? Try applying that correction and running it again."},
        {"type": "listen", "expected_intent": "student_applies_correction", "prompt_if_silent": "Give it a shot, I'm here to help!"}
    ]
}
"""

        llm_prompt = f"""
You are 'The Structuralist', an expert AI Tutor specializing in delivering clear, encouraging, and constructive feedback.

**Student Context:**
- User Profile: {user_data}
- Their diagnosed error: {diagnosed_error}
- What they just said/did: "{user_transcript}"
- Expert Feedback Strategies (for inspiration): {json.dumps(rag_data, indent=2)}

**Your Task:**
Generate a "Layered Content" payload to provide corrective feedback. The goal is to help the student understand their mistake and guide them to self-correct without just giving them the answer.

The payload must be a single JSON object with exactly these four keys:
1.  `main_explanation`: A clear, concise explanation of the error. Focus on the 'why' behind the error.
2.  `simplified_explanation`: A simple analogy or rephrasing of the main explanation.
3.  `clarifications`: A JSON object containing 2-3 pre-written answers to likely, simple clarification questions about your feedback.
4.  `sequence`: An interactive sequence of `tts` and `listen` steps. Acknowledge their effort, explain the issue conversationally, provide a corrected example or hint, and then prompt them to try again. The final step MUST be a `listen` step.

**Example JSON Output:**
{json_output_example}

**Instructions:**
- Your tone should be encouraging and supportive, not critical.
- The `sequence` should guide the user to fix the error themselves.
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
        response = await model.generate_content_async(llm_prompt)
        response_json = json.loads(response.text)
        logger.info(f"LLM Response for feedback: {response_json}")

        return {"intermediate_feedback_payload": response_json}

    except Exception as e:
        logger.error(f"FeedbackGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"intermediate_feedback_payload": {"error": True, "error_message": str(e)}}