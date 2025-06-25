# langgraph-service/agents/cowriting_generator.py
import logging
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def cowriting_generator_node(state: AgentGraphState) -> dict:
    """
    Acts as a creative co-writer, providing students with layered content suggestions.
    """
    logger.info("---Executing Co-Writing Generator Node (Layered Content Version)---")
    try:
        student_writing = state.get("transcript", "")
        if not student_writing:
            logger.warning("CoWritingNode: 'transcript' not found in state.")
            return {
                "intermediate_cowriting_payload": {
                    "error": True,
                    "error_message": "It looks like there's nothing to work with yet. Try writing a sentence or two first!"
                }
            }

        user_data = {
            "proficiency": state.get("student_proficiency", "N/A"),
            "affective_state": state.get("student_affective_state", "N/A")
        }

        json_output_example = """
{
    "main_explanation": "To keep the momentum going, we could either add more detail to the last point, or start transitioning to the next main idea. Both are great options!",
    "simplified_explanation": "Let's either add more detail or move on to the next topic.",
    "clarifications": {
        "Which one is better?": "It depends on your goal! Adding detail makes your current point stronger, while moving on covers more ground.",
        "Can you give me a different idea?": "Of course! We could also try to add a personal story here to make it more engaging."
    },
    "sequence": [
        {"type": "tts", "content": "That's a fantastic start! I have a couple of ideas for what we could do next."},
        {"type": "tts", "content": "One option is to add a specific example to support your last sentence. Another idea is to start a new paragraph that introduces your next point."},
        {"type": "tts", "content": "What feels like the right next step for you?"},
        {"type": "listen", "expected_intent": "user_chooses_cowriting_suggestion", "prompt_if_silent": "Let me know which direction you'd like to go!"}
    ]
}
"""

        prompt = f"""
        You are 'The Creative Collaborator', an AI partner helping a student write.

        **Student Context:**
        - User Profile: {user_data}
        - Their writing so far: "{student_writing}"

        **Your Task:**
        Generate a "Layered Content" payload to offer helpful suggestions for continuing the writing. The goal is to provide creative options and let the student choose.

        The payload must be a single JSON object with exactly these four keys:
        1.  `main_explanation`: A clear, concise summary of the different directions the student could take their writing next.
        2.  `simplified_explanation`: A very simple rephrasing of the main explanation.
        3.  `clarifications`: A JSON object containing 2-3 pre-written answers to likely, simple clarification questions about your suggestions.
        4.  `sequence`: An interactive sequence of `tts` and `listen` steps. Introduce the suggestions conversationally and then prompt the student to choose or share their own idea. The final step MUST be a `listen` step.

        **Example JSON Output:**
        {json_output_example}

        **Instructions:**
        - Your tone should be encouraging and collaborative.
        - The `sequence` should present at least two clear, distinct options.
        - The `expected_intent` in the `listen` step should be `user_chooses_cowriting_suggestion`.

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
        logger.info(f"LLM Response for co-writing: {response_json}")

        return {"intermediate_cowriting_payload": response_json}

    except Exception as e:
        logger.error(f"CoWritingGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"intermediate_cowriting_payload": {"error": True, "error_message": str(e)}}
