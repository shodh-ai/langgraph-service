import json
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)


async def error_generator_node(state: AgentGraphState) -> dict:
    logger.info(
        f"ErrorGeneratorNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    transcript = state.get("transcript", "")
    question_stage = state.get("question_stage", "")
    user_data = state.get("user_data", {})
    if user_data == {}:
        user_level = "No level, so concider beginner"
    else:
        user_level = user_data.get("level", "Beginner")
    logger.info(f"ErrorGeneratorNode: Transcript: {transcript}")
    logger.info(f"ErrorGeneratorNode: Question Stage: {question_stage}")
    logger.info(f"ErrorGeneratorNode: User Level: {user_level}")

    error_prompt = f"""
    You are an NLU assistant for the Rox AI Tutor on its writing evaluation page.
    The student submitted the following response:
    '{transcript}'

    Your task is to analyze this response and determine the **primary type of English learning error** it contains. You must choose **only one** from the following list of error categories:

    - "Fluency"
    - "Pronunciation (specific phonemes)"
    - "Grammar (tense, S-V agreement, articles)"
    - "Vocabulary (range, academic)"
    - "Coherence"
    - "Cohesion"
    - "Task Fulfillment"
    - "Essay Structure (thesis, topic sentences)"
    - "Paraphrasing"
    - "Summarizing"
    - "Synthesizing"

    Select the category that **best represents the main weakness** in the student's response. Then, provide a brief explanation for your choice.

    Return JSON in the following format:
    {{
    "primary_error": "<ERROR_TYPE>",
    "explanation": "<elaboration of the error in objective tone>"
    }}

    Rule for explaination:
    The explaination must be objective and refer to the user as 'The student'.
    A few examples are:
    - The student is at a Beginner level and struggling with fluency, likely due to a limited vocabulary and difficulty constructing sentences spontaneously.
    - The student is struggling with basic essay structure, specifically the thesis statement and topic sentences. This indicates a lack of understanding of how to organize ideas logically in writing.
    - The student is struggling with coherence in their spoken response, likely due to a lack of clear organization and logical flow.
    """

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.5-flash-preview-05-20",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = model.generate_content(error_prompt)
        response_json = json.loads(response.text)
        primary_error = response_json.get("primary_error", "Unknown Error")
        explanation = response_json.get("explanation", "")
        logger.info(f"Primary Error: {primary_error}")
        logger.info(f"Explanation: {explanation}")
        return {"primary_error": primary_error, "explanation": explanation}
    except Exception as e:
        logger.error(f"Error processing with GenerativeModel: {e}")
        return {
            "primary_error": "Unknown Error",
            "explanation": "Error processing with GenerativeModel",
        }


#         "output_content": {
#             "response": "I'm ready to help with your practice!",
#             "ui_actions": [],
#         }
