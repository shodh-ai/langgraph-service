import json
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)


async def feedback_generator_node(state: AgentGraphState) -> dict:
    logger.info(
        f"FeedbackGeneratorNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    diagnosis = state.get("explanation", "")
    user_data = state.get("user_data", {})
    data = state.get("document_data", [])
    pedagogical_strategy = state.get("chosen_pedagogical_strategy", "")
    prioritize = state.get("prioritized_issue", "")

    if data == []:
        raise ValueError("No data provided")

    explain_strategy = [entry.get("Explain Strategy", "") for entry in data]
    example_feedback = [entry.get("Provide Example Feedback", "") for entry in data]

    prompt = f"""
    You are 'The Structuralist' AI TOEFL Tutor.
    Student Diagnosis: {diagnosis}
    We are prioritizing {prioritize}
    Student Profile: {user_data}
    Here's how 'The Structuralist' typically explains the current strategy ('{pedagogical_strategy}'):
    {explain_strategy}
    Here are examples of how 'The Structuralist' gives feedback for this kind of issue to a frustrated beginner:
    {example_feedback}
    Based on all this, generate the specific feedback text for the student regarding their diagnosed issue. Ensure your tone is patient and encouraging. Include UI actions for highlighting.
    Format as JSON: {{\"text_for_tts\": \"...\"}}
    """

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = model.generate_content(prompt)
        response_json = json.loads(response.text)
        text_for_tts = response_json.get("text_for_tts", "")
        logger.info(f"Text for TTS: {text_for_tts}")
        return {"greeting_data": {"greeting_tts": text_for_tts}}
    except Exception as e:
        logger.error(f"Error processing with GenerativeModel: {e}")
        return {
            "greeting_data": {"greeting_tts": "Error processing with GenerativeModel"}
        }
