import json
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)


async def initial_report_generation_node(state: AgentGraphState) -> dict:
    logger.info(
        f"InitialReportGenerationNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    transcript = state.get("transcript")
    api_key = os.getenv("GOOGLE_API_KEY")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.5-flash-preview-05-20",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        prompt = f"""
        You are an expert analyser in English. A student has written the following paragraph:
        {transcript}
        
        Analyse the paragraph and generate a diagnosis report.
        The report must answer the questions:
        1. Initial thoughts
        2. Errors in the paragraph
        3. Vocabulary Enrichment

        Return a JSON object:
        {{
            "initial_thoughts": "",
            "errors": "",
            "vocabulary_enrichment": ""
        }}
        
        """
        response = model.generate_content(prompt)
        response_json = json.loads(response.text)

        prompt = f"""
        You are an expert analyser in English. A student has written the following paragraph:
        {transcript}

        A second analyser has given you the following report:
        {response_json}
        
        Analyse the paragraph and generate a diagnosis report.
        The report must answer the questions:
        1. Estimated overall English comfort level ("Beginner", "Conversational", "Fluent", "Proficient", "Near-Native", "Native")
        2. Initial Impression
        3. Speaking Strengths
        4. Fluency
        5. Grammar
        6. Vocabulary

        Return a JSON object:
        {{
            "estimated_overall_english_comfort_level": "",
            "initial_impression": "",
            "speaking_strengths": "",
            "fluency": "",
            "grammar": "",
            "vocabulary": ""
        }}
        """
        response = model.generate_content(prompt)
        response_json = json.loads(response.text)
        print(response_json)
        return response_json

    except Exception as e:
        logger.error(f"Error processing with GenerativeModel: {e}")
        return {
            "estimated_overall_english_comfort_level": "Unknown Diagnosis",
            "initial_impression": "Error processing with GenerativeModel",
            "speaking_strengths": "Error processing with GenerativeModel",
            "fluency": "Error processing with GenerativeModel",
            "grammar": "Error processing with GenerativeModel",
            "vocabulary": "Error processing with GenerativeModel",
        }
