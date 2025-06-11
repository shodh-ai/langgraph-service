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
            "gemini-2.0-flash", # Consider making model name configurable
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        
        # First prompt for initial analysis (internal)
        prompt1 = f"""
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
        internal_response = model.generate_content(prompt1)
        internal_report_json = json.loads(internal_response.text)

        # Second prompt for the main report, using the internal analysis
        prompt2 = f"""
        You are an expert analyser in English. A student has written the following paragraph:
        {transcript}

        A second analyser has given you the following report:
        {internal_report_json}
        
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
        final_response_genai = model.generate_content(prompt2)
        final_report_json = json.loads(final_response_genai.text)
        
        logger.debug(f"Generated initial report: {final_report_json}") # Changed print to logger.debug

        # Prepare output for AgentGraphState
        conversational_text = final_report_json.get(
            "initial_impression", 
            "I've reviewed your submission. Let's look at the details." # Fallback TTS
        )
        
        # This is what updates the state for downstream nodes
        return {
            "conversational_tts": conversational_text,
            "initial_report_content": final_report_json # Store the full report
        }

    except Exception as e:
        logger.error(f"Error processing with GenerativeModel in initial_report_generation_node: {e}")
        error_tts = "I encountered an issue while generating your initial report. Please try again."
        error_report_content = {
            "error": str(e),
            "estimated_overall_english_comfort_level": "Unknown",
            "initial_impression": "Could not generate report due to an error.",
            "speaking_strengths": "N/A",
            "fluency": "N/A",
            "grammar": "N/A",
            "vocabulary": "N/A",
        }
        return {
            "conversational_tts": error_tts,
            "initial_report_content": error_report_content
        }
