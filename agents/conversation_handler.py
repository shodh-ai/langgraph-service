import json
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)


async def conversation_handler_node(state: AgentGraphState) -> dict:
    logger.info(
        f"ConversationHandlerNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    api_key = os.getenv("GOOGLE_API_KEY")
    try:
        logger.debug("Attempting genai.configure()...")
        genai.configure(api_key=api_key)
        logger.debug("genai.configure() successful.")

        logger.debug(
            f"Attempting to initialize GenerativeModel: gemini-2.5-flash-preview-05-20"
        )
        model = genai.GenerativeModel(
            "gemini-2.5-flash-preview-05-20",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        logger.debug("GenerativeModel initialized successfully.")

        system_prompt_text = """
        You are Rox, a friendly and encouraging AI guide.
        
        Your output MUST be a JSON object with the following structure:
        {{
            "tts": "The text-to-speech for the message."
        }}
        """

        llm_instruction = state.get("llm_instruction", "")
        user_data = state.get("user_data", {})
        transcript = state.get("transcript", "")
        logger.info(f"LLM instruction: {llm_instruction}")
        logger.info(f"User data: {user_data}")
        logger.info(f"Transcript: {transcript}")
        user_prompt_text = f"""
        {llm_instruction}
        
        User data: {user_data}
        User Query: {transcript}
        """
        full_prompt = f"{system_prompt_text}\n\n{user_prompt_text}"

        logger.debug(f"Full prompt for greeting LLM: {full_prompt}")

        raw_llm_response_text = ""
        try:
            logger.debug("Attempting model.generate_content_async()...")
            response = await model.generate_content_async(full_prompt)
            logger.debug("model.generate_content_async() successful.")
            raw_llm_response_text = response.text
            logger.info(f"Raw LLM Response for greeting: {raw_llm_response_text}")
        except Exception as gen_err:
            logger.error(
                f"Error during model.generate_content_async(): {gen_err}", exc_info=True
            )
            fallback_tts = f"Hello! Welcome. (LLM Generation Error)"
            logger.info(
                f"ConversationalManagerNode: Using fallback greeting due to generation error: {fallback_tts}"
            )
            return {"greeting_data": {"greeting_tts": fallback_tts}}

        try:
            llm_output_json = json.loads(raw_llm_response_text)
            greeting_tts = llm_output_json.get(
                "tts",
                f"Hello! I'm Rox, welcome! (JSON Key Missing)",
            )
        except json.JSONDecodeError as json_err:
            logger.error(
                f"JSONDecodeError parsing LLM greeting response. Error: {json_err}. Raw text: {raw_llm_response_text}"
            )
            greeting_tts = f"Hello! I'm Rox. (JSON Parse Error)"
        except Exception as parse_err:
            logger.error(
                f"Unexpected error parsing LLM greeting response. Error: {parse_err}. Raw text: {raw_llm_response_text}"
            )
            greeting_tts = f"Hello! I'm Rox. (Unexpected Parse Error)"

        logger.info(
            f"ConversationalManagerNode: LLM-generated greeting: {greeting_tts}"
        )
        return {"greeting_data": {"greeting_tts": greeting_tts}}

    except Exception as e:
        logger.error(
            f"Outer error in handle_home_greeting_node (e.g., config, model init): {e}",
            exc_info=True,
        )
        fallback_tts = f"Hello! Welcome. (LLM Setup Error)"
        logger.info(
            f"ConversationalManagerNode: Using fallback greeting due to setup error: {fallback_tts}"
        )
        return {"greeting_data": {"greeting_tts": fallback_tts}}
