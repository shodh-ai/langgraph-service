import json
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from pydantic import BaseModel

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

        system_prompt_text = """You are Rox, a friendly and encouraging AI guide. Your role is to welcome a student to the platform. \n\n Your output MUST be a JSON object with the following structure:
        {{
            "tts": "The text-to-speech for the message."
        }}"""
        # Helper function to convert state to a JSON-serializable dictionary
        def make_serializable(obj):
            if isinstance(obj, BaseModel):
                # For Pydantic models, use their built-in serialization
                return obj.model_dump()
            elif isinstance(obj, dict):
                # For dictionaries, recursively convert values
                return {k: make_serializable(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, list):
                # For lists, recursively convert items
                return [make_serializable(item) for item in obj]
            else:
                # Return other types as is (assuming they're JSON serializable)
                return obj
            
        # Convert the state object to a JSON-serializable dictionary
        serializable_state = make_serializable(dict(state))
        user_prompt_text = json.dumps(serializable_state, indent=2)
        full_prompt = f"{system_prompt_text}\n\n{user_prompt_text}"

        logger.info(
            f"GOOGLE_API_KEY loaded: {api_key[:5]}...{api_key[-5:] if len(api_key) > 10 else ''}"
        )
        logger.info(f"Full prompt for greeting LLM: {full_prompt}")

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
