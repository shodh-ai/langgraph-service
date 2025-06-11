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

        llm_instruction = state.get("llm_instruction", "") # This may come from welcome_prompt_node
        user_data = state.get("user_data", {})
        transcript = state.get("transcript", "") # Will be empty on the first turn after welcome_prompt

        logger.info(f"Retrieved llm_instruction from state: '{llm_instruction[:100]}...' if present else 'Not present'")
        logger.info(f"User data: {user_data}")
        logger.info(f"Transcript: '{transcript}'")

        if llm_instruction:
            # If welcome_prompt_node (or another node) provided specific instructions, use them directly.
            # This instruction is assumed to be complete and already formatted for the LLM,
            # including persona, desired output format (JSON with "tts" key), and the specific message.
            full_prompt = llm_instruction
            logger.info(f"Using llm_instruction from state directly as full_prompt.")
        else:
            # For generic conversation turns, build the prompt using the default system prompt
            system_prompt_text = """
            You are Rox, a friendly and encouraging AI guide.
            Your output MUST be a JSON object with the following structure:
            {{
                "tts": "The text-to-speech for the message."
            }}
            """
            # Construct prompt for general conversation using transcript
            user_prompt_text = f"""
            User data: {user_data}
            User Query: {transcript}
            """ # llm_instruction is empty here, so it won't be duplicated in this branch
            full_prompt = f"{system_prompt_text}\n\n{user_prompt_text}"
            logger.info(f"Constructed generic prompt as llm_instruction was not found in state.")

        logger.debug(f"Full prompt for LLM: {full_prompt}")

        raw_llm_response_text = ""
        try:
            logger.debug("Attempting model.generate_content_async()...")
            response = await model.generate_content_async(full_prompt)
            logger.debug("model.generate_content_async() successful.")
            raw_llm_response_text = response.text
            logger.info(f"Raw LLM Response: {raw_llm_response_text}")
        except Exception as gen_err:
            logger.error(
                f"Error during model.generate_content_async(): {gen_err}", exc_info=True
            )
            fallback_tts = f"I'm having a little trouble thinking right now, but I'm here to help! (LLM Generation Error)"
            logger.info(
                f"ConversationHandlerNode: Using fallback TTS due to generation error: {fallback_tts}"
            )
            return {"conversational_tts": fallback_tts}

        try:
            llm_output_json = json.loads(raw_llm_response_text)
            extracted_tts = llm_output_json.get(
                "tts",
                f"I seem to have misplaced my words! Let's try that again. (JSON Key Missing)",
            )
        except json.JSONDecodeError as json_err:
            logger.error(
                f"JSONDecodeError parsing LLM response. Error: {json_err}. Raw text: {raw_llm_response_text}"
            )
            extracted_tts = f"My thoughts got a bit jumbled! What were we saying? (JSON Parse Error)"
        except Exception as parse_err:
            logger.error(
                f"Unexpected error parsing LLM response. Error: {parse_err}. Raw text: {raw_llm_response_text}"
            )
            extracted_tts = f"A tiny hiccup in our chat! Let's continue. (Unexpected Parse Error)"

        logger.info(
            f"ConversationHandlerNode: LLM-generated TTS: {extracted_tts}"
        )
        return {"conversational_tts": extracted_tts}

    except Exception as e:
        logger.error(
            f"Outer error in ConversationHandlerNode (e.g., config, model init): {e}",
            exc_info=True,
        )
        fallback_tts = f"Looks like I'm having a bit of trouble starting up. Let's try again in a moment. (LLM Setup Error)"
        logger.info(
            f"ConversationHandlerNode: Using fallback TTS due to setup error: {fallback_tts}"
        )
        return {"conversational_tts": fallback_tts}
