import json
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from typing import Dict, AsyncIterator

logger = logging.getLogger(__name__)

async def conversation_handler_node(state: AgentGraphState) -> AsyncIterator[Dict]:
    """
    Handles the main conversational turn. It streams the response back to the client
    via 'streaming_text_chunk' and yields the final complete response to update the 
    graph state for subsequent nodes using 'conversational_tts' and 'output_content'.
    """
    logger.info(
        f"ConversationHandlerNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    api_key = os.getenv("GOOGLE_API_KEY")
    full_response_text = ""  # Accumulator for the full response

    try:
        logger.debug("Attempting genai.configure()...")
        genai.configure(api_key=api_key)
        logger.debug("genai.configure() successful.")

        logger.debug(
            f"Attempting to initialize GenerativeModel: gemini-1.5-flash-latest"
        )
        # Corrected model name and removed restrictive JSON mime type from original code
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        logger.debug("GenerativeModel initialized successfully.")

        llm_instruction = state.get("llm_instruction", "")
        user_data = state.get("user_data", {})
        transcript = state.get("transcript", "")

        logger.info(f"Retrieved llm_instruction from state: '{llm_instruction[:100]}...' if present else 'Not present'")
        logger.info(f"User data: {user_data}")
        logger.info(f"Transcript: '{transcript}'")

        if llm_instruction:
            full_prompt = llm_instruction
            logger.info(f"Using llm_instruction from state directly as full_prompt.")
        else:
            system_prompt_text = """
            You are Rox, a friendly and encouraging AI guide.
            Respond directly with the text that should be spoken for text-to-speech. Do NOT use JSON or any other special formatting.
            """
            user_prompt_text = f"User data: {json.dumps(user_data)}\nUser Query: {transcript}"
            full_prompt = f"{system_prompt_text}\n\n{user_prompt_text}"
            logger.info(f"Constructed generic prompt as llm_instruction was not found in state.")

        logger.debug(f"Full prompt for LLM: {full_prompt}")

        try:
            logger.debug("Attempting model.generate_content_async(stream=True)...")
            async_stream = await model.generate_content_async(full_prompt, stream=True)
            logger.debug("model.generate_content_async(stream=True) call successful, iterating over stream...")
            
            streamed_something = False
            async for chunk in async_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    logger.debug(f"Streaming chunk text: {chunk.text}")
                    yield {"streaming_text_chunk": chunk.text}
                    full_response_text += chunk.text
                    streamed_something = True
                else:
                    logger.debug(f"Received a chunk without text or empty text: {chunk}")
            
            if not streamed_something:
                logger.warning("LLM stream completed without yielding any text chunks.")
                fallback_message = "I'm ready, but it seems I have no specific words for this moment!"
                yield {"streaming_text_chunk": fallback_message}
                full_response_text = fallback_message

        except Exception as gen_err:
            logger.error(
                f"Error during model.generate_content_async(stream=True): {gen_err}", exc_info=True
            )
            fallback_tts_chunk = "I'm having a little trouble thinking right now, but I'm here to help! (LLM Stream Error)"
            logger.info(
                f"ConversationHandlerNode: Yielding fallback TTS chunk due to stream generation error: {fallback_tts_chunk}"
            )
            yield {"streaming_text_chunk": fallback_tts_chunk}
            full_response_text = fallback_tts_chunk
        
        logger.info(f"ConversationHandlerNode: Finished streaming. Yielding final accumulated response to state.")
        yield {
            "conversational_tts": full_response_text,
            "output_content": full_response_text,
        }

    except Exception as e:
        logger.error(
            f"Outer error in ConversationHandlerNode (e.g., config, model init): {e}",
            exc_info=True,
        )
        fallback_tts_chunk = "Looks like I'm having a bit of trouble starting up. Let's try again in a moment. (LLM Setup Error)"
        logger.info(
            f"ConversationHandlerNode: Yielding fallback TTS due to setup error: {fallback_tts_chunk}"
        )
        yield {"streaming_text_chunk": fallback_tts_chunk}
        yield {
            "conversational_tts": fallback_tts_chunk,
            "output_content": fallback_tts_chunk,
        }
