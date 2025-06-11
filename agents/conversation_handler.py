import json
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from typing import Dict, AsyncIterator

logger = logging.getLogger(__name__)


async def conversation_handler_node(state: AgentGraphState) -> AsyncIterator[Dict]:
    logger.info(
        f"ConversationHandlerNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    api_key = os.getenv("GOOGLE_API_KEY")
    try:
        logger.debug("Attempting genai.configure()...")
        genai.configure(api_key=api_key)
        logger.debug("genai.configure() successful.")

        logger.debug(
            f"Attempting to initialize GenerativeModel: gemini-2.0-flash"
        )
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
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
            Respond directly with the text that should be spoken for text-to-speech. Do NOT use JSON or any other special formatting.
            """
            # Construct prompt for general conversation using transcript
            user_prompt_text = f"""
            User data: {user_data}
            User Query: {transcript}
            """ # llm_instruction is empty here, so it won't be duplicated in this branch
            full_prompt = f"{system_prompt_text}\n\n{user_prompt_text}"
            logger.info(f"Constructed generic prompt as llm_instruction was not found in state.")

        logger.debug(f"Full prompt for LLM: {full_prompt}")

        try:
            logger.debug("Attempting model.generate_content_async(stream=True)...")
            # Use stream=True for streaming responses
            async_stream = await model.generate_content_async(full_prompt, stream=True)
            logger.debug("model.generate_content_async(stream=True) call successful, iterating over stream...")
            
            streamed_something = False
            async for chunk in async_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    logger.debug(f"Streaming chunk text: {chunk.text}")
                    yield {"streaming_text_chunk": chunk.text}
                    streamed_something = True
                else:
                    # Log if a chunk doesn't have text, might indicate empty parts or metadata chunks
                    logger.debug(f"Received a chunk without text or empty text: {chunk}")
            
            if not streamed_something:
                logger.warning("LLM stream completed without yielding any text chunks.")
                # Optionally yield a fallback message if nothing was streamed
                # yield {"streaming_text_chunk": "I'm ready, but it seems I have no specific words for this moment!"}

        except Exception as gen_err:
            logger.error(
                f"Error during model.generate_content_async(stream=True): {gen_err}", exc_info=True
            )
            fallback_tts_chunk = f"I'm having a little trouble thinking right now, but I'm here to help! (LLM Stream Error)"
            logger.info(
                f"ConversationHandlerNode: Yielding fallback TTS chunk due to stream generation error: {fallback_tts_chunk}"
            )
            yield {"streaming_text_chunk": fallback_tts_chunk}
        # No explicit final return needed for an async generator node that only yields state updates.
        # If there were UI actions to yield *after* the stream, they would go here.
        # For example: yield {"final_ui_actions": [...]} 
        # For now, we are only streaming text from this node. 

    except Exception as e:
        logger.error(
            f"Outer error in ConversationHandlerNode (e.g., config, model init): {e}",
            exc_info=True,
        )
        fallback_tts_chunk = f"Looks like I'm having a bit of trouble starting up. Let's try again in a moment. (LLM Setup Error)"
        logger.info(
            f"ConversationHandlerNode: Yielding fallback TTS chunk due to setup error: {fallback_tts_chunk}"
        )
        yield {"streaming_text_chunk": fallback_tts_chunk}
        # Ensure the generator actually finishes if an error occurs here, though yield should handle it.
        return # Explicitly return to signify the end of the generator in this error path
