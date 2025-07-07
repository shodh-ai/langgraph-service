# agents/modeling_nlu_node.py
import json
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def modeling_nlu_node(state: AgentGraphState) -> dict:
    """
    Handles NLU for a modeling session turn AND preserves the critical modeling state.
    """
    logger.info("---Executing Modeling-Specific NLU Node (State-Preserving)---")
    
    # Read the critical state we must pass along.
    intermediate_payload = state.get("intermediate_modelling_payload")
    current_context = state.get("current_context", {})
    
    transcript = state.get("transcript")
    if not transcript:
        # If there's no transcript, we can't classify, but we must preserve state.
        return {
            "student_intent_for_model_turn": "STATE_CONFUSION", 
            "intermediate_modelling_payload": intermediate_payload,
            "current_context": current_context
        }

    possible_intents = ["ASK_ABOUT_MODEL", "CONFIRM_UNDERSTANDING", "GENERAL_QUESTION", "STATE_CONFUSION"]
    chat_history = state.get("chat_history", [])
    # In a modeling session, the last AI statement is the explanation of the model.
    last_ai_statement = chat_history[-2]['content'] if len(chat_history) > 1 else "the demonstrated example"
    
    prompt = f"""
    You are an NLU assistant for an AI tutor. A student is in a modeling session where the tutor just demonstrated something and said: '{last_ai_statement}'.
    The student responded: "{transcript}"

    Your task is to categorize the student's intent into ONE of the following:
    - "ASK_ABOUT_MODEL": The student is asking a direct question about the demonstration (e.g., "Why did you do that?", "What does that mean?").
    - "CONFIRM_UNDERSTANDING": The student indicates they understand and are ready to move on (e.g., "Okay," "I get it," "Got it").
    - "GENERAL_QUESTION": The student is asking a question not directly related to the immediate demonstration.
    - "STATE_CONFUSION": The student seems confused or lost.

    Return ONLY a single JSON object in the format: {{"intent": "<INTENT_NAME>"}}
    """
    
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config=GenerationConfig(response_mime_type="application/json"),
    )
    
    try:
        response = await model.generate_content_async(prompt)
        llm_response_json = json.loads(response.text)
        classified_intent = llm_response_json.get("intent", "STATE_CONFUSION")
    except Exception as e:
        logger.error(f"Error during modeling NLU classification: {e}")
        classified_intent = "STATE_CONFUSION"

    logger.info(f"Modeling NLU classified intent as: {classified_intent}")

    # Return the new intent AND the preserved state.
    return {
        "student_intent_for_model_turn": classified_intent,
        "intermediate_modelling_payload": intermediate_payload,
        "current_context": current_context
    }
