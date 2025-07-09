# agents/modelling_nlu_node.py

import json
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def modelling_nlu_node(state: AgentGraphState) -> dict:
    """
    Handles NLU for a modelling session turn.
    """
    logger.info("---Executing Modelling NLU Node---")

    transcript = state.get("transcript")
    if not transcript:
        logger.warning("No transcript found, defaulting to STATE_CONFUSION.")
        return {"student_intent_for_modelling_turn": "STATE_CONFUSION"}

    possible_intents = ["ASK_ABOUT_MODEL", "CONFIRM_UNDERSTANDING", "GENERAL_QUESTION", "STATE_CONFUSION"]
    chat_history = state.get("chat_history", [])
    last_ai_statement = chat_history[-2]['content'] if len(chat_history) > 1 else "the demonstrated example"
    
    prompt = f"""
    You are an NLU assistant for an AI tutor. A student is in a modellingg session and just saw a demonstration.
    Tutor's last statement: '{last_ai_statement}'.
    Student's response: "{transcript}"

    Categorize the student's intent into ONE of these: {possible_intents}
    - "ASK_ABOUT_MODEL": Direct question about the demonstration.
    - "CONFIRM_UNDERSTANDING": Student indicates they get it.
    - "GENERAL_QUESTION": Question not about the immediate demonstration.
    - "STATE_CONFUSION": Student seems lost.

    Return ONLY a single JSON object: {{"intent": "<INTENT_NAME>"}}
    """
    
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config=GenerationConfig(response_mime_type="application/json"),
    )
    
    try:
        response = await model.generate_content_async(prompt)
        llm_response_json = json.loads(response.text)
        classified_intent = llm_response_json.get("intent", "STATE_CONFUSION")
    except Exception as e:
        logger.error(f"Error during modelling NLU classification: {e}")
        classified_intent = "STATE_CONFUSION"

    logger.info(f"Modelling NLU classified intent as: {classified_intent}")

    # Return ONLY the new information.
    return {
        "student_intent_for_modelling_turn": classified_intent,
    }
