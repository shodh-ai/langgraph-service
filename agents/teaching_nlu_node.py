# agents/teaching_nlu_node.py (The Definitive Version)
import json
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def teaching_nlu_node(state: AgentGraphState) -> dict:
    """
    Handles NLU for a teaching turn AND preserves the critical plan state.
    """
    logger.info("---Executing Teaching-Specific NLU Node (State-Preserving)---")
    
    # Read the critical state we must pass along.
    plan = state.get("pedagogical_plan")
    # THIS IS THE TRUE FIX: Provide a default value to prevent passing None downstream.
    current_index = state.get("current_plan_step_index", 0)
    
    transcript = state.get("transcript")
    if not transcript:
        return {"student_intent_for_lesson_turn": "STATE_CONFUSION", "pedagogical_plan": plan, "current_plan_step_index": current_index}

    possible_intents = ["CONFIRM_UNDERSTANDING", "ASK_CLARIFICATION_QUESTION", "STATE_CONFUSION"]
    chat_history = state.get("chat_history", [])
    last_ai_statement = chat_history[-2]['content'] if len(chat_history) > 1 else "the current topic"
    
    prompt = f"""
    You are an NLU assistant. A student is in a lesson and just heard: '{last_ai_statement}'.
    The student responded: "{transcript}"
    Categorize the intent into ONE of these: {possible_intents}
    Return ONLY JSON: {{"intent": "<INTENT_NAME>"}}
    """
    
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config=GenerationConfig(response_mime_type="application/json"),
    )
    response = await model.generate_content_async(prompt)
    llm_response_json = json.loads(response.text)
    
    classified_intent = llm_response_json.get("intent")
    logger.info(f"Teaching NLU classified intent as: {classified_intent}")

    # Return the new intent AND the preserved state.
    return {
        "student_intent_for_lesson_turn": classified_intent,
        "pedagogical_plan": plan,
        "current_plan_step_index": current_index
    }
