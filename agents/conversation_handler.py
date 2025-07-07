# agents/conversation_handler.py
import json
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)

# --- HYBRID NLU ENGINE ---

async def conversation_handler_node(state: AgentGraphState) -> dict:
    """
    This node performs NLU using a hybrid approach.
    1. It first checks for simple, hardcoded commands (Fast Path).
    2. If no rule matches, it uses an LLM for flexible, contextual intent classification (Smart Path).
    """
    logger.info("---Executing Hybrid Conversation Handler Node ---")
    
    try:
        transcript = state.get("transcript")
        if not transcript or not transcript.strip():
            logger.warning("Transcript is empty. Classifying as STUDENT_SILENT.")
            return {"classified_student_intent": "STUDENT_SILENT"}

        logger.info(f"ConversationHandlerNode: Received transcript: '{transcript}'")
        transcript_lower = transcript.lower()

        # ======================================================================
        #  LAYER 1: THE "FAST PATH" - Rule-based matching for clear commands
        # ======================================================================
        
        # --- Test Plan Implementation (A special case of the fast path) ---
        # This is now more specific to avoid false positives.
        if "can you hear me" in transcript_lower:
             # Check for a positive affirmation in the response.
            if any(phrase in transcript_lower for phrase in ["yes i can", "i can hear you"]):
                logger.info("Test plan condition met: Positive response to 'can you hear me'.")
                return {
                    "final_ui_actions": [{
                        "action_type": "SPEAK_TEXT",
                        "parameters": {"text": "Great! Welcome to the course. We can now begin teaching you."}
                    }],
                    "classified_student_intent": "TEST_PASSED_TERMINATE"
                }
        
        # You could add other simple rules here if needed, but it's better to
        # handle most things in the LLM or in flow-specific NLU nodes (like rox_nlu_handler).

        # ======================================================================
        #  LAYER 2: THE "SMART PATH" - LLM-based classification for everything else
        # =====================================================================S=====================================

        logger.info("No fast-path rule matched. Proceeding with LLM-based NLU.")
        
        current_context = state.get("current_context", {})
        task_stage = current_context.get("task_stage", "")
        chat_history = state.get("chat_history", [])

        # --- DYNAMIC PROMPT SELECTION BASED ON CONTEXT ---
        possible_intents = []
        nlu_instructions = ""

        # This logic is excellent and should be kept. It makes your LLM very context-aware.
        if task_stage == "TEACHING_PAGE_TURN":
            possible_intents = ["CONFIRM_UNDERSTANDING", "ASK_CLARIFICATION_QUESTION", "STATE_CONFUSION", "CONTINUE_LESSON"]
            last_ai_statement = chat_history[-2]['content'] if len(chat_history) > 1 else "the current topic"
            nlu_instructions = f"The AI is in the middle of a lesson and just explained a concept. The AI's last statement was: '{last_ai_statement}'. Now, classify the student's following statement."

        elif task_stage == "MODELLING_PAGE_TURN":
            possible_intents = ["CONFIRM_UNDERSTANDING", "ASK_ABOUT_MODEL", "STATE_CONFUSION", "GENERAL_QUESTION"]
            last_ai_statement = chat_history[-2]['content'] if len(chat_history) > 1 else "the current modeling example"
            nlu_instructions = f"The AI is in the middle of a modeling session, showing the student how to do something. The AI's last statement was: '{last_ai_statement}'. Now, classify the student's following statement."
            
        else: # Default case
            possible_intents = ["GENERAL_CONFIRM", "GENERAL_REJECT", "GENERAL_QUESTION", "SMALL_TALK"]
            nlu_instructions = "Classify the user's general intent."

        prompt = f"""
        You are an NLU assistant for an AI Tutor.
        Context: {nlu_instructions}
        Student said: "{transcript}"

        Categorize the student's intent into ONE of the following types: {possible_intents}
        Return ONLY a single valid JSON object with the format: {{"intent": "<INTENT_NAME>"}}
        """
        
        # --- Call the LLM ---
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(prompt)
        llm_response_json = json.loads(response.text)
        
        classified_intent = llm_response_json.get("intent")
        logger.info(f"LLM NLU classified intent as: {classified_intent}")

        return {"classified_student_intent": classified_intent}

    except Exception as e:
        logger.error(f"ConversationHandlerNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": str(e), "route_to_error_handler": True}
