import logging
import os
import json
import datetime
from typing import Dict, Any, List
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def session_wrap_up_node(state: AgentGraphState) -> Dict[str, Any]:
    user_id = state.get("user_id", "unknown_user")
    logger.info(f"Session Wrap Up Node activated for user {user_id}")

    student_memory = state.get("student_memory_context") or {}
    student_name = student_memory.get("profile", {}).get("name", "student")
    active_persona = state.get("active_persona", "Nurturer")
    current_context = state.get("current_context", {})
    task_stage = getattr(current_context, 'task_stage', None) if hasattr(current_context, 'task_stage') else current_context.get('task_stage', 'UNKNOWN_REASON')

    # Determine reason for wrap-up (simplified for now)
    # In a more complex system, this could be more detailed based on NLU intent or system flags
    reason_for_ending = f"Session ended by student or system (task_stage: {task_stage})"
    if state.get('nlu_intent') == 'INTENT_TO_QUIT_SESSION':
        reason_for_ending = "Student indicated they are done for today."
    elif task_stage == "SYSTEM_MAX_INACTIVITY_REACHED": # Assuming such a stage exists
        reason_for_ending = "Session ended due to prolonged inactivity."

    # Craft concluding message
    api_key = os.getenv("GOOGLE_API_KEY")
    tts_text = f"Okay, {student_name}, we'll stop here for today. Your progress has been saved. Great work, and I look forward to our next session!"

    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                "gemini-2.0-flash",
                generation_config=GenerationConfig(response_mime_type="application/json"),
            )
            
            # Simplified summary for now
            session_summary_placeholder = "We had a productive session."
            if student_memory.get("last_ai_action_on_p1"):
                session_summary_placeholder = f"We worked on: {student_memory.get('last_ai_action_on_p1')}."
            
            prompt_parts = [
                f"You are Rox, an AI TOEFL Tutor, speaking as the '{active_persona}' persona.",
                f"The student, {student_name}, is ending their session.",
                f"Reason for ending: {reason_for_ending}",
                f"Brief summary of today's work: {session_summary_placeholder}",
                "Generate a polite and encouraging concluding message:",
                "1. Acknowledge the session is ending.",
                "2. Briefly mention something positive about the session if applicable.",
                "3. Reassure them their progress is saved.",
                "4. Invite them to return.",
                "Keep it concise (1-3 sentences).",
                "Return JSON: {\"text_for_tts\": \"<your concluding message>\"}"
            ]
            prompt = "\n".join(prompt_parts)
            logger.debug(f"Session Wrap Up LLM Prompt:\n{prompt}")
            response = await model.generate_content_async(prompt)
            response_json = json.loads(response.text)
            tts_text = response_json.get("text_for_tts", tts_text) # Fallback to template if LLM fails
        except Exception as e:
            logger.error(f"Error in session_wrap_up_node LLM call: {e}", exc_info=True)
            # Fallback to templated response is already set
    
    # Prepare UI Actions
    ui_actions: List[Dict[str, Any]] = [
        {
            "action_type": "DISABLE_INTERACTION_ELEMENTS",
            "parameters": {"element_ids": ["user_input_field", "mic_button", "send_button"]} # Example IDs
        },
        {
            "action_type": "SHOW_ALERT", # Optional, can be a toast or modal
            "parameters": {
                "message_type": "INFO",
                "title": "Session Ended",
                "message_body": "Your session with Rox has concluded. Your progress is saved. See you next time!",
                "duration_ms": 5000
            }
        }
        # Consider NAVIGATE_TO_PAGE if there's a dedicated 'goodbye' or 'session summary' page
        # e.g., { "action_type": "NAVIGATE_TO_PAGE", "parameters": { "page_name": "P_SessionSummary" } }
    ]

    # Prepare data for final session save
    final_session_data = {
        "session_id": state.get("session_id", "unknown_session"),
        "user_id": user_id,
        "session_end_time_utc": datetime.datetime.utcnow().isoformat(),
        "reason_for_ending": reason_for_ending,
        "session_activity_summary": student_memory.get("interaction_history_summary", "No summary available"), # Placeholder
        "chat_history_on_end": state.get("chat_history", [])
        # Add any other relevant session data, e.g., incomplete tasks, final state of student_memory_context
    }

    # Update state
    output_content = {
        "text_for_tts": tts_text,
        "ui_actions": ui_actions
    }
    
    updated_student_memory = student_memory.copy()
    updated_student_memory["last_ai_action_on_p1"] = "session_wrap_up_completed"
    # Potentially clear or archive parts of student_memory_context if it's session-specific and large

    return {
        "output_content": output_content,
        "student_memory_context": updated_student_memory,
        "final_session_data_to_save": final_session_data,
        "session_is_ending": True
    }
