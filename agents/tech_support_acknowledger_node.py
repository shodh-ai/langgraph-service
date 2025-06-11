import logging
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

async def tech_support_acknowledger_node(state: AgentGraphState) -> Dict[str, Any]:
    user_id = state.get("user_id", "unknown_user")
    logger.info(f"Tech Support Acknowledger Node activated for user {user_id}")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set for tech support node.")
        return {
            "output_content": {
                "text_for_tts": "I understand you're facing an issue. I'm having a bit of trouble accessing my full support features right now, but please know our team is always working to improve things.",
                "ui_actions": []
            },
            "student_memory_context": state.get("student_memory_context", {})
        }

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )

        transcript = state.get("transcript", "")
        # Assuming p1_extracted_entities or a similar field holds NLU results
        # If this node can be called from various points, this field name might need to be more generic
        # or passed consistently by the router.
        extracted_entities = state.get("p1_extracted_entities", state.get("extracted_entities", {}))
        student_memory = state.get("student_memory_context")
        if not isinstance(student_memory, dict):
            student_memory = {} # Ensure student_memory is a dictionary to prevent errors

        active_persona = state.get("active_persona", "Nurturer")
        current_context_details = state.get("current_context") # Get as object or None

        # Safely get student name from student_memory
        profile_data = student_memory.get("profile", {})
        student_name = profile_data.get("name", "Student")
        issue_description = extracted_entities.get("issue_description", "the issue you mentioned")
        reported_emotion = extracted_entities.get("reported_emotion", "concerned")

        # Initialize defaults for context-dependent fields
        # user_id is initialized earlier in the function as: user_id = state.get("user_id", "student")
        page = "your current activity"
        task = "what you were doing"
        request_timestamp_str = "unknown_time"

        if current_context_details: # If it's an InteractionRequestContext object
            user_id = getattr(current_context_details, "user_id", user_id) # Update user_id if available in context, else keep state's
            student_name = getattr(current_context_details, "student_name", student_name) 
            page = getattr(current_context_details, "current_page_name", "your current activity")
            task = getattr(current_context_details, "current_task_name", "what you were doing")
            request_timestamp_dt = getattr(current_context_details, "request_timestamp", None)
            if request_timestamp_dt and hasattr(request_timestamp_dt, 'isoformat'):
                request_timestamp_str = request_timestamp_dt.isoformat()
            else:
                request_timestamp_str = "unknown_time"
            logger.info(f"TechSupportNode: Fetched from current_context_details - user_id: {user_id}, student_name: {student_name}, page: {page}, task: {task}, timestamp: {request_timestamp_str}")
        else:
            logger.warning("TechSupportNode: current_context_details is None. Using default/state values.")

        prompt_parts = [
            f"You are Rox, an AI TOEFL Tutor, speaking as the '{active_persona}' persona.",
            f"The student (user_id: {user_id}, name: {student_name}) reported a technical issue.",
            f"Student's report (transcript): '{transcript}'.",
            f"Detected issue: '{issue_description}'. Detected emotion: '{reported_emotion}'.",
            f"Context: Student was on page '{page}' performing task '{task}'.",
            "Your task is to respond empathetically, acknowledge the issue, state it's been noted, and offer helpful next steps.",
            "Instructions:",
            "1. Acknowledge the student's report and validate their feeling (e.g., frustration, concern). Use their name.",
            "2. Clearly state that the issue has been noted/logged for the technical team.",
            "3. (Optional, if simple and common) If you know a very common quick fix (like 'refreshing the page sometimes helps with display issues'), you can mention it briefly.",
            "4. Offer 1-2 clear, actionable next steps. Examples:",
            "   - 'Would you like to try [the previous task/activity] again?'",
            "   - 'Perhaps we can move to a different type of activity for now, like a [specific alternative e.g., vocabulary drill]?'",
            "   - 'You could also try refreshing the page. Would you like to do that, or shall we pick a different activity?'",
            "5. Keep the overall tone supportive and reassuring.",
            "Return your response as a JSON object with the EXACT following structure:",
            "{\"text_for_tts\": \"Your empathetic response and suggestions...\", \"ui_actions\": [{\"action_type\": \"SHOW_BUTTON_OPTIONS\", \"target_element_id\": \"p1ActionButtons\", \"parameters\": {\"buttons\": [{\"label\": \"Suggested Action 1\", \"next_graph_hint\": \"HINT_FOR_ACTION_1\"}, {\"label\": \"Suggested Action 2\", \"next_graph_hint\": \"HINT_FOR_ACTION_2\"}]}}], \"logged_issue_summary\": \"A brief summary of the issue for internal logging purposes.\"}",
            "The 'next_graph_hint' for buttons should be a placeholder or a conceptual hint that the graph router can later interpret (e.g., 'TRY_AGAIN_CURRENT_TASK', 'SWITCH_TO_VOCABULARY')."
        ]
        prompt = "\n".join(prompt_parts)

        logger.debug(f"Tech Support Acknowledger LLM Prompt:\n{prompt}")
        response = await model.generate_content_async(prompt)
        logger.debug(f"Tech Support Acknowledger LLM Raw Response: {response.text}")
        response_json = json.loads(response.text)

        # Update student memory
        updated_student_memory = {**student_memory}
        updated_student_memory["last_ai_action"] = "acknowledged_tech_issue_offered_options" # Generic field
        if "profile" in updated_student_memory and isinstance(updated_student_memory["profile"], dict):
            updated_student_memory["profile"]["affective_state"] = f"acknowledged_{reported_emotion}"
        else:
            updated_student_memory["profile"] = {"affective_state": f"acknowledged_{reported_emotion}"}
        
        # Log the issue summary to memory
        context_for_log = current_context_details.model_dump(mode='json') if current_context_details and hasattr(current_context_details, 'model_dump') else None
        
        reported_issues_list = updated_student_memory.get("reported_technical_issues", [])
        
        issue_log_entry = {
            "timestamp": request_timestamp_str,
            "issue_transcript": transcript,
            "parsed_description": issue_description,
            "llm_summary_for_log": response_json.get("logged_issue_summary", "LLM did not provide summary."),
            "context": context_for_log
        }

        if isinstance(reported_issues_list, list):
            reported_issues_list.append(issue_log_entry)
            updated_student_memory["reported_technical_issues"] = reported_issues_list
        else: # In case it was not a list, reinitialize
            logger.warning("student_memory_context.reported_technical_issues was not a list. Reinitializing.")
            updated_student_memory["reported_technical_issues"] = [issue_log_entry]

        return {
            "output_content": {
                "text_for_tts": response_json.get("text_for_tts", "I'm sorry to hear you're having trouble. I've noted the issue."),
                "ui_actions": response_json.get("ui_actions", [])
            },
            "student_memory_context": updated_student_memory
        }

    except Exception as e:
        logger.error(f"Error in tech_support_acknowledger_node: {e}", exc_info=True)
        return {
            "output_content": {
                "text_for_tts": "I understand you've hit a snag. While I can't fix it directly, I'll make sure our team is aware.",
                "ui_actions": []
            },
            "student_memory_context": state.get("student_memory_context", {})
        }
