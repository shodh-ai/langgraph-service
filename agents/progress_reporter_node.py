import logging
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def progress_reporter_node(state: AgentGraphState) -> Dict[str, Any]:
    logger.info(f"Progress Reporter Node activated for user {state.get('user_id', 'unknown_user')}")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set for progress reporter node.")
        return {
            "output_content": {
                "text_for_tts": "I'd love to show you your progress, but I'm having a bit of trouble accessing that right now.",
                "ui_actions": []
            }
        }

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-1.5-flash-latest",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )

        user_id = state.get("user_id", "student")
        transcript = state.get("transcript", "") # Student's query
        student_memory = state.get("student_memory_context", {})
        active_persona = state.get("active_persona", "Nurturer")
        # p1_extracted_entities = state.get("p1_extracted_entities", {})
        # current_context = state.get("current_context")

        # For this example, we'll assume student_memory_context contains relevant progress data.
        # A more complex implementation might involve fetching data from Mem0 here.
        # Example data that might be in student_memory_context:
        # student_memory = {
        #     "name": "Alex",
        #     "overall_progress_summary": "Showing good improvement in vocabulary, consistent effort in grammar.",
        #     "recent_scores": [{"skill": "Reading", "score": 75}, {"skill": "Speaking Fluency", "score": 68}],
        #     "areas_improved": ["Vocabulary recall", "Understanding of past tense"],
        #     "common_errors_summary": "Occasional subject-verb agreement issues, some hesitation in complex sentences.",
        #     "time_spent_summary": "Average 30 mins/day over the last week.",
        #     "stated_goals": ["Improve speaking confidence", "Score 90+ in TOEFL"],
        #     "fluency_trend_data": [{"date": "2024-05-01", "score": 60}, {"date": "2024-05-15", "score": 65}, {"date": "2024-05-30", "score": 68}]
        # }

        prompt_parts = [
            f"You are Rox, an AI TOEFL Tutor, currently embodying the '{active_persona}' persona.",
            f"The student (user_id: {user_id}) asked: '{transcript}'.",
            f"Here is the student's profile data: {json.dumps(student_memory.get('profile', {}), indent=2)}",
            f"And here is the student's interaction history: {json.dumps(student_memory.get('interaction_history', []), indent=2)}",
            "Your task is to synthesize this data into a clear, encouraging, and actionable progress report.",
            "Instructions:",
            "1. Acknowledge the student's query or concern (from their transcript).",
            "2. Use the 'profile' data for general student information (e.g., name, level).",
            "3. Analyze the 'interaction_history' to identify trends, recent performance, topics covered, scores, and feedback received.",
            "4. Provide an honest but positive overview of their progress based on this analysis.",
            "5. Highlight 1-2 recent specific improvements or strengths visible in the interaction history.",
            "6. Mention 1-2 areas to focus on, framing them constructively, based on patterns in the interaction history.",
            "7. If possible, suggest a concrete next step or a relevant area to practice, informed by their history and profile.",
            f"8. Maintain the '{active_persona}' persona's style (e.g., Structuralist: data-driven, direct; Nurturer: empathetic, encouraging).",
            "9. Return your response as a JSON object with the EXACT following structure:",
            "   {\"text_for_tts\": \"Your progress report message...\", \"ui_actions\": []}",
            "   ui_actions can include:",
            "     - {\"action_type\": \"UPDATE_TEXT_CONTENT\", \"target_element_id\": \"roxStudentStatusSummary\", \"parameters\": {\"text\": \"<HTML or text summary for UI>\"}}",
            "     - {\"action_type\": \"DISPLAY_CHART\", \"chart_type\": \"line_graph\", \"data\": [ARRAY_OF_DATA_POINTS], \"title\": \"<Chart Title>\"} (e.g., data from interaction_history if it contains scores over time)",
            "     - {\"action_type\": \"SHOW_BUTTON_OPTIONS\", \"target_element_id\": \"p1ActionButtons\", \"parameters\": {\"buttons\": [{\"label\": \"Practice [Skill]\", \"next_graph_hint\": \"NODE_PRACTICE_SELECTOR_[SKILL]\"}]}}",
            "Ensure the 'data' for DISPLAY_CHART is an array of objects if you include it."
        ]
        prompt = "\n".join(prompt_parts)

        logger.debug(f"Progress Reporter Prompt:\n{prompt}")
        response = await model.generate_content_async(prompt)
        
        logger.debug(f"Progress Reporter LLM Raw Response: {response.text}")
        response_json = json.loads(response.text)

        return {
            "output_content": {
                "text_for_tts": response_json.get("text_for_tts", "Let's look at how you're doing!"),
                "ui_actions": response_json.get("ui_actions", [])
            }
            # No next_node_hint needed here as it will likely flow to a standard handler
        }

    except Exception as e:
        logger.error(f"Error in progress_reporter_node: {e}", exc_info=True)
        return {
            "output_content": {
                "text_for_tts": "I tried to check your progress, but something went wrong. Let's try again a bit later?",
                "ui_actions": []
            }
        }
