import logging
from typing import Dict, Any, List, Optional
import pandas as pd # For pd.notna and potentially other pandas operations if CSV fields are complex

from state import AgentGraphState
# Assuming an LLM utility might be added later for actual calls:
# from backend_ai_service_langgraph.utils.llm_utils import invoke_llm_for_text, invoke_llm_for_json

logger = logging.getLogger(__name__)

async def teaching_delivery_node(state: AgentGraphState) -> Dict[str, Any]:
    logger.info("TeachingDeliveryNode: Processing started.")

    retrieved_row: Optional[Dict[str, Any]] = state.get("retrieved_teaching_row")
    rag_error: Optional[str] = state.get("rag_error")
    current_lesson_step_number: int = state.get("current_lesson_step_number", 1) # Default to step 1
    current_affective_state: str = state.get('current_student_affective_state', 'NEUTRAL').upper()

    if rag_error or not retrieved_row:
        error_msg = f"TeachingDeliveryNode: Cannot proceed. RAG error: '{rag_error}', Retrieved row is None: {retrieved_row is None}."
        logger.error(error_msg)
        return {
            "teaching_output_content": {
                "text_for_tts": "I'm sorry, I encountered an issue retrieving the teaching material for this step.",
                "ui_actions": [],
                "error": error_msg
            }
        }

    logger.info(f"TeachingDeliveryNode: Retrieved row keys: {list(retrieved_row.keys())}")
    logger.info(f"TeachingDeliveryNode: Current lesson step: {current_lesson_step_number}, Affective state: {current_affective_state}")

    # Helper function to parse multi-step content from a CSV field
    def get_content_for_step(field_name: str, default_value: str = "Content not available for this step.") -> str:
        raw_content = retrieved_row.get(field_name)
        
        if pd.isna(raw_content) or raw_content == "": # Handle NaN or empty strings explicitly
            logger.warning(f"TeachingDeliveryNode: Field '{field_name}' is empty or NaN.")
            return default_value
        
        if not isinstance(raw_content, str):
            logger.warning(f"TeachingDeliveryNode: Field '{field_name}' is not a string ({type(raw_content)}), using as is or default.")
            return str(raw_content) # Attempt to convert to string, or return default if it's problematic

        if "|" in raw_content:
            parts = [part.strip() for part in raw_content.split("|")]
            if 0 < current_lesson_step_number <= len(parts):
                selected_part = parts[current_lesson_step_number - 1]
                logger.info(f"TeachingDeliveryNode: For field '{field_name}', selected step {current_lesson_step_number}: '{selected_part[:50]}...' ")
                return selected_part
            else:
                logger.warning(f"TeachingDeliveryNode: Step {current_lesson_step_number} out of range for '{field_name}' (1 to {len(parts)}). Using first part ('{parts[0][:50]}...') or default.")
                return parts[0] if parts else default_value # Fallback to first part if available
        else:
            # Field does not contain a delimiter, so use the whole content for any step
            logger.info(f"TeachingDeliveryNode: Field '{field_name}' has no delimiter, using full content: '{raw_content[:50]}...' ")
            return raw_content

    # Extract data using the helper, providing defaults
    teacher_persona = str(retrieved_row.get("TEACHER_PERSONAS", "Default Teacher"))
    learning_objective = str(retrieved_row.get("LEARNING_OBJECTIVE", "Unknown Objective"))
    student_proficiency = str(retrieved_row.get("STUDENT_PROFICIENCY", "Unknown Proficiency"))

    core_explanation = get_content_for_step("CORE_EXPLANATION_STRATEGY")
    key_examples = get_content_for_step("KEY_EXAMPLES")
    visual_aid_desc = get_content_for_step("VISUAL_AID_STRATEGY")
    common_misconceptions = get_content_for_step("COMMON_MISCONCEPTIONS")
    comprehension_check_q = get_content_for_step("COMPREHENSION_CHECK")

    # Affective state adaptation: Read directly from the dedicated column
    affective_adaptation_strategy = str(retrieved_row.get("ADAPTATION_FOR_OVERALL_STUDENT_AFFECTIVE_STATE", ""))
    if pd.isna(retrieved_row.get("ADAPTATION_FOR_OVERALL_STUDENT_AFFECTIVE_STATE")):
        affective_adaptation_strategy = "" # Ensure it's an empty string if NaN
    if affective_adaptation_strategy:
        logger.info(f"TeachingDeliveryNode: Found overall affective adaptation strategy: '{affective_adaptation_strategy[:100]}...'")
    else:
        logger.warning(f"TeachingDeliveryNode: No overall affective adaptation strategy found in the retrieved row.")

    # --- LLM PROMPT CONSTRUCTION (Placeholders for now) ---
    # This is where you'd carefully craft prompts for your LLM (e.g., Gemini)
    
    # TTS Prompt Construction (Placeholder)
    # In a real scenario, this prompt would be sent to an LLM.
    # For now, we'll simulate the TTS text directly from the parsed content.
    # We prepend the affective adaptation strategy to the core lesson content.
    simulated_tts_text = f"{affective_adaptation_strategy} {core_explanation}".strip()
    if key_examples and key_examples != "Content not available for this step.":
        simulated_tts_text += f" For example: {key_examples}."
    if common_misconceptions and common_misconceptions != "Content not available for this step.":
        simulated_tts_text += f" A common mistake to avoid is: {common_misconceptions}."
    if comprehension_check_q and comprehension_check_q != "Content not available for this step.":
        simulated_tts_text += f" To check your understanding, think about this: {comprehension_check_q}"

    logger.info(f"TeachingDeliveryNode: Simulated TTS text generated (first 150 chars): {simulated_tts_text[:150]}")

    # Visual Aid Command Generation (Placeholder)
    # In a real scenario, visual_aid_desc (or a more detailed prompt) would go to an LLM
    # that returns structured JSON for UI actions.
    ui_actions_for_visuals: List[Dict[str, Any]] = []
    if visual_aid_desc and visual_aid_desc != "Content not available for this step.":
        # Simulate a UI action based on the description
        ui_actions_for_visuals.append({
            "action_type": "DISPLAY_VISUAL_AID", 
            "visual_description": visual_aid_desc,
            "details": {
                "source": "teaching_delivery_node",
                "suggestion_type": "parsed_from_csv"
            }
        })
        logger.info(f"TeachingDeliveryNode: Generated UI action for visual: {visual_aid_desc[:50]}...")
    else:
        logger.info("TeachingDeliveryNode: No visual aid description provided or applicable for this step.")

    # --- Prepare teaching_output_content --- 
    output_content = {
        "text_for_tts": simulated_tts_text.strip(),
        "ui_actions": ui_actions_for_visuals,
        "lesson_step_details": { # For debugging and richer state logging
            "teacher_persona": teacher_persona,
            "learning_objective": learning_objective,
            "student_proficiency": student_proficiency,
            "current_lesson_step_number": current_lesson_step_number,
            "current_affective_state": current_affective_state,
            "parsed_core_explanation": core_explanation,
            "parsed_key_examples": key_examples,
            "parsed_visual_aid_desc": visual_aid_desc,
            "parsed_common_misconceptions": common_misconceptions,
            "parsed_comprehension_check_q": comprehension_check_q,
            "parsed_affective_adaptation_strategy": affective_adaptation_strategy,
            "raw_retrieved_row_preview": {k: str(v)[:50] + '...' if isinstance(v, str) and len(str(v)) > 50 else str(v) for k, v in retrieved_row.items()} # Preview of raw data
        }
    }
    
    logger.info(f"TeachingDeliveryNode: Processing complete. Output TTS preview: {output_content['text_for_tts'][:100]}... UI Actions count: {len(output_content['ui_actions'])}")
    
    return {"teaching_output_content": output_content}
