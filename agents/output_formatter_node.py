# agents/output_formatter_node.py
import logging
from state import AgentGraphState
from models import ReactUIAction # Assuming your Pydantic model

logger = logging.getLogger(__name__)

async def format_final_output_for_client_node(state: AgentGraphState) -> dict:
    user_id = state.get("user_id", "unknown_user")
    task_stage = state.get("current_context").task_stage if state.get("current_context") else "UNKNOWN_STAGE"
    logger.info(f"OutputFormatterNode: Preparing final output for user '{user_id}', task_stage '{task_stage}'")

    # Initialize defaults for final output structure
    final_tts = "I'm ready for your next instruction."
    final_ui_actions: list[dict] = [] # Will be list of ReactUIAction-like dicts
    final_next_task = state.get("next_task_details") # Pass through if set by CurriculumNavigator
    final_nav_instruction = None # Usually set by routers or specific action nodes

    # --- Specific Formatting for P1 Welcome Flow ---
    if task_stage == "ROX_WELCOME_INIT":
        # These keys are expected to be set by preceding nodes in the P1 welcome flow
        greeting_tts = state.get("greeting_tts_intermediate", "")
        suggestion_tts = state.get("task_suggestion_tts_intermediate", "") # Key from curriculum_navigator

        # Combine TTS parts for a single, coherent AI utterance
        tts_parts = [part for part in [greeting_tts.strip(), suggestion_tts.strip()] if part]
        final_tts = " ".join(tts_parts) if tts_parts else "Welcome! Let's get started."

        # Combine UI actions
        # greeting_ui_actions_intermediate is expected to be an empty list from generate_welcome_greeting_node
        final_ui_actions.extend(state.get("greeting_ui_actions_intermediate", []))
        # task_suggestion_ui_actions_intermediate is from curriculum_navigator_node
        final_ui_actions.extend(state.get("task_suggestion_ui_actions_intermediate", []))
        
        # final_next_task is already populated from state.get("next_task_details") at the beginning

    # --- Specific Formatting for P2 Speaking Submission Acknowledgment & P6 Navigation ---
    elif task_stage == "SPEAKING_TESTING_SUBMITTED":
        # This assumes a node like 'GenerateSpeakingFeedbackStubNode' or a 'PrepareFeedbackDisplayNode'
        # has prepared the initial content for P6 and a navigation instruction.
        # Let's say 'generate_speaking_feedback_stub_node' put its output into 'feedback_stub_output'
        feedback_stub_output = state.get("output_from_speaking_feedback_stub", {}) # Check this key
        final_tts = feedback_stub_output.get("text_for_tts", "Your speaking feedback is being prepared.")
        final_ui_actions = feedback_stub_output.get("ui_actions", []) # Should include NAVIGATE_TO_PAGE for P6
        
        # Example: ensure navigation is part of ui_actions or set separately
        # if not any(ua.get("action_type") == "NAVIGATE_TO_PAGE" for ua in final_ui_actions):
        #    final_ui_actions.append({
        #        "action_type": "NAVIGATE_TO_PAGE",
        #        "parameters": {"page_name": "P6_SpeakingFeedback", "data_for_page": state.get("data_for_p6_page")}
        #    })
        # Or set final_navigation_instruction
        final_nav_instruction = {"page_name": "P6_SpeakingFeedback", "data_for_page": state.get("data_for_p6_page")}


    # --- Formatting for General Conversational Turns (from ConversationalTurnManagerNode) ---
    elif task_stage in ["ROX_CONVERSATION_TURN", "SPEAKING_FEEDBACK_QA", "TEACHING_PAGE_QA", "AI_MODELING_QA"]:
        # Assuming ConversationalTurnManagerNode sets a field like 'conversational_response_content'
        conv_output = state.get("output_from_conversational_manager", {}) # Node should set this
        final_tts = conv_output.get("text_for_tts", "Okay, I understand.")
        final_ui_actions = conv_output.get("ui_actions", [])
        # Any next_task or navigation would typically be decided by a router *after* this conversational turn.

    # ... other elif blocks for other final states that need specific formatting ...
    
    else:
        logger.warning(f"OutputFormatterNode: No specific formatting logic for task_stage '{task_stage}'. Using defaults or pass-through.")
        # Fallback to checking if a previous node already fully formed 'output_content'
        existing_output_content = state.get("output_content") # A well-behaved node might set this directly
        if isinstance(existing_output_content, dict):
            final_tts = existing_output_content.get("text_for_tts", final_tts)
            final_ui_actions = existing_output_content.get("ui_actions", final_ui_actions)
        

    # This node populates the consistently named final fields that app.py will use
    return {
        "final_response_text_for_api": final_tts.strip(),
        "final_ui_actions_for_api": final_ui_actions,
        "final_next_task_info_for_api": final_next_task, 
        "final_navigation_instruction_for_api": final_nav_instruction
    }