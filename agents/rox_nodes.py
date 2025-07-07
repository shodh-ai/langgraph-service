# agents/rox_nodes.py
import logging
import json
from typing import Dict, Any
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from state import AgentGraphState
# Assume knowledge_client is available for fetching task titles etc.
from knowledge.knowledge_client import knowledge_client

logger = logging.getLogger(__name__)
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def rox_welcome_and_reflection_node(state: AgentGraphState) -> dict:
    """
    Generates a welcome message. Handles the initial page load proactive greeting.
    """
    logger.info("--- Executing Rox Welcome & Reflection Node ---")

    # This node is triggered on page load, so transcript is usually empty.
    # We can fetch the student's name from the memory context if it was loaded.
    student_profile = state.get("student_memory_context", {})
    student_name = student_profile.get("name", "there") # Fallback to "there"

    # The welcome message the AI will speak.
    welcome_speech = f"Hello {student_name}! I'm Rox, your AI TOEFL tutor. My job is to guide you on a personalized learning path. What would you like to focus on today?"

    # THIS IS THE IMPLEMENTATION OF STEP 1
    # We create the final output for the client.
    return {
        "final_text_for_tts": welcome_speech,
        "final_ui_actions": [
            {
                # This action tells the frontend to update the h1 tag or a specific div.
                "action_type": "UPDATE_TEXT_CONTENT",
                "target_element_id": "roxWelcomeMessage",
                "parameters": {
                    "text": welcome_speech
                }
            },
            {
                # This action tells the agent session to listen for the user's response.
                "action_type": "SPEAK_THEN_LISTEN",
                "parameters": {
                    "text_to_speak": welcome_speech,
                    "listen_config": {
                        "silence_timeout_s": 10.0
                    }
                }
            }
        ],
        "is_simple_greeting": False # Ensure we don't follow the old test path
    }


async def rox_propose_plan_node(state: AgentGraphState) -> dict:
    """
    Takes the output of the CurriculumNavigatorNode and presents the plan to the student.
    """
    logger.info("--- Executing Rox Propose Plan Node ---")
    
    # current_lo_to_address was set by curriculum_navigator_node
    next_lo_info = state.get("current_lo_to_address", {})
    lo_title = next_lo_info.get("title", "the next topic")
    
    # Here you could fetch the full LO Tree from knowledge base to explain the map
    # curriculum_map = await knowledge_client.get_full_curriculum_map()
    # map_explanation = "Our curriculum map shows that after mastering X, the next step is Y..." # LLM could generate this
    
    proposal_text = f"Based on your progress, our curriculum map suggests the best thing to focus on now is '{lo_title}'. This is a key skill for improving your overall writing score. Are you ready to start this lesson?"

    # The UI Action to display the plan/map would be generated here
    ui_actions = [{
        "action_type": "DISPLAY_CURRICULUM_MAP",
        "parameters": {
            "full_map_data": {}, # The full curriculum map object
            "current_lo_to_highlight": next_lo_info.get("id")
        }
    }]
    
    # Must pass the LO forward so the next node (the planner) can use it.
    return {
        "output_content": {"text_for_tts": proposal_text, "ui_actions": ui_actions},
        "last_action_was": "CURRICULUM_PROPOSAL", # Flag for saving this interaction
        "current_lo_to_address": next_lo_info
    }

async def rox_nlu_and_qa_handler_node(state: AgentGraphState) -> dict:
    """Handles natural language questions and intent classification on the Rox page."""
    logger.info("---Executing Rox NLU & QA Handler Node---")

    transcript = state.get("transcript", "").lower().strip()

    # --- NEW LOGIC FOR HANDLING SPECIFIC COMMANDS ---
    if "course map" in transcript or "show me the map" in transcript:
        logger.info("Rox NLU: Detected intent to view the course map.")
        # The key to update is `student_intent_for_rox_turn` because
        # that's what the `after_proposal_router` in rox_flow.py uses.
        return {
            "student_intent_for_rox_turn": "REQUEST_COURSE_MAP",
            "output_content": { # We can still provide immediate TTS
                "text_for_tts": "Of course. Here is your course map.",
                "ui_actions": []
            }
        }

    # Your existing logic for other intents
    intent = "ACKNOWLEDGE"
    response_text = "Okay, let's move on."

    if "what should we do" in transcript or "what's next" in transcript:
        intent = "REQUEST_ALTERNATIVE_TASK"
        response_text = "Good question. Let me check the curriculum for you."
    elif "yes" in transcript or "sounds good" in transcript or "okay" in transcript or "let's do it" in transcript:
        intent = "CONFIRM_START_TASK"
        response_text = "Great! I'm setting up the first task for you now."
    else:
        logger.info("Input not a standard command, treating as a QA query.")
        response_text = f"That's an interesting question. I'll make a note of it."
        intent = "ASKED_QUESTION"

    logger.info(f"Rox NLU classified intent as: {intent}")
    return {
        "student_intent_for_rox_turn": intent,
        "output_content": {
            "text_for_tts": response_text,
            "ui_actions": []
        }
    }

async def rox_navigate_to_task_node(state: AgentGraphState) -> dict:
    """Prepares the state to navigate the user to their confirmed task."""
    logger.info("--- Executing Rox Navigate to Task Node ---")
    
    # The pedagogical_plan should have been created by a planner node by now
    plan = state.get("pedagogical_plan")
    if not plan:
        # Fallback if plan wasn't created - should not happen in happy path
        return {"output_content": {"text_for_tts": "Looks like there was an issue planning your next task. Let's try again."}}
        
    first_step = plan[0]
    modality = first_step.get("modality")
    target_page = first_step.get("target_page") # e.g., P7, P8
    details = first_step.get("details", {})
    
    response_text = f"Great! Taking you to your {modality} session on '{details.get('focus')}' now."

    # This UI action will trigger the frontend router
    ui_actions = [{
        "action_type": "NAVIGATE_TO_PAGE",
        "parameters": {
            "page_name": target_page,
            "data_for_page": details # Pass all details the new page will need
        }
    }]
    
    return {"output_content": {"text_for_tts": response_text, "ui_actions": ui_actions}}

async def show_course_map_node(state: AgentGraphState) -> dict:
    """
    Generates the UI action to display the course map on the frontend.
    """
    logger.info("--- Executing Show Course Map Node ---")

    # This node's only job is to create the UI action.
    # The TTS was already handled by the NLU node.
    return {
        "final_ui_actions": [
            {
                "action_type": "EXECUTE_CLIENT_ACTION",
                "parameters": {
                    "action_name": "showCourseMap"
                }
            }
        ]
    }