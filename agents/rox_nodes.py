# agents/rox_nodes.py
import logging
import json
from state import AgentGraphState

# This import makes the cache accessible to the node.
# This assumes your FastAPI app object is defined in a file named `app.py` at the root.
# Adjust the import path if necessary.
from cache import SESSION_DATA_CACHE
# ... other imports

logger = logging.getLogger(__name__)

async def rox_welcome_and_reflection_node(state: AgentGraphState) -> dict:
    """
    Generates a welcome message and ALWAYS returns a SPEAK_THEN_LISTEN action
    to create a proper conversational turn.
    """
    logger.info("--- Executing Rox Welcome & Reflection Node ---")

    student_profile = state.get("student_memory_context", {})
    student_name = student_profile.get("name", "there")

    # This can be your standard greeting.
    welcome_speech = f"Hello {student_name}! I'm Rox, your AI TOEFL tutor. It's great to see you. Let's get started."
    
    logger.info(f"Generated welcome speech: '{welcome_speech}'")

    # THIS IS THE FIX: We always return the complete action for this turn.
    return {
        "final_text_for_tts": welcome_speech,
        "final_ui_actions": [
            {
                # This action tells the agent to speak and then listen for the user's reply.
                # This is what creates the conversational turn.
                "action_type": "SPEAK_THEN_LISTEN",
                "parameters": {
                    "text_to_speak": welcome_speech,
                    "listen_config": {
                        # A reasonable timeout to wait for the user to say "okay" or ask a question.
                        "silence_timeout_s": 10.0,
                        # <<< THIS IS THE FIX >>>
                        # Tell the agent that the user's reply belongs to the Rox flow.
                        "context_for_next_turn": "ROX_CONVERSATION_TURN"
                    }
                }
            }
        ],
        "is_simple_greeting": False, # Flag that this was not the simple test
        "last_action_was": "PROACTIVE_GREETING" # Add this for better logging in the save_node
    }


async def rox_propose_plan_node(state: AgentGraphState) -> dict:
    """
    Generates map data, SAVES it to a cache, and sends a LIGHTWEIGHT
    command to the frontend to fetch it.
    """
    logger.info("--- Executing Rox Propose Plan Node (with Caching) ---")
    
    intent = state.get("student_intent_for_rox_turn")
    next_lo_info = state.get("current_lo_to_address", {})
    course_map_data = state.get("course_map_data")
    session_id = state.get("session_id")

    # --- Add validation ---
    if not course_map_data:
        logger.error("CRITICAL FALLBACK: Proposer node was called but 'course_map_data' is missing from state.")
        return {"final_text_for_tts": "I'm sorry, I seem to have lost my train of thought. Could you ask that again?"}
    if not session_id:
        logger.error("CRITICAL FALLBACK: Proposer node was called but 'session_id' is missing from state.")
        return {"final_text_for_tts": "I'm having trouble with our connection. Let's try to refresh."}
    
    # --- THIS IS THE FIX ---
    # 1. Save the large data to the cache.
    if session_id not in SESSION_DATA_CACHE:
        SESSION_DATA_CACHE[session_id] = {}
    SESSION_DATA_CACHE[session_id]["course_map"] = course_map_data
    # Add a log to be 100% sure it was saved
    logger.info(f"Saved 'course_map' data to cache for session '{session_id}'. Cache now has keys: {list(SESSION_DATA_CACHE[session_id].keys())}")
    
    # 2. Prepare the speech
    speech = next_lo_info.get("proposal_script", "I've found our next lesson.")
    if intent == "REQUEST_COURSE_MAP":
        speech = "Of course. Pulling up your learning map now."
        
    # 3. Create the lightweight UI Action
    return {
        "final_text_for_tts": speech,
        "final_ui_actions": [
            {
                # This is the command for the frontend.
                "action_type": "FETCH_AND_DISPLAY_COURSE_MAP",
                "parameters": {
                    # We send the identifiers the frontend needs to build the fetch URL.
                    "session_id": session_id,
                    "data_key": "course_map"
                }
            },
            {
                "action_type": "SPEAK_THEN_LISTEN",
                "parameters": { 
                    "text_to_speak": speech,
                    "listen_config": {
                        "context_for_next_turn": "ROX_CONVERSATION_TURN"
                    }
                }
            }
        ],
        "last_action_was": "CURRICULUM_PROPOSAL",
    }


# ENHANCED NODE: The NLU handler now understands more intents.
async def rox_nlu_and_qa_handler_node(state: AgentGraphState) -> dict:
    """Handles natural language questions and intent classification on the Rox page."""
    logger.info("---Executing Rox NLU & QA Handler Node---")
    transcript = (state.get("transcript") or "").lower().strip()
    intent = "GENERAL_QUESTION" # Default intent

    # Rule-based intent classification
    if "course map" in transcript or "show me the map" in transcript:
        intent = "REQUEST_COURSE_MAP"
    elif any(phrase in transcript for phrase in ["yes", "okay", "sure", "sounds good", "let's do it"]):
        intent = "CONFIRM_START_TASK"
    elif any(phrase in transcript for phrase in ["no", "something else", "not that", "what else"]):
        intent = "REJECT_OR_QUESTION_LO"
    
    logger.info(f"Rox NLU classified intent as: {intent}")
    
    # We only update the intent. The next node in the graph will handle the response.
    return {"student_intent_for_rox_turn": intent}


# This node prepares the final navigation action.
async def rox_navigate_to_task_node(state: AgentGraphState) -> dict:
    """Prepares the state to navigate the user to their confirmed task.""" 
    logger.info("--- Executing Rox Navigate to Task Node ---")
    plan = state.get("pedagogical_plan")
    if not plan:
        return { "final_text_for_tts": "Looks like there was an issue planning your next task. Let's try again." }
        
    first_step = plan[0]
    modality = first_step.get("modality")
    target_page = first_step.get("target_page") # e.g., "P7", "P8"
    
    response_text = f"Great! I've created a plan for '{first_step.get('focus')}'. Taking you to your {modality} session now."

    return {
        "final_text_for_tts": response_text,
        "final_ui_actions": [{
            "action_type": "NAVIGATE_TO_PAGE",
            "parameters": {
                "page_name": target_page,
                "data_for_page": first_step 
            }
        }]
    }
