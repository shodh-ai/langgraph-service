import logging
from state import AgentGraphState
from memory import mem0_memory
from utils.db_utils import fetch_user_by_id, fetch_user_skills

logger = logging.getLogger(__name__)

async def load_or_initialize_student_profile(state: AgentGraphState) -> dict:
    user_id = state["user_id"]
    logger.info(f"StudentModelNode: Loading profile for user_id: '{user_id}'")
    
    student_data = mem0_memory.get_student_data(user_id)
    
    # Attempt to fetch from DB if mem0 profile is incomplete or default-named
    profile_from_mem0 = student_data.get("profile", {})
    if not profile_from_mem0 or profile_from_mem0.get("name", "").startswith("New Student") or not profile_from_mem0.get("name"): # Added check for empty name
        logger.info(f"StudentModelNode: Profile from mem0 is incomplete or default. Attempting DB fetch for user_id: '{user_id}'")
        try:
            db_user_data = fetch_user_by_id(user_id) # This function needs to be defined or imported
            
            if db_user_data:
                user_skills = fetch_user_skills(user_id) # This function needs to be defined or imported
                
                profile = {
                    "name": f"{db_user_data.get('firstName', '')} {db_user_data.get('lastName', '')}".strip(),
                    "occupation": db_user_data.get('occupation', ''),
                    "major": db_user_data.get('major', ''),
                    "native_language": db_user_data.get('nativeLanguage', ''),
                    "level": "Beginner", 
                    "skills": user_skills or {}
                }
                
                if db_user_data.get('createdAt'):
                    profile["account_created"] = db_user_data.get('createdAt').isoformat()
                
                # Ensure name is not empty after DB fetch
                if not profile.get("name"):
                    logger.warning(f"StudentModelNode: DB fetch for user_id '{user_id}' resulted in an empty name. Will use default.")
                    # Let it fall through to the default Harshit profile logic
                else:
                    student_data["profile"] = profile
                    mem0_memory.update_student_profile(user_id, profile)
                    logger.info(f"StudentModelNode: Updated profile from PostgreSQL for user_id: '{user_id}'")

            else:
                logger.warning(f"StudentModelNode: No user found in PostgreSQL for user_id: '{user_id}'.")
        except Exception as e:
            logger.error(f"StudentModelNode: Error fetching user data from PostgreSQL: {e}")

    # Fallback to default "Harshit" profile if still no valid name
    current_profile = student_data.get("profile", {})
    if not current_profile.get("name") or current_profile.get("name").startswith("New Student"):
        logger.warning(f"StudentModelNode: No valid student name found for user_id '{user_id}'. Using default 'Harshit' profile.")
        student_data["profile"] = {
            "first_name": "Harshit", # Changed from "name"
            "last_name": "",         # Added for completeness
            "occupation": "Student",
            "major": "Computer Science",
            "native_language": "Hindi",
            "level": "Intermediate",
            "skills": {"speaking_fluency": 70, "grammar_sva": 65}
        }
        # Optionally, update mem0 with this default if it's the first time
        # mem0_memory.update_student_profile(user_id, student_data["profile"])

    return {"student_memory_context": student_data}

async def update_student_skills_after_diagnosis(state: AgentGraphState) -> dict:
    user_id = state["user_id"]
    diagnosis = state.get("diagnosis_result", {})
    student_data = state.get("student_memory_context", {})
    
    # update skills based on diagnosis
    profile = student_data.get("profile", {})
    skills = profile.get("skills", {})
    
    if diagnosis and "speaking" in diagnosis:
        for skill, score in diagnosis["speaking"].get("skill_scores", {}).items():
            skills[f"speaking_{skill}"] = score
    
    profile["skills"] = skills
    
    mem0_memory.update_student_profile(user_id, profile)
    
    student_data["profile"] = profile
    return {"student_memory_context": student_data}

async def log_interaction_to_memory(state: AgentGraphState) -> dict:
    user_id = state["user_id"]
    
    # Extract relevant information from the state to log
    interaction_summary = {
        "task_type": state.get("task_type", ""),
        "transcript": state.get("transcript", ""),
        "feedback": state.get("feedback", ""),
        "timestamp": state.get("timestamp", "")
    }
    
    if "diagnosis_result" in state:
        interaction_summary["diagnosis"] = state["diagnosis_result"]
    
    # Log to memory
    mem0_memory.add_interaction_to_history(user_id, interaction_summary)
    
    logger.info(f"StudentModelNode: Logged interaction for user_id: '{user_id}'")
    return {}

async def save_generated_notes_to_memory(state: AgentGraphState) -> dict:
    user_id = state["user_id"]
    notes = state.get("session_notes", {})
    
    if notes:
        mem0_memory.add_interaction_to_history(user_id, {
            "type": "session_notes",
            "content": notes,
            "timestamp": "AUTO_TIMESTAMP"
        })
    
    return {}

async def get_student_affective_state(state: AgentGraphState) -> dict:
    transcript = state.get("transcript", "")
    history = state.get("chat_history", [])
    
    affect = {"primary_emotion": "neutral", "confidence": 0.7}
    
    if "frustrated" in transcript.lower() or "confused" in transcript.lower():
        affect = {"primary_emotion": "frustrated", "confidence": 0.8}
    elif "happy" in transcript.lower() or "understand" in transcript.lower():
        affect = {"primary_emotion": "satisfied", "confidence": 0.8}
    
    return {"student_affective_state": affect}