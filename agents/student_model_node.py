import logging
from state import AgentGraphState
from memory import mem0_memory

logger = logging.getLogger(__name__)

async def load_or_initialize_student_profile(state: AgentGraphState) -> dict:
    user_id = state["user_id"]
    logger.info(f"StudentModelNode: Loading profile for user_id: '{user_id}'")
    student_data = mem0_memory.get_student_data(user_id)
    return {"student_memory_context": student_data}

async def update_student_skills_after_diagnosis(state: AgentGraphState) -> dict:
    user_id = state["user_id"]
    diagnosis = state.get("diagnosis_result", {})
    student_data = state.get("student_memory_context", {})
    
    # update skills based on diagnosis
    profile = student_data.get("profile", {})
    skills = profile.get("skills", {})
    
    if "speaking" in diagnosis:
        for skill, score in diagnosis["speaking"].get("skill_scores", {}).items():
            skills[f"speaking_{skill}"] = score
    
    # update the profile
    profile["skills"] = skills
    student_data["profile"] = profile
    
    # save to memory
    mem0_memory.update_student_profile(user_id, profile)
    
    return {"student_memory_context": student_data}

async def log_interaction_to_memory(state: AgentGraphState) -> dict:
    user_id = state["user_id"]
    interaction_data = {
        "transcript": state.get("transcript"),
        "diagnosis": state.get("diagnosis_result"),
        "feedback": state.get("feedback_content", {}).get("text"),
        "task_details": state.get("current_context").dict() if state.get("current_context") else {},
        "timestamp": "AUTO_TIMESTAMP"  # Mem0 can handle this
    }
    
    mem0_memory.add_interaction_to_history(user_id, interaction_data)
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