import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

# Define teacher personas
TEACHER_PERSONAS = {
    "structuralist": {
        "description": "Focuses on organization and structure of language and responses",
        "tone": "clear, organized, systematic",
        "emphasis": "structure, organization, clarity"
    },
    "nurturer": {
        "description": "Emphasizes encouragement and building confidence",
        "tone": "warm, supportive, encouraging",
        "emphasis": "progress, effort, growth mindset"
    },
    "challenger": {
        "description": "Pushes students to think critically and deeply",
        "tone": "direct, thought-provoking, analytical",
        "emphasis": "critical thinking, deeper analysis, precision"
    },
    "socratic": {
        "description": "Uses questions to guide learning and discovery",
        "tone": "inquisitive, thoughtful, guiding",
        "emphasis": "self-discovery, reflection, guided inquiry"
    },
    "Rox_Welcoming_Guide": {
        "description": "Provides a warm welcome, gives a brief status, and suggests a starting point.",
        "tone": "friendly, encouraging, clear",
        "emphasis": "welcome, orientation, getting_started"
    }
}

async def apply_teacher_persona_node(state: AgentGraphState) -> dict:
    """Sets the current AI teacher persona in the graph state."""
    context = state.get("current_context")
    student_data = state.get("student_memory_context", {})
    diagnosis = state.get("diagnosis_result", {})
    affective_state = state.get("student_affective_state", {})
    
    # Default persona
    selected_persona_name = "nurturer" # Default to nurturer

    # Check for ROX_WELCOME_INIT first
    if context and getattr(context, 'task_stage', None) == "ROX_WELCOME_INIT":
        selected_persona_name = "Rox_Welcoming_Guide"
        logger.info(f"PerspectiveShaperNode: Applying Rox_Welcoming_Guide for ROX_WELCOME_INIT.")
    else:
        # Logic to select appropriate persona based on context and student state
        if diagnosis and diagnosis.get("needs_structure"):
            selected_persona_name = "structuralist"
        elif diagnosis and diagnosis.get("needs_challenge"):
            selected_persona_name = "challenger"
        elif diagnosis and diagnosis.get("needs_guidance"):
            selected_persona_name = "socratic"
        
        # Consider student's affective state
        if affective_state and affective_state.get("primary_emotion") == "frustrated":
            # When student is frustrated, nurturer is often best
            selected_persona_name = "nurturer"
        
        # Get student's preferred persona if available (overrides other selections, unless ROX_WELCOME_INIT)
        if student_data:
            student_preferred_persona = student_data.get("profile", {}).get("preferred_teacher_persona")
            if student_preferred_persona and student_preferred_persona in TEACHER_PERSONAS:
                selected_persona_name = student_preferred_persona
                logger.info(f"PerspectiveShaperNode: Using student's preferred persona: {selected_persona_name}")
    
    logger.info(f"PerspectiveShaperNode: Applied teacher persona: {selected_persona_name}")
    
    # Get the details of the selected persona
    persona_details_template = TEACHER_PERSONAS.get(selected_persona_name, TEACHER_PERSONAS["nurturer"])
    # Create a copy to avoid modifying the original template and add the name
    active_persona_details = persona_details_template.copy()
    active_persona_details["name"] = selected_persona_name
    
    return {
        "active_persona_details": active_persona_details
    }