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
    }
}

async def apply_teacher_persona_node(state: AgentGraphState) -> dict:
    """Sets the current AI teacher persona in the graph state."""
    context = state.get("current_context")
    student_data = state.get("student_memory_context", {})
    diagnosis = state.get("diagnosis_result", {})
    affective_state = state.get("student_affective_state", {})
    
    # Default persona
    persona = "nurturer"
    
    # Logic to select appropriate persona based on context and student state
    if diagnosis.get("needs_structure"):
        persona = "structuralist"
    elif diagnosis.get("needs_challenge"):
        persona = "challenger"
    elif diagnosis.get("needs_guidance"):
        persona = "socratic"
    
    # Consider student's affective state
    if affective_state.get("primary_emotion") == "frustrated":
        # When student is frustrated, nurturer is often best
        persona = "nurturer"
    
    # Get student's preferred persona if available (overrides other selections)
    student_preferred_persona = student_data.get("profile", {}).get("preferred_teacher_persona")
    if student_preferred_persona and student_preferred_persona in TEACHER_PERSONAS:
        persona = student_preferred_persona
        logger.info(f"PerspectiveShaperNode: Using student's preferred persona: {persona}")
    
    logger.info(f"PerspectiveShaperNode: Applied teacher persona: {persona}")
    
    # Return the selected persona and its details
    return {
        "current_teacher_persona": persona,
        "persona_details": TEACHER_PERSONAS.get(persona, TEACHER_PERSONAS["nurturer"])
    }