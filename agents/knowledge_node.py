import logging
import yaml
import os
from state import AgentGraphState

logger = logging.getLogger(__name__)

# Load knowledge content
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "knowledge_content.yaml")

# Initialize with empty data, will attempt to load from file
KNOWLEDGE_BASE = {}

# Try to load the knowledge content if the file exists
try:
    if os.path.exists(KNOWLEDGE_BASE_PATH):
        with open(KNOWLEDGE_BASE_PATH, 'r') as file:
            KNOWLEDGE_BASE = yaml.safe_load(file)
        logger.info(f"KnowledgeNode: Loaded content from {KNOWLEDGE_BASE_PATH}")
    else:
        logger.warning(f"KnowledgeNode: Knowledge content file not found at {KNOWLEDGE_BASE_PATH}")
except Exception as e:
    logger.error(f"KnowledgeNode: Failed to load knowledge content: {e}")

async def get_task_prompt_node(state: AgentGraphState) -> dict:
    """Fetches prompt for current_context.current_prompt_id."""
    context = state.get("current_context")
    prompt_id = getattr(context, "current_prompt_id", None) if context else None
    
    if not prompt_id:
        logger.info("KnowledgeNode: No prompt_id provided in context")
        return {"task_prompt": None}
    
    prompt = KNOWLEDGE_BASE.get("prompts", {}).get(prompt_id)
    logger.info(f"KnowledgeNode: Retrieved prompt for ID: {prompt_id}")
    
    return {"task_prompt": prompt}

async def get_teaching_material_node(state: AgentGraphState) -> dict:
    """Fetches content for a specific lesson."""
    context = state.get("current_context")
    lesson_id = getattr(context, "lesson_id", None) if context else None
    
    if not lesson_id:
        logger.info("KnowledgeNode: No lesson_id provided in context")
        return {"teaching_material": None}
    
    material = KNOWLEDGE_BASE.get("teaching_materials", {}).get(lesson_id)
    logger.info(f"KnowledgeNode: Retrieved teaching material for lesson: {lesson_id}")
    
    return {"teaching_material": material}

async def get_model_answer_node(state: AgentGraphState) -> dict:
    """Fetches a model speaking script or essay."""
    context = state.get("current_context")
    prompt_id = getattr(context, "current_prompt_id", None) if context else None
    
    if not prompt_id:
        logger.info("KnowledgeNode: No prompt_id provided in context for model answer")
        return {"model_answer": None}
    
    model_answer = KNOWLEDGE_BASE.get("model_answers", {}).get(prompt_id)
    logger.info(f"KnowledgeNode: Retrieved model answer for prompt: {prompt_id}")
    
    return {"model_answer": model_answer}

async def get_skill_drill_content_node(state: AgentGraphState) -> dict:
    """Fetches content for an exercise."""
    context = state.get("current_context")
    
    # Check if context and selected_tools_or_options exist
    if not context or not hasattr(context, "selected_tools_or_options") or not context.selected_tools_or_options:
        logger.info("KnowledgeNode: No drill_id available in context")
        return {"skill_drill_content": None}
    
    drill_id = context.selected_tools_or_options.get("drill_id")
    
    if not drill_id:
        logger.info("KnowledgeNode: No drill_id specified in selected_tools_or_options")
        return {"skill_drill_content": None}
    
    drill_content = KNOWLEDGE_BASE.get("skill_drills", {}).get(drill_id)
    logger.info(f"KnowledgeNode: Retrieved skill drill content for ID: {drill_id}")
    
    return {"skill_drill_content": drill_content}

async def get_rubric_details_node(state: AgentGraphState) -> dict:
    """Provides scoring rubrics to DiagnosticNode."""
    context = state.get("current_context")
    
    if not context:
        logger.info("KnowledgeNode: No context provided for rubric")
        return {"rubric_details": None}
    
    section = getattr(context, "toefl_section", None)
    question_type = getattr(context, "question_type", None)
    
    if not section or not question_type:
        logger.info("KnowledgeNode: Missing section or question_type in context")
        return {"rubric_details": None}
    
    rubric_key = f"{section}_{question_type}"
    rubric = KNOWLEDGE_BASE.get("rubrics", {}).get(rubric_key)
    logger.info(f"KnowledgeNode: Retrieved rubric for {rubric_key}")
    
    return {"rubric_details": rubric}