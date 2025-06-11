import logging
import copy
from typing import Dict, Any

logger = logging.getLogger(__name__)

def cowriting_student_data_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Entering cowriting_student_data_node")
    
    state_copy = copy.deepcopy(state)
    
    try:
        current_context = state_copy.get("current_context")
        
        if current_context:
            student_written_chunk = getattr(current_context, "current_written_text", "")
            student_articulated_thought = getattr(current_context, "articulated_thought", "")
            task_type = getattr(current_context, "writing_task_type", "Independent Essay")
            section = getattr(current_context, "writing_section", "Body Paragraph")
            learning_objective = getattr(current_context, "learning_objective", "")
            comfort_level = getattr(current_context, "comfort_level", "Conversational")
        else:
            student_written_chunk = ""
            student_articulated_thought = ""
            task_type = "Independent Essay"
            section = "Body Paragraph"
            learning_objective = ""
            comfort_level = "Conversational"
        
        state_copy["student_written_chunk"] = student_written_chunk
        state_copy["student_articulated_thought"] = student_articulated_thought
        state_copy["writing_task_context"] = {
            "task_type": task_type,
            "section": section
        }
        state_copy["cowriting_lo_focus"] = learning_objective
        state_copy["student_comfort_level"] = comfort_level
        
        logger.info("Successfully extracted student data for cowriting")
        return state_copy
        
    except Exception as e:
        logger.error(f"Error in cowriting_student_data_node: {str(e)}")
        state_copy["error"] = f"Cowriting student data error: {str(e)}"
        
        if "student_written_chunk" not in state_copy:
            state_copy["student_written_chunk"] = ""
        if "writing_task_context" not in state_copy:
            state_copy["writing_task_context"] = {"task_type": "Unknown", "section": "Unknown"}
        
        return state_copy
