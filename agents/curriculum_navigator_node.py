import logging
import json
from typing import Dict, Any, List
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from state import AgentGraphState
from knowledge.knowledge_client import knowledge_client
from graph.utils import query_knowledge_base

logger = logging.getLogger(__name__)

# This is a new helper function to build the map data. You can place it inside the same file.
def _create_course_map_data(master_curriculum_map: Dict, student_profile: Dict, chosen_lo_id: str) -> Dict:
    """Formats the curriculum and student progress into React Flow compatible JSON."""
    nodes = []
    edges = []
    completed_los = student_profile.get("completed_los", [])
    
    for lo_id, lo_data in master_curriculum_map.items():
        status = "locked" # Default status
        if lo_id in completed_los:
            status = "completed"
        elif lo_id == chosen_lo_id:
            status = "next_up"
        else:
            # Check if all prerequisites are met
            prereqs = lo_data.get("prerequisites", [])
            if all(prereq in completed_los for prereq in prereqs):
                status = "unlocked"

        nodes.append({
            "id": lo_id,
            "data": {
                "label": lo_data.get("title", "Unknown LO"),
                "status": status
            }
        })
        
        for prereq_id in lo_data.get("prerequisites", []):
            edges.append({
                "id": f"e-{prereq_id}-{lo_id}",
                "source": prereq_id,
                "target": lo_id
            })
            
    return {"nodes": nodes, "edges": edges}


async def curriculum_navigator_node(state: AgentGraphState) -> dict:
    """
    The MACRO PLANNER.
    This node determines the next LO, creates the map data, and RETURNS
    a dictionary with only the new data to be merged into the state.
    """
    logger.info("--- Executing Curriculum Navigator Node (Canonical Pattern) ---")
    try:
        # Restore your actual LLM call here. I am using the mock for clarity.
        planner_output = {
            "chosen_lo_id": "GRM_SS_CORE_001",
            "chosen_lo_title": "Identify subjects, verbs, and objects",
            "proposal_script_for_student": "Based on our goals, I think we should start with the basics of sentence structure. How does that sound?"
        }

        chosen_lo_id = planner_output.get("chosen_lo_id")
        student_profile = state.get("student_memory_context", {})
        master_curriculum_map = await knowledge_client.get_full_curriculum_map()
        
        course_map_data = _create_course_map_data(master_curriculum_map, student_profile, chosen_lo_id)
        
        # <<< THIS IS THE FINAL, CORRECT IMPLEMENTATION >>>
        # Create a new dictionary with ONLY the new keys.
        # DO NOT modify the 'state' object directly.
        updates_to_state = {
            "current_lo_to_address": {
                "id": chosen_lo_id,
                "title": planner_output.get("chosen_lo_title"),
                "proposal_script": planner_output.get("proposal_script_for_student")
            },
            "course_map_data": course_map_data
        }
        
        logger.info(f"Navigator is RETURNING a dictionary with keys: {list(updates_to_state.keys())}")
        
        # Return this new dictionary. LangGraph will merge it into the main state.
        return updates_to_state

    except Exception as e:
        logger.error(f"CurriculumNavigatorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": f"Failed to determine next learning objective: {e}"} 