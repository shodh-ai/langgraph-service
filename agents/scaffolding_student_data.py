from state import AgentGraphState
import logging

logger = logging.getLogger(__name__)


async def scaffolding_student_data_node(state: AgentGraphState) -> dict:
    """
    Node that retrieves student data for scaffolding purposes.
    
    This node gathers information about the student's profile, including their
    English proficiency level, learning objectives, and specific struggles
    that will be used to determine appropriate scaffolding strategies.
    """
    logger.info(
        f"Scaffolding Student data node entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    new_state = {key: value for key, value in state.items()}
    
    logger.info(f"Starting with state keys: {list(new_state.keys())}")
    
    new_state["user_data"] = {
        "name": "Harshit", 
        "level": "Beginner",
        "goal": "Improve TOEFL speaking score",
        "confidence": "Low confidence in speaking under time constraints",
        "attitude": "Eager but anxious about performance"
    }
    
    logger.info(f"Student data node returning state keys: {list(new_state.keys())}")
    
    return new_state
