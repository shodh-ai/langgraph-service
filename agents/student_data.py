from state import AgentGraphState
import logging

logger = logging.getLogger(__name__)


async def student_data_node(state: AgentGraphState) -> dict:
    logger.info(
        f"Student data node entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    return {"user_data": {"name": "Harshit", "level": "Beginner"}}
