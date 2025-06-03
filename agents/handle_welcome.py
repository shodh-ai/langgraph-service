from state import AgentGraphState
import logging

logger = logging.getLogger(__name__)


async def handle_welcome_node(state: AgentGraphState) -> dict:
    logger.info(f"Welcome node entry point activated for user {state.get('user_id', 'unknown_user')}")
    return {}