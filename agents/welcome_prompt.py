from state import AgentGraphState
import logging

logger = logging.getLogger(__name__)


async def welcome_prompt_node(state: AgentGraphState) -> dict:
    logger.info(
        f"Welcome prompt node entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    return {
        "llm_instruction": "Greet the student in a warm and welcoming tone. Use the student's name given in the greeting. Introduce yourself as Rox."
    }
