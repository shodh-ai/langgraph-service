from state import AgentGraphState
import logging

logger = logging.getLogger(__name__)


async def welcome_prompt_node(state: AgentGraphState) -> dict:
    user_id = state.get("user_id", "unknown_user")
    logger.info(f"Welcome prompt node entry point activated for user {user_id}")

    student_name = (state.get("student_memory_context") or {}).get("profile", {}).get("name", "student")
    active_persona = state.get("active_persona", "Nurturer") # Assuming active_persona might be in state

    # Define the task to be suggested
    next_task = {
        "task_id": "TEST_VOCAB_01",
        "task_type": "VOCABULARY_PRACTICE",
        "title": "Test Vocabulary Task",
        "description": "Practice some test vocabulary words.",
        "page_target": "P_VOCAB_DRILL",
        "config": {"words": ["test", "example", "practice"]}
    }

    # Define the example JSON structure string that the LLM should produce
    json_output_example_for_llm_prompt = '{"tts": "The text-to-speech for the message."}' # Note: escaped quotes for the string itself

    # This instruction will be used by conversation_handler_node's LLM
    # It prompts the LLM to greet the student and suggest the 'next_task'
    instruction_for_conversation_handler = (
        f"You are Rox, an AI TOEFL Tutor, speaking as the '{active_persona}' persona. "
        f"The student's name is {student_name}. "
        f"Greet the student warmly and introduce yourself. "
        f"Then, suggest they start with the '{next_task['title']}'. "
        f"Keep it conversational and brief (1-2 sentences). "
        f"Your output MUST be a JSON object with the following structure: {json_output_example_for_llm_prompt}"
    )

    logger.info(f"Setting llm_instruction for conversation_handler and next_task_details for user {user_id}. Instruction: {instruction_for_conversation_handler}")

    return {
        "llm_instruction": instruction_for_conversation_handler,
        "next_task_details": next_task
    }
