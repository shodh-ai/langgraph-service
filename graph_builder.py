import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import json

# Import all the agent node functions
from agents import (
    save_interaction_node,
    format_final_output_node,
    handle_welcome_node,
    student_data_node,
    welcome_prompt_node,
    conversation_handler_node,
)

logger = logging.getLogger(__name__)

# Define node names for clarity
NODE_SAVE_INTERACTION = "save_interaction"
NODE_CONVERSATION_HANDLER = "conversation_handler"
NODE_FORMAT_FINAL_OUTPUT = "format_final_output"
NODE_HANDLE_WELCOME = "handle_welcome"
NODE_STUDENT_DATA = "student_data"
NODE_WELCOME_PROMPT = "welcome_prompt"


# Define a router node (empty function that doesn't modify state)
async def router_node(state: AgentGraphState) -> dict:
    logger.info(
        f"Router node entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    return {}


# Define the initial router function based on task_stage
async def initial_router_logic(state: AgentGraphState) -> str:
    context = state.get("current_context")
    transcript = state.get("transcript")
    chat_history = state.get("chat_history")
    task_stage = getattr(context, "task_stage", None)

    if task_stage == "ROX_WELCOME_INIT":
        return NODE_HANDLE_WELCOME

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.5-flash-preview-05-20",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        prompt = (
            f"""
        You are an NLU assistant for the Rox AI Tutor on its welcome page.
        The student responded: '{transcript}'.
        Chat History: {chat_history}
        Categorize the student's intent from the possible intents. Possible intents:
        - 'CONFIRM_START_SUGGESTED_TASK'
        - 'REJECT_SUGGESTED_TASK_REQUEST_ALTERNATIVE'
        - 'ASK_CLARIFYING_QUESTION_ABOUT_SUGGESTED_TASK'
        - 'ASK_GENERAL_KNOWLEDGE_QUESTION' (e.g., about a grammar topic, TOEFL strategy)
        - 'REQUEST_STATUS_DETAIL'
        - 'GENERAL_CHITCHAT'
        - 'OTHER_OFF_TOPIC'
        If 'ASK_GENERAL_KNOWLEDGE_QUESTION', extract the core topic/question.
        """
            + "Return JSON: {'intent': '<INTENT_NAME>', 'extracted_topic': '<topic if any>'}"
        )
        response = model.generate_content(prompt)
        print("Response Text:", response.text)
        response_json = json.loads(response.text)
        intent = response_json.get("intent", "GENERAL_CHITCHAT")
        extracted_topic = response_json.get("extracted_topic", "")
        print("Intent:", intent)
        print("Extracted Topic:", extracted_topic)
        if intent == "CONFIRM_START_SUGGESTED_TASK":
            return NODE_CONVERSATION_HANDLER
        elif intent == "REJECT_SUGGESTED_TASK_REQUEST_ALTERNATIVE":
            return NODE_CONVERSATION_HANDLER
        elif intent == "ASK_CLARIFYING_QUESTION_ABOUT_SUGGESTED_TASK":
            return NODE_CONVERSATION_HANDLER
        elif intent == "ASK_GENERAL_KNOWLEDGE_QUESTION":
            return NODE_CONVERSATION_HANDLER
        elif intent == "REQUEST_STATUS_DETAIL":
            return NODE_CONVERSATION_HANDLER
        elif intent == "GENERAL_CHITCHAT":
            return NODE_CONVERSATION_HANDLER
        elif intent == "OTHER_OFF_TOPIC":
            return NODE_CONVERSATION_HANDLER
        else:
            return NODE_CONVERSATION_HANDLER
    except Exception as e:
        logger.error(f"Error processing with GenerativeModel: {e}")
        return NODE_CONVERSATION_HANDLER


def build_graph():
    """Builds and compiles the LangGraph application with the P1 and P2 submission flows."""
    logger.info("Building LangGraph with AgentGraphState...")
    NODE_ROUTER = "router"
    workflow = StateGraph(AgentGraphState)

    workflow.add_node(NODE_SAVE_INTERACTION, save_interaction_node)
    workflow.add_node(NODE_CONVERSATION_HANDLER, conversation_handler_node)
    workflow.add_node(NODE_FORMAT_FINAL_OUTPUT, format_final_output_node)
    workflow.add_node(NODE_HANDLE_WELCOME, handle_welcome_node)
    workflow.add_node(NODE_STUDENT_DATA, student_data_node)
    workflow.add_node(NODE_WELCOME_PROMPT, welcome_prompt_node)
    workflow.add_node(NODE_ROUTER, router_node)

    workflow.set_entry_point(NODE_ROUTER)

    workflow.add_conditional_edges(
        NODE_ROUTER,
        initial_router_logic,
        {
            NODE_HANDLE_WELCOME: NODE_HANDLE_WELCOME,
            NODE_CONVERSATION_HANDLER: NODE_CONVERSATION_HANDLER,
        },
    )

    # Welcome flow
    workflow.add_edge(NODE_HANDLE_WELCOME, NODE_STUDENT_DATA)
    workflow.add_edge(NODE_HANDLE_WELCOME, NODE_WELCOME_PROMPT)
    workflow.add_edge(NODE_STUDENT_DATA, NODE_CONVERSATION_HANDLER)
    workflow.add_edge(NODE_WELCOME_PROMPT, NODE_CONVERSATION_HANDLER)

    # Conversation flow
    workflow.add_edge(NODE_CONVERSATION_HANDLER, NODE_FORMAT_FINAL_OUTPUT)
    workflow.add_edge(NODE_FORMAT_FINAL_OUTPUT, NODE_SAVE_INTERACTION)
    workflow.add_edge(NODE_SAVE_INTERACTION, END)

    # Compile the graph
    app_graph = workflow.compile()
    logger.info("LangGraph with flows built and compiled successfully.")
    return app_graph


# To test the graph building process (optional, can be run directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        graph = build_graph()
        logger.info("Graph compiled successfully in __main__.")
    except Exception as e_build:
        logger.error(f"Error building graph in __main__: {e_build}", exc_info=True)
