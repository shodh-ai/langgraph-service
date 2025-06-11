import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import json

# Import all the agent node functions
from graph.teaching_flow import create_teaching_subgraph
from graph.feedback_flow import create_feedback_subgraph
from graph.scaffolding_flow import create_scaffolding_subgraph
from graph.modeling_flow import create_modeling_subgraph
from graph.cowriting_flow import create_cowriting_subgraph
from agents import (
    save_interaction_node,
    format_final_output_node,
    handle_welcome_node,
    student_data_node,
    welcome_prompt_node,
    conversation_handler_node,
    error_generator_node,
    feedback_student_data_node,
    query_document_node,
    RAG_document_node,
    feedback_planner_node,
    feedback_generator_node,
    # Import the new scaffolding nodes
    scaffolding_student_data_node,
    struggle_analyzer_node,
    scaffolding_retriever_node,
    scaffolding_planner_node,
    scaffolding_generator_node,
)

logger = logging.getLogger(__name__)

# Define node names for clarity
NODE_SAVE_INTERACTION = "save_interaction"
NODE_CONVERSATION_HANDLER = "conversation_handler"
NODE_FORMAT_FINAL_OUTPUT = "format_final_output"
NODE_HANDLE_WELCOME = "handle_welcome"
NODE_STUDENT_DATA = "student_data"
NODE_WELCOME_PROMPT = "welcome_prompt"
NODE_ERROR_GENERATION = "error_generation"
NODE_FEEDBACK_STUDENT_DATA = "feedback_student_data"
NODE_QUERY_DOCUMENT = "query_document"
NODE_RAG_DOCUMENT = "RAG_document"
NODE_FEEDBACK_PLANNER = "feedback_planner"
NODE_FEEDBACK_GENERATOR = "feedback_generator"

# Define node names for scaffolding system
NODE_SCAFFOLDING_STUDENT_DATA = "scaffolding_student_data"
NODE_STRUGGLE_ANALYZER = "struggle_analyzer"
NODE_SCAFFOLDING_RETRIEVER = "scaffolding_retriever"
NODE_SCAFFOLDING_PLANNER = "scaffolding_planner"
NODE_SCAFFOLDING_GENERATOR = "scaffolding_generator"

# Define node names for teaching module and other subgraphs
NODE_TEACHING_MODULE = "TEACHING_MODULE"
NODE_FEEDBACK_MODULE = "FEEDBACK_MODULE"
NODE_SCAFFOLDING_MODULE = "SCAFFOLDING_MODULE"
NODE_MODELING_MODULE = "MODELING_MODULE"
NODE_COWRITING_MODULE = "COWRITING_MODULE"
NODE_P1_CURRICULUM_NAVIGATOR = "p1_curriculum_navigator"

# Define a router node (empty function that doesn't modify state)
async def router_node(state: AgentGraphState) -> dict:
    logger.info(
        f"Router node entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    return {}


# Placeholder for P1 Curriculum Navigator node
async def p1_curriculum_navigator_node(state: AgentGraphState) -> dict:
    logger.info(
        f"P1 Curriculum Navigator activated for user {state.get('user_id', 'unknown_user')}. Deciding next steps after teaching module."
    )
    # This node would typically set state to guide the next actions, e.g., next lesson, or switch to another mode.
    # For now, it prepares for a general conversation handler.
    return {"message": "Teaching module completed. Navigating to next steps."}

# Define the initial router function based on task_stage
async def initial_router_logic(state: AgentGraphState) -> str:
    next_task_details = state.get("next_task_details", {})
    user_id = state.get('user_id', 'unknown_user')

    # Priority routing for specific task types like LESSON
    if next_task_details.get("type") == "LESSON":
        logger.info(f"Routing to TEACHING_MODULE for user {user_id}")
        return NODE_TEACHING_MODULE

    context = state.get("current_context")
    transcript = state.get("transcript")
    chat_history = state.get("chat_history")
    task_stage = getattr(context, "task_stage", None)

    if task_stage == "ROX_WELCOME_INIT":
        return NODE_HANDLE_WELCOME
    if task_stage == "FEEDBACK_GENERATION":
        return NODE_FEEDBACK_MODULE
    if task_stage == "SCAFFOLDING_GENERATION":
        return NODE_SCAFFOLDING_MODULE
    # Add hypothetical task stages for new modules
    if task_stage == "MODELING":
        return NODE_MODELING_MODULE
    if task_stage == "COWRITING":
        return NODE_COWRITING_MODULE

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
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

    # Add nodes for core components
    workflow.add_node(NODE_SAVE_INTERACTION, save_interaction_node)
    workflow.add_node(NODE_CONVERSATION_HANDLER, conversation_handler_node)
    workflow.add_node(NODE_FORMAT_FINAL_OUTPUT, format_final_output_node)
    workflow.add_node(NODE_HANDLE_WELCOME, handle_welcome_node)
    workflow.add_node(NODE_STUDENT_DATA, student_data_node)
    workflow.add_node(NODE_WELCOME_PROMPT, welcome_prompt_node)
    
    workflow.add_node(NODE_ROUTER, router_node)

    # Instantiate and add all subgraphs
    teaching_subgraph_instance = create_teaching_subgraph()
    feedback_subgraph_instance = create_feedback_subgraph()
    scaffolding_subgraph_instance = create_scaffolding_subgraph()
    modeling_subgraph_instance = create_modeling_subgraph()
    cowriting_subgraph_instance = create_cowriting_subgraph()

    workflow.add_node(NODE_TEACHING_MODULE, teaching_subgraph_instance)
    workflow.add_node(NODE_FEEDBACK_MODULE, feedback_subgraph_instance)
    workflow.add_node(NODE_SCAFFOLDING_MODULE, scaffolding_subgraph_instance)
    workflow.add_node(NODE_MODELING_MODULE, modeling_subgraph_instance)
    workflow.add_node(NODE_COWRITING_MODULE, cowriting_subgraph_instance)
    
    workflow.add_node(NODE_P1_CURRICULUM_NAVIGATOR, p1_curriculum_navigator_node)

    workflow.set_entry_point(NODE_ROUTER)

    workflow.add_conditional_edges(
        NODE_ROUTER,
        initial_router_logic,
        {
            NODE_HANDLE_WELCOME: NODE_HANDLE_WELCOME,
            NODE_CONVERSATION_HANDLER: NODE_CONVERSATION_HANDLER,
            NODE_TEACHING_MODULE: NODE_TEACHING_MODULE,
            NODE_FEEDBACK_MODULE: NODE_FEEDBACK_MODULE,
            NODE_SCAFFOLDING_MODULE: NODE_SCAFFOLDING_MODULE,
            NODE_MODELING_MODULE: NODE_MODELING_MODULE,
            NODE_COWRITING_MODULE: NODE_COWRITING_MODULE,
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

    # Edges from subgraphs back to the main flow
    workflow.add_edge(NODE_FEEDBACK_MODULE, NODE_FORMAT_FINAL_OUTPUT)
    workflow.add_edge(NODE_SCAFFOLDING_MODULE, NODE_FORMAT_FINAL_OUTPUT)
    workflow.add_edge(NODE_MODELING_MODULE, NODE_FORMAT_FINAL_OUTPUT)
    workflow.add_edge(NODE_COWRITING_MODULE, NODE_FORMAT_FINAL_OUTPUT)

    # Edges for the teaching module flow
    workflow.add_edge(NODE_TEACHING_MODULE, NODE_P1_CURRICULUM_NAVIGATOR)
    workflow.add_edge(NODE_P1_CURRICULUM_NAVIGATOR, NODE_CONVERSATION_HANDLER) # Or to another router/handler if needed

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
