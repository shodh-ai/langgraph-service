import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import AgentGraphState
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import json

# Import all the agent node functions
from agents import (
    save_interaction_node,
    format_final_output_for_client_node,
    handle_welcome_node,
    student_data_node,
    welcome_prompt_node,
    conversation_handler_node,
    motivational_support_node, # Added motivational_support_node
    progress_reporter_node, # Added progress_reporter_node
    inactivity_prompt_node, # Added inactivity_prompt_node
    tech_support_acknowledger_node, # Added tech support node
    prepare_navigation_node, # Added prepare_navigation_node
    session_wrap_up_node, # Added session_wrap_up_node
    finalize_session_in_mem0_node, # Added finalize_session_in_mem0_node
    error_generator_node,
    feedback_student_data_node,
    query_document_node,
    RAG_document_node,
    feedback_planner_node,
    feedback_generator_node,
    initial_report_generation_node,
    pedagogy_generator_node,
    # Modelling System Nodes
    modelling_query_document_node,
    modelling_RAG_document_node,
    modelling_generator_node,
    modelling_output_formatter_node,
    # Teaching System Nodes
    teaching_rag_node,
    teaching_delivery_node,
    teaching_generator_node,
)

logger = logging.getLogger(__name__)

# Define node names for clarity
NODE_SAVE_INTERACTION = "save_interaction"
NODE_CONVERSATION_HANDLER = "conversation_handler"
NODE_FORMAT_FINAL_OUTPUT = "format_final_output"
NODE_HANDLE_WELCOME = "handle_welcome"
NODE_STUDENT_DATA = "student_data"
NODE_WELCOME_PROMPT = "welcome_prompt"
NODE_MOTIVATIONAL_SUPPORT = "motivational_support" # Added motivational node name
NODE_PROGRESS_REPORTER = "progress_reporter" # Added progress reporter node name
NODE_INACTIVITY_PROMPT = "inactivity_prompt" # Added inactivity prompt node name
NODE_TECH_SUPPORT_ACKNOWLEDGER = "tech_support_acknowledger" # Added tech support node name
NODE_PREPARE_NAVIGATION = "prepare_navigation" # Added prepare navigation node name
NODE_SESSION_WRAP_UP = "session_wrap_up" # Added session wrap up node name
NODE_FINALIZE_SESSION_IN_MEM0 = "finalize_session_in_mem0" # Added finalize session in Mem0 node name
NODE_ERROR_GENERATION = "error_generation"
NODE_FEEDBACK_STUDENT_DATA = "feedback_student_data"
NODE_QUERY_DOCUMENT = "query_document"
NODE_RAG_DOCUMENT = "RAG_document"
NODE_FEEDBACK_PLANNER = "feedback_planner"
NODE_FEEDBACK_GENERATOR = "feedback_generator"
NODE_INITIAL_REPORT_GENERATION = "initial_report_generation"
NODE_PEDAGOGY_GENERATION = "pedagogy_generation"

# Modelling System Node Names
NODE_MODELLING_QUERY_DOCUMENT = "modelling_query_document"
NODE_MODELLING_RAG_DOCUMENT = "modelling_RAG_document"
NODE_MODELLING_GENERATOR = "modelling_generator"
NODE_MODELLING_OUTPUT_FORMATTER = "modelling_output_formatter"

# Teaching System Node Names
NODE_TEACHING_RAG = "teaching_RAG_node"
NODE_TEACHING_DELIVERY = "teaching_delivery_node" # This might become obsolete or used for non-LLM paths
NODE_TEACHING_GENERATOR = "teaching_generator_node" # New LLM-based teaching node

# Define a router node (empty function that doesn't modify state)
async def router_node(state: AgentGraphState) -> dict:
    logger.info(
        f"Router node entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    return {}


# Router function after saving an interaction
async def route_after_save_interaction(state: AgentGraphState) -> str:
    if state.get("session_is_ending", False):
        logger.info("Session is ending. Routing from SaveInteraction to FinalizeSessionInMem0.")
        return NODE_FINALIZE_SESSION_IN_MEM0
    else:
        logger.info("Session not ending. Interaction saved. Routing to END.")
        return END


# Define the initial router function based on task_stage
# Router function after motivational support
async def route_after_motivation(state: AgentGraphState) -> str:
    hint = state.get("next_node_hint_from_motivation")
    logger.info(f"Routing after motivational support. Hint: {hint}")
    if hint and hint in [NODE_CONVERSATION_HANDLER, NODE_WELCOME_PROMPT, NODE_STUDENT_DATA, NODE_HANDLE_WELCOME]: # Add other valid node names
        return hint
    # Add more sophisticated logic if the hint can be more complex
    # For example, if hint is 'REVIEW_COHERENCE_NOTES', map it to a specific graph node
    # if hint == "REVIEW_COHERENCE_NOTES":
    #     return "NODE_KNOWLEDGE_REVIEW_COHERENCE" # Assuming such a node exists
    logger.warning(f"No specific route or unknown hint '{hint}' after motivational support. Defaulting to conversation handler.")
    return "DEFAULT_FALLBACK_AFTER_MOTIVATION" # Fallback to conversation_handler via the conditional edge map

async def initial_router_logic(state: AgentGraphState) -> str:
    context = state.get("current_context")
    task_stage_from_context = getattr(context, "task_stage", None) # Get task_stage from context object if it's an object
    if isinstance(context, dict):
        task_stage_from_context = context.get("task_stage") # Get task_stage if context is a dict

    # Handle inactivity prompt first
    if task_stage_from_context == "SYSTEM_USER_INACTIVITY_DETECTED":
        logger.info(f"Detected user inactivity. Routing to NODE_INACTIVITY_PROMPT.")
        return NODE_INACTIVITY_PROMPT

    transcript = state.get("transcript")
    chat_history = state.get("chat_history")
    # task_stage is already fetched as task_stage_from_context
    if task_stage_from_context == "ROX_WELCOME_INIT":

        return NODE_HANDLE_WELCOME
    if task_stage_from_context == "FEEDBACK_GENERATION": # Corrected to use task_stage_from_context
        return NODE_FEEDBACK_STUDENT_DATA
    if task_stage_from_context == "INITIAL_REPORT_GENERATION":
        return NODE_INITIAL_REPORT_GENERATION
    if task_stage_from_context == "PEDAGOGY_GENERATION":
        return NODE_PEDAGOGY_GENERATION
    if task_stage_from_context == "MODELLING_ACTIVITY_REQUESTED":
        logger.info(f"Task stage is MODELLING_ACTIVITY_REQUESTED. Routing to NODE_MODELLING_QUERY_DOCUMENT.")
        return NODE_MODELLING_QUERY_DOCUMENT
    if task_stage_from_context == "TEACHING_LESSON_REQUESTED": # New task stage for teaching
        logger.info(f"Task stage is TEACHING_LESSON_REQUESTED. Routing to NODE_TEACHING_RAG.")
        return NODE_TEACHING_RAG

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
        - 'REPORT_TECHNICAL_ISSUE'  # New intent
        - 'INTENT_TO_QUIT_SESSION' # New intent for ending the session
        - 'OTHER_OFF_TOPIC'
        If 'ASK_GENERAL_KNOWLEDGE_QUESTION', extract the core topic/question.
        If 'REPORT_TECHNICAL_ISSUE', extract 'issue_description' (what the student says is wrong, e.g., 'my audio is not working') and 'reported_emotion' (e.g., 'frustration', 'confusion', 'annoyance' based on their language).
        """
            + "Return JSON: {'intent': '<INTENT_NAME>', 'extracted_topic': '<topic if ASK_GENERAL_KNOWLEDGE_QUESTION>', 'extracted_entities': {'issue_description': '<description if REPORT_TECHNICAL_ISSUE>', 'reported_emotion': '<emotion if REPORT_TECHNICAL_ISSUE>'}}"
        )
        response = model.generate_content(prompt)
        print("Response Text:", response.text)
        # Check for progress-related queries
        progress_queries = [
            "how am i doing", "what's my progress", "am i improving", 
            "my score went down", "check my progress", "show me my progress"
        ]
        if any(query in transcript.lower() for query in progress_queries):
            logger.info(f"Detected progress query in transcript: {transcript}")
            state['triggering_event_for_motivation'] = None # Clear motivational trigger if any
            return NODE_PROGRESS_REPORTER

        if "frustrated" in transcript.lower() or "nervous" in transcript.lower() or "bored" in transcript.lower() or "i can't do this" in transcript.lower():
            logger.info(f"Detected potential need for motivational support in transcript: {transcript}")
            # Set the trigger event for the motivational node
            state['triggering_event_for_motivation'] = f"Detected sentiment in user transcript: {transcript}"
            return NODE_MOTIVATIONAL_SUPPORT

        response_json = json.loads(response.text)
        intent = response_json.get("intent", "GENERAL_CHITCHAT")
        extracted_topic = response_json.get("extracted_topic", "") # For ASK_GENERAL_KNOWLEDGE_QUESTION
        extracted_entities_from_nlu = response_json.get("extracted_entities", {}) # For REPORT_TECHNICAL_ISSUE
        state['nlu_intent'] = intent # Store NLU intent in state

        # Store extracted entities in state if present (for tech support node)
        if extracted_entities_from_nlu and isinstance(extracted_entities_from_nlu, dict) and (extracted_entities_from_nlu.get('issue_description') or extracted_entities_from_nlu.get('reported_emotion')):
            state['extracted_entities'] = extracted_entities_from_nlu
            logger.info(f"NLU extracted entities for technical issue: {extracted_entities_from_nlu}")
        else: # Ensure the key exists even if empty, or clear previous
            state['extracted_entities'] = {}

        print("Intent:", intent)
        print("Extracted Topic:", extracted_topic)
        print("Extracted Entities (for tech issue):", state.get('extracted_entities'))

        if intent == "INTENT_TO_QUIT_SESSION":
            logger.info(f"Detected intent: INTENT_TO_QUIT_SESSION. Routing to NODE_SESSION_WRAP_UP.")
            return NODE_SESSION_WRAP_UP
        elif intent == "REPORT_TECHNICAL_ISSUE":
            logger.info(f"Detected intent: REPORT_TECHNICAL_ISSUE. Routing to NODE_TECH_SUPPORT_ACKNOWLEDGER.")
            return NODE_TECH_SUPPORT_ACKNOWLEDGER
        elif intent == "CONFIRM_START_SUGGESTED_TASK":
            next_task_details = state.get("next_task_details")
            if next_task_details and isinstance(next_task_details, dict) and next_task_details.get("page_target"):
                logger.info(f"Intent is CONFIRM_START_SUGGESTED_TASK and next_task_details are present. Routing to NODE_PREPARE_NAVIGATION.")
                return NODE_PREPARE_NAVIGATION
            else:
                logger.warning(f"Intent is CONFIRM_START_SUGGESTED_TASK, but 'next_task_details' are missing or invalid. Routing to NODE_CONVERSATION_HANDLER to clarify.")
                # Potentially add a flag to state to inform conversation_handler about this situation
                state['missing_next_task_for_confirmation'] = True
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

    # Instantiate the checkpointer for state persistence
    memory_saver = MemorySaver()

    workflow = StateGraph(AgentGraphState)

    workflow.add_node(NODE_SAVE_INTERACTION, save_interaction_node)
    workflow.add_node(NODE_CONVERSATION_HANDLER, conversation_handler_node)
    workflow.add_node(NODE_FORMAT_FINAL_OUTPUT, format_final_output_for_client_node)
    workflow.add_node(NODE_HANDLE_WELCOME, handle_welcome_node)
    workflow.add_node(NODE_STUDENT_DATA, student_data_node)
    workflow.add_node(NODE_WELCOME_PROMPT, welcome_prompt_node)
    workflow.add_node(NODE_MOTIVATIONAL_SUPPORT, motivational_support_node) # Added motivational node
    workflow.add_node(NODE_PROGRESS_REPORTER, progress_reporter_node) # Added progress reporter node
    workflow.add_node(NODE_INACTIVITY_PROMPT, inactivity_prompt_node) # Added inactivity prompt node
    workflow.add_node(NODE_TECH_SUPPORT_ACKNOWLEDGER, tech_support_acknowledger_node) # Added tech support node
    workflow.add_node(NODE_PREPARE_NAVIGATION, prepare_navigation_node) # Added prepare_navigation_node
    workflow.add_node(NODE_SESSION_WRAP_UP, session_wrap_up_node) # Added session_wrap_up_node
    workflow.add_node(NODE_FINALIZE_SESSION_IN_MEM0, finalize_session_in_mem0_node) # Added finalize_session_in_mem0_node
    workflow.add_node(NODE_ERROR_GENERATION, error_generator_node)
    workflow.add_node(NODE_FEEDBACK_STUDENT_DATA, feedback_student_data_node)
    workflow.add_node(NODE_QUERY_DOCUMENT, query_document_node)
    workflow.add_node(NODE_RAG_DOCUMENT, RAG_document_node)
    workflow.add_node(NODE_FEEDBACK_PLANNER, feedback_planner_node)
    workflow.add_node(NODE_FEEDBACK_GENERATOR, feedback_generator_node)
    workflow.add_node(NODE_INITIAL_REPORT_GENERATION, initial_report_generation_node)
    workflow.add_node(NODE_PEDAGOGY_GENERATION, pedagogy_generator_node)

    # Modelling System Nodes
    workflow.add_node(NODE_MODELLING_QUERY_DOCUMENT, modelling_query_document_node)
    workflow.add_node(NODE_MODELLING_RAG_DOCUMENT, modelling_RAG_document_node)
    workflow.add_node(NODE_MODELLING_GENERATOR, modelling_generator_node)
    workflow.add_node(NODE_MODELLING_OUTPUT_FORMATTER, modelling_output_formatter_node)

    # Teaching System Nodes
    workflow.add_node(NODE_TEACHING_RAG, teaching_rag_node)
    workflow.add_node(NODE_TEACHING_DELIVERY, teaching_delivery_node) # This might become obsolete or used for non-LLM paths
    workflow.add_node(NODE_TEACHING_GENERATOR, teaching_generator_node)

    workflow.add_node(NODE_ROUTER, router_node)

    workflow.set_entry_point(NODE_ROUTER)

    # Conditional Edges from the main router
    workflow.add_conditional_edges(
        NODE_ROUTER,
        initial_router_logic,
        {
            NODE_HANDLE_WELCOME: NODE_HANDLE_WELCOME,
            NODE_CONVERSATION_HANDLER: NODE_CONVERSATION_HANDLER,
            NODE_MOTIVATIONAL_SUPPORT: NODE_MOTIVATIONAL_SUPPORT,
            NODE_PROGRESS_REPORTER: NODE_PROGRESS_REPORTER,
            NODE_INACTIVITY_PROMPT: NODE_INACTIVITY_PROMPT,
            NODE_TECH_SUPPORT_ACKNOWLEDGER: NODE_TECH_SUPPORT_ACKNOWLEDGER,
            NODE_PREPARE_NAVIGATION: NODE_PREPARE_NAVIGATION,
            NODE_SESSION_WRAP_UP: NODE_SESSION_WRAP_UP,
            NODE_FEEDBACK_STUDENT_DATA: NODE_FEEDBACK_STUDENT_DATA,
            NODE_INITIAL_REPORT_GENERATION: NODE_INITIAL_REPORT_GENERATION,
            NODE_PEDAGOGY_GENERATION: NODE_PEDAGOGY_GENERATION,
            NODE_MODELLING_QUERY_DOCUMENT: NODE_MODELLING_QUERY_DOCUMENT, # Added modelling route
            NODE_TEACHING_RAG: NODE_TEACHING_RAG, # Added teaching route
        },
    )

    # Standard flow: Node Logic -> Format Output -> Save Interaction -> [Conditional Finalize] -> END

    # Welcome Path
    workflow.add_edge(NODE_HANDLE_WELCOME, NODE_STUDENT_DATA)
    workflow.add_edge(NODE_STUDENT_DATA, NODE_WELCOME_PROMPT)
    workflow.add_edge(NODE_WELCOME_PROMPT, NODE_CONVERSATION_HANDLER)  # Changed to route to conversation_handler

    # General Conversation Path
    workflow.add_edge(NODE_CONVERSATION_HANDLER, NODE_FORMAT_FINAL_OUTPUT)

    # Motivational Support Path
    workflow.add_conditional_edges(
        NODE_MOTIVATIONAL_SUPPORT,
        route_after_motivation, # This router decides the next logical step (e.g., back to conversation or welcome)
        {
            NODE_CONVERSATION_HANDLER: NODE_CONVERSATION_HANDLER,
            NODE_WELCOME_PROMPT: NODE_WELCOME_PROMPT, # Or directly to format if motivation has output
            NODE_STUDENT_DATA: NODE_STUDENT_DATA,
            NODE_HANDLE_WELCOME: NODE_HANDLE_WELCOME,
            "DEFAULT_FALLBACK_AFTER_MOTIVATION": NODE_CONVERSATION_HANDLER, 
        }
    )
    # If motivational_support_node itself produces output for the user immediately, it should also go to FORMAT_FINAL_OUTPUT.
    # Assuming for now route_after_motivation sends it to a node that will eventually lead to FORMAT_FINAL_OUTPUT.
    # If NODE_MOTIVATIONAL_SUPPORT directly has output, add: workflow.add_edge(NODE_MOTIVATIONAL_SUPPORT, NODE_FORMAT_FINAL_OUTPUT)

    # Progress Reporter Path
    workflow.add_edge(NODE_PROGRESS_REPORTER, NODE_FORMAT_FINAL_OUTPUT)

    # Inactivity Prompt Path
    workflow.add_edge(NODE_INACTIVITY_PROMPT, NODE_FORMAT_FINAL_OUTPUT)

    # Tech Support Acknowledger Path
    workflow.add_edge(NODE_TECH_SUPPORT_ACKNOWLEDGER, NODE_FORMAT_FINAL_OUTPUT)

    # Prepare Navigation Path
    workflow.add_edge(NODE_PREPARE_NAVIGATION, NODE_FORMAT_FINAL_OUTPUT)

    # Session Wrap Up Path
    workflow.add_edge(NODE_SESSION_WRAP_UP, NODE_FORMAT_FINAL_OUTPUT) # Wrap-up message needs formatting

    # Common path after output is formatted
    workflow.add_edge(NODE_FORMAT_FINAL_OUTPUT, NODE_SAVE_INTERACTION)

    # Conditional path after interaction is saved
    workflow.add_conditional_edges(
        NODE_SAVE_INTERACTION,
        route_after_save_interaction, # Decides if session finalization is needed
        {
            NODE_FINALIZE_SESSION_IN_MEM0: NODE_FINALIZE_SESSION_IN_MEM0,
            END: END, # If route_after_save_interaction returns END (i.e., __end__), then terminate the graph.
                                         # This was an error in previous logic, FormatFinalOutput should not be hit again here.
                                         # Corrected to END for non-session-ending paths.
        },
    )
    # If session is NOT ending, route_after_save_interaction returns NODE_FORMAT_FINAL_OUTPUT.
    # This should be END, as formatting and saving are done.
    # Let's refine route_after_save_interaction to return END for the 'else' case.

    # Path after session data is finalized in Mem0
    workflow.add_edge(NODE_FINALIZE_SESSION_IN_MEM0, END)

    # Feedback generation flow
    workflow.add_edge(NODE_FEEDBACK_STUDENT_DATA, NODE_ERROR_GENERATION)
    workflow.add_edge(NODE_ERROR_GENERATION, NODE_QUERY_DOCUMENT)
    workflow.add_edge(NODE_QUERY_DOCUMENT, NODE_RAG_DOCUMENT)
    workflow.add_edge(NODE_RAG_DOCUMENT, NODE_FEEDBACK_PLANNER)
    workflow.add_edge(NODE_FEEDBACK_PLANNER, NODE_FEEDBACK_GENERATOR)
    workflow.add_edge(NODE_FEEDBACK_GENERATOR, NODE_FORMAT_FINAL_OUTPUT)

    # Initial report generation flow
    workflow.add_edge(NODE_INITIAL_REPORT_GENERATION, NODE_FORMAT_FINAL_OUTPUT)

    # Pedagogy generation flow
    workflow.add_edge(NODE_PEDAGOGY_GENERATION, NODE_FORMAT_FINAL_OUTPUT)

    # Modelling system flow
    workflow.add_edge(NODE_MODELLING_QUERY_DOCUMENT, NODE_MODELLING_RAG_DOCUMENT)
    workflow.add_edge(NODE_MODELLING_RAG_DOCUMENT, NODE_MODELLING_GENERATOR)
    workflow.add_edge(NODE_MODELLING_GENERATOR, NODE_MODELLING_OUTPUT_FORMATTER)
    workflow.add_edge(NODE_MODELLING_OUTPUT_FORMATTER, NODE_FORMAT_FINAL_OUTPUT)

    # Teaching System Edges (LLM-based flow)
    workflow.add_edge(NODE_TEACHING_RAG, NODE_TEACHING_GENERATOR) # RAG output goes to the new generator
    workflow.add_edge(NODE_TEACHING_GENERATOR, NODE_FORMAT_FINAL_OUTPUT) # Generator output goes to formatter

    # Compile the graph with the checkpointer
    app = workflow.compile(checkpointer=memory_saver)
    logger.info("LangGraph compiled successfully with MemorySaver checkpointer.")
    return app


# To test the graph building process (optional, can be run directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        graph = build_graph()
        logger.info("Graph compiled successfully in __main__.")
    except Exception as e_build:
        logger.error(f"Error building graph in __main__: {e_build}", exc_info=True)
