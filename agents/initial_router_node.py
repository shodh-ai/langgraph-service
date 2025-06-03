import logging
from state import AgentGraphState
import yaml
import os
import json
import vertexai
from vertexai.generative_models import GenerativeModel, Content

from typing import Dict

logger = logging.getLogger(__name__)
PROMPTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "llm_prompts.yaml")

try:
    with open(PROMPTS_PATH, 'r') as file:
        PROMPTS = yaml.safe_load(file)
except Exception as e:
    logger.error(f"Failed to load LLM prompts: {e}")
    PROMPTS = {}

try:
    if 'vertexai' in globals() and not hasattr(vertexai, '_initialized'):
        vertexai.init(project="windy-orb-460108-t0", location="us-central1")
        vertexai._initialized = True
        logger.info("Vertex AI initialized in initial_router_node")
except Exception as e:
    logger.error(f"Failed to initialize Vertex AI in initial_router_node: {e}")

try:
    gemini_model = GenerativeModel("gemini-1.5-pro")
    logger.info("Gemini model loaded in initial_router_node")
except Exception as e:
    logger.error(f"Failed to load Gemini model in initial_router_node: {e}")
    gemini_model = None

async def start_graph_node(state: AgentGraphState) -> Dict:
    """
    Entry point node for the graph. 
    It doesn't modify the state itself but allows conditional routing to begin.
    """
    logger.info("Graph execution started at entry point node.")
    return {} # Must return a dictionary


async def route_initial_request_node(state: AgentGraphState) -> str:
    task_stage = state["current_context"].task_stage # Assume current_context is always present
    logger.info(f"InitialRouter: TaskStage='{task_stage}' for user '{state['user_id']}'")

    # Define node names as string literals for clarity if not imported as constants
    NODE_LOAD_STUDENT_PROFILE = "load_or_initialize_student_profile"
    NODE_PROCESS_SPEAKING_SUBMISSION = "process_speaking_submission_node" # Placeholder, ensure this node exists
    NODE_PROCESS_CONVERSATIONAL_TURN = "process_conversational_turn_node"
    NODE_HANDLE_UNMATCHED_INTERACTION = "handle_unmatched_interaction_node" # Placeholder, ensure this node exists

    if task_stage == "ROX_WELCOME_INIT":
        return NODE_LOAD_STUDENT_PROFILE # Start of the P1 welcome flow
    elif task_stage == "SPEAKING_TESTING_SUBMITTED":
        return NODE_PROCESS_SPEAKING_SUBMISSION # Start of P2 post-submission flow
    elif task_stage == "ROX_CONVERSATION_TURN":
        return NODE_PROCESS_CONVERSATIONAL_TURN # If student speaks on P1 after welcome
    # ... other top-level task_stage routes ...
    else:
        logger.warning(f"InitialRouter: Unhandled task_stage '{task_stage}'. Defaulting to fallback.")
        return NODE_HANDLE_UNMATCHED_INTERACTION # Or a more graceful default welcome
