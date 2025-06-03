import logging
from state import AgentGraphState
import yaml
import os
import json
# Try to import vertexai, but don't fail if it's not available
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Content, Part
    vertexai_available = True
except ImportError:
    vertexai_available = False
    
# Import our fallback utilities
from utils.fallback_utils import get_model_with_fallback

logger = logging.getLogger(__name__)
PROMPTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "llm_prompts.yaml")

try:
    with open(PROMPTS_PATH, 'r') as file:
        PROMPTS = yaml.safe_load(file)
except Exception as e:
    logger.error(f"Failed to load LLM prompts: {e}")
    PROMPTS = {}

# Use our fallback utility to get a model (either real or fallback)
gemini_model = get_model_with_fallback("gemini-2.5-flash-preview-05-20")
logger.info(f"Using model: {type(gemini_model).__name__} in initial_router_node")

async def route_initial_request_node(state: AgentGraphState) -> dict:
    """
    Analyzes the initial request and determines the appropriate routing path.
    This node acts as the entry point for the agent graph, making intelligent
    decisions about which node should handle the request based on content analysis.
    
    Args:
        state (AgentGraphState): The current state of the agent graph
        
    Returns:
        dict: Contains the routing_decision with next_node and rationale
    """
    context = state.get("current_context", {})
    student_data = state.get("student_memory_context", {})
    current_message = state.get("current_message", "")
    
    logger.info(f"InitialRouterNode: Routing initial request for user: {state.get('user_id', 'unknown')}")
    
    # Default routing based on context if available
    task_stage = getattr(context, 'task_stage', None)
    toefl_section = getattr(context, 'toefl_section', None)
    
    # Default routing decision
    routing_decision = {
        "next_node": "apply_teacher_persona",  # Default to persona node as entry point
        "rationale": "Starting standard flow with teacher persona application",
        "detected_intent": "standard_flow"
    }
    
    # If we have explicit task_stage, use that for routing
    if task_stage:
        if task_stage == "active_response_speaking":
            routing_decision["next_node"] = "diagnose_speaking"
            routing_decision["rationale"] = "Routing to speaking diagnosis based on task stage"
        elif task_stage == "active_response_writing":
            routing_decision["next_node"] = "diagnose_writing"
            routing_decision["rationale"] = "Routing to writing diagnosis based on task stage"
        elif task_stage == "live_writing_analysis":
            routing_decision["next_node"] = "analyze_live_writing"
            routing_decision["rationale"] = "Routing to live writing analysis based on task stage"
        elif task_stage == "viewing_prompt":
            routing_decision["next_node"] = "provide_scaffolding"
            routing_decision["rationale"] = "Routing to scaffolding based on task stage"
        elif task_stage == "practice_selection":
            routing_decision["next_node"] = "select_practice"
            routing_decision["rationale"] = "Routing to practice selection based on task stage"
        elif task_stage == "curriculum_navigation":
            routing_decision["next_node"] = "determine_next_step"
            routing_decision["rationale"] = "Routing to curriculum navigation based on task stage"
        elif task_stage == "testing_specific_context_from_button":
            routing_decision["next_node"] = "feedback_for_test_button_node"
            routing_decision["rationale"] = "Routing to test button feedback based on task stage"
    
    # If Gemini is available, we can use it for more intelligent routing
    if gemini_model and current_message:
        try:
            # Get the appropriate prompt template
            system_prompt = PROMPTS.get("initial_router", {}).get("system_prompt", 
                """You are an expert TOEFL tutor analyzing a student's request.
                Determine the most appropriate routing path based on the message content and context.
                Consider the student's profile, current task stage, and message intent.
                Format your response as a JSON with:
                {
                    "next_node": "One of: diagnose_speaking, diagnose_writing, analyze_live_writing, provide_scaffolding, select_practice, determine_next_step, generate_feedback, process_conversational_turn, handle_unmatched_interaction",
                    "rationale": "Explanation of why this routing decision was made",
                    "detected_intent": "Brief description of the detected student intent"
                }
                """)
            
            # Replace placeholders with actual data
            system_prompt = system_prompt.replace("{{current_context}}", json.dumps(context))
            system_prompt = system_prompt.replace("{{student_memory_context}}", json.dumps(student_data))
            system_prompt = system_prompt.replace("{{current_message}}", current_message)
            
            user_prompt = PROMPTS.get("initial_router", {}).get("user_prompt", 
                "Please analyze this request and determine the appropriate routing path.")
            
            # Create content for the model
            contents = [
                Content(role="user", parts=[system_prompt]),
                Content(role="model", parts=["I'll analyze this request and determine routing."]),
                Content(role="user", parts=[user_prompt])
            ]
            
            # Call the Gemini model
            response = gemini_model.generate_content(contents, generation_config={
                "temperature": 0.3,  # Lower temperature for more predictable routing
                "max_output_tokens": 512
            })
            
            response_text = response.text
            
            # Parse the JSON response
            try:
                # Extract JSON if it's embedded in a code block
                if "```json" in response_text and "```" in response_text:
                    json_text = response_text.split("```json")[1].split("```")[0].strip()
                    ai_routing = json.loads(json_text)
                elif response_text.strip().startswith("{") and response_text.strip().endswith("}"):
                    ai_routing = json.loads(response_text)
                else:
                    # If not in JSON format, use default routing
                    logger.warning("InitialRouterNode: Response not in JSON format, using default routing")
                    ai_routing = routing_decision
                
                # Validate that the next_node is valid
                valid_nodes = [
                    "diagnose_speaking", "diagnose_writing", "analyze_live_writing", 
                    "provide_scaffolding", "select_practice", "determine_next_step", 
                    "generate_feedback", "process_conversational_turn", 
                    "handle_unmatched_interaction", "apply_teacher_persona"
                ]
                
                if "next_node" in ai_routing and ai_routing["next_node"] in valid_nodes:
                    routing_decision = ai_routing
                    logger.info(f"InitialRouterNode: AI routing to {routing_decision['next_node']}")
                else:
                    logger.warning(f"InitialRouterNode: Invalid next_node in AI routing: {ai_routing.get('next_node')}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for routing: {e}")
                # Continue with the default routing decision
                
        except Exception as e:
            logger.error(f"Error in AI-based routing: {e}")
            # Continue with the default routing decision
    
    # Log the final routing decision
    logger.info(f"InitialRouterNode: Routing to {routing_decision['next_node']} with rationale: {routing_decision['rationale']}")
    
    return {"routing_decision": routing_decision}
