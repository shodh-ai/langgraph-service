from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import uuid
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List

from graph_builder import build_graph
from state import AgentGraphState
from models import InteractionRequest, InteractionResponse, InteractionRequestContext, ReactUIAction

load_dotenv()

# Configure logging
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="TOEFL Tutor AI Backend", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the graph when the app starts
toefl_tutor_graph = build_graph()
logger.info("TOEFL Tutor LangGraph with P1 and P2 flows compiled and ready.")


@app.post("/process_interaction", response_model=InteractionResponse)
async def process_interaction_route(request_data: InteractionRequest):
    logger.info(f"FastAPI backend: Processing interaction request")
    logger.debug(f"Request data: {request_data.model_dump(exclude_none=True)}")
    
    # Extract and validate data from the request
    # Default values
    default_user_id = "default_user_for_testing"
    default_session_id = str(uuid.uuid4())
    
    # Extract user_id
    user_id = default_user_id
    if request_data.current_context and request_data.current_context.user_id:
        user_id = request_data.current_context.user_id
    
    # Extract session_id
    session_id = request_data.session_id or default_session_id
    
    # Extract context
    context = request_data.current_context or InteractionRequestContext(user_id=user_id)
    
    # Extract chat history
    chat_history = request_data.chat_history
    
    # Extract transcript - differentiate between regular transcript and full submitted transcript
    transcript = request_data.transcript
    full_submitted_transcript = None
    
    # If this is a task submission, set the full_submitted_transcript
    if context.task_stage == "speaking_task_submitted":
        full_submitted_transcript = transcript
        logger.info(f"Processing speaking submission, transcript length: {len(full_submitted_transcript or '')}")
    
    logger.info(f"/process_interaction: user_id='{user_id}', session_id='{session_id}', task_stage='{context.task_stage}'")
    
    # Construct the initial graph state
    initial_graph_state = AgentGraphState(
        # User and session identifiers
        user_id=user_id,
        session_id=session_id,
        
        # Input data
        transcript=transcript,
        full_submitted_transcript=full_submitted_transcript,
        current_context=context,
        chat_history=chat_history,
        
        # These will be populated by nodes
        student_memory_context=None,
        next_task_details=None,
        diagnosis_result=None,
        output_content=None,
        feedback_content=None  # For backward compatibility
    )

    try:
        # Configure for LangSmith tracing
        config = {"configurable": {"thread_id": session_id}}
        
        # Execute the graph with our initial state
        final_state = await toefl_tutor_graph.ainvoke(initial_graph_state, config=config)
        
        logger.debug(f"LangGraph execution completed")
        
        # Extract output content from the final state
        # First try the new output_content field, fall back to feedback_content for backward compatibility
        output_content: Optional[Dict[str, Any]] = final_state.get("output_content")
        if output_content is None:
            # Fall back to feedback_content for backward compatibility
            output_content = final_state.get("feedback_content", {})
            logger.info("Using feedback_content for backward compatibility")
        
        # Extract response text from output_content
        response_text = ""
        if output_content:
            # First look for the new field name 'response', then fall back to old 'text_for_tts'
            response_text = output_content.get("response", output_content.get("text_for_tts", output_content.get("text", "")))
        
        if not response_text:
            response_text = "No response text was generated. Please check the system logs."
            logger.warning("No response text found in output_content")
        
        # Extract UI actions from output_content
        ui_actions = None
        if output_content:
            # Get ui_actions or dom_actions (for backward compatibility)
            ui_actions = output_content.get("ui_actions") or output_content.get("dom_actions")
            # Convert action_type to action_type_str
            if ui_actions:
                for action in ui_actions:
                    # If 'action_type' exists, use it to set 'action_type_str' and then delete 'action_type'
                    if "action_type" in action:
                        action["action_type_str"] = action.get("action_type")
                        del action["action_type"]
                    # If 'action_type' does not exist, but 'action_type_str' does, ensure 'action_type_str' is preserved.
                    # If neither exists, action_type_str will be None or not set, which is handled later.
                    elif "action_type_str" not in action:
                        action["action_type_str"] = None # Explicitly set to None if neither was found
        
        # Extract next task details if available
        next_task = final_state.get("next_task_details")
        if next_task:
            logger.info(f"Next task details available: {next_task.get('prompt_id', 'unknown')}")
            # Add next task details to frontend_rpc_calls if not already in ui_actions
            if ui_actions is None:
                ui_actions = []
            
            # Check if we already have a task button action
            has_task_button = any(action.get("action_type_str") == "DISPLAY_NEXT_TASK_BUTTON" for action in ui_actions)
            
            if not has_task_button and next_task:
                # Map non-standard action types to standard ones supported by the frontend
                mapped_action_type_str = "SHOW_ALERT" # Use string representation of enum
                task_title = next_task.get("title", "Unknown Task")
                task_desc = next_task.get("description", "")
                mapped_parameters = {
                    "message": f"Next Task: {task_title}\n{task_desc}"
                }
                logger.info(f"Mapping next_task to UI action: {mapped_action_type_str}")
                
                ui_actions.append({
                    "action_type_str": mapped_action_type_str, # Use 'action_type_str'
                    "parameters": mapped_parameters,
                })
        
        # Create the response
        # Convert ui_actions (list of dicts) to List[ReactUIAction]
        ui_actions_list: Optional[List[ReactUIAction]] = None
        if ui_actions: # ui_actions is the list of dicts prepared by current app.py logic
            ui_actions_list = []
            for action_dict in ui_actions:
                # Extract the action_type from action_type_str (new field name)
                action_type = action_dict.get("action_type_str", "")
                
                # Extract the target_element_id, preferring the standard format if present
                target_element_id = None
                if "target_element_id" in action_dict:
                    target_element_id = action_dict["target_element_id"]
                elif "targetElementId" in action_dict:
                    target_element_id = action_dict["targetElementId"]
                
                # Extract the parameters
                parameters = action_dict.get("parameters", {})
                
                # Create the ReactUIAction with the new field names
                ui_actions_list.append(
                    ReactUIAction(
                        action_type=action_type,
                        target_element_id=target_element_id,
                        parameters=parameters
                    )
                )

        response = InteractionResponse(
            response=response_text,  # Using new field name and new variable name
            ui_actions=ui_actions_list  # New field name: ui_actions instead of dom_actions
        )
        
        logger.info(f"Response prepared, text length: {len(response_text)}, ui_actions: {len(ui_actions_list) if ui_actions_list else 0}")
        return response

    except Exception as e:
        logger.error(f"Error processing LangGraph interaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during AI processing")

# Simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# If running directly with uvicorn for local dev:
if __name__ == "__main__":
    import uvicorn
    import os # Added import for os.getenv
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5005")))