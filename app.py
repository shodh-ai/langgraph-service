from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # If your LB/LiveKit agent is on a different domain
import logging
import uuid
from dotenv import load_dotenv

from graph_builder import build_graph
from state import AgentGraphState
from models import InteractionRequest, InteractionResponse, InteractionRequestContext

load_dotenv()

# Configure logging (FastAPI uses uvicorn's logger by default, but can be customized)
logger = logging.getLogger("uvicorn.error") # Or your custom logger
# logging.basicConfig(level=logging.DEBUG) # For local debugging

app = FastAPI(title="TOEFL Tutor AI Backend", version="0.1.0")

# CORS middleware (optional, depending on your setup)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the graph when the app starts - ensure this is efficient
# For production, consider how often this needs to be rebuilt if graph def changes
toefl_tutor_graph = build_graph()
logger.info("TOEFL Tutor LangGraph compiled and ready.")


@app.post("/process_interaction", response_model=InteractionResponse)
async def process_interaction_route(request_data: InteractionRequest):
    # Define default values
    default_user_id = "default_user_for_testing"
    default_session_id = str(uuid.uuid4()) # Generate a new session ID if none provided
    default_context_data = InteractionRequestContext(user_id=default_user_id) # Default context

    user_id_to_use = default_user_id
    session_id_to_use = default_session_id
    actual_context = default_context_data
    chat_history_to_use = request_data.chat_history
    transcript_to_use = request_data.transcript

    if request_data.current_context:
        actual_context = request_data.current_context # it's already a Pydantic model
        if request_data.current_context.user_id:
            user_id_to_use = request_data.current_context.user_id
    
    if request_data.session_id:
        session_id_to_use = request_data.session_id

    logger.debug(f"/process_interaction: user_id='{user_id_to_use}', session_id='{session_id_to_use}', transcript='{transcript_to_use}'")

    initial_graph_state = AgentGraphState(
        user_id=user_id_to_use,
        session_id=session_id_to_use,
        transcript=transcript_to_use,
        current_context=actual_context, # Use the determined context
        chat_history=chat_history_to_use,
        student_memory_context=None, # Will be loaded by a node
        diagnosis_result=None,     # Will be populated by a node
        feedback_content=None,     # Will be populated by a node
    )

    try:
        # Ensure thread_id for LangSmith is always present
        config = {"configurable": {"thread_id": session_id_to_use}} # For LangSmith tracing
        
        # Use ainvoke for async graph execution if your nodes are async
        final_state = await toefl_tutor_graph.ainvoke(initial_graph_state, config=config)
        
        logger.debug(f"LangGraph final state: {final_state}")

        feedback_content = final_state.get("feedback_content", {})
        response_text = feedback_content.get("text", "FastAPI/LangGraph: No final text response.")
        dom_actions_data = feedback_content.get("dom_actions")

        return InteractionResponse(response=response_text, dom_actions=dom_actions_data)

    except Exception as e:
        logger.error(f"Error processing LangGraph interaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during AI processing")

# Simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# If running directly with uvicorn for local dev:
# if __name__ == "__main__":
#     import uvicorn
#     import os # Added import for os.getenv
#     uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
