import os
import asyncio
from dotenv import load_dotenv

# Load environment variables only once at the beginning
load_dotenv()
if os.getenv("MEM0_API_KEY"):
    print(f"--- DEBUG: MEM0_API_KEY is set. ---")
else:
    print("--- WARNING: MEM0_API_KEY is NOT set. Mem0 functionalities will fail. ---")

import logging
# Configure basic logging early, this might be reconfigured by uvicorn but good for initial script execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger("uvicorn.error") # Ensure logger is defined before use in endpoints

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import json
import asyncio
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field # For UserRegistrationRequest
from fastapi import UploadFile, File
from deepgram import DeepgramClient
from deepgram import PrerecordedOptions

from memory import initialize_memory
import memory
 
from graph_builder import build_graph
from state import AgentGraphState
import uuid

# Define the directory for uploads
UPLOAD_DIR = "uploads"

# +++ DEFINE THE NEW REQUEST MODEL +++
# This matches the payload that livekit-service is sending.
class InvokeTaskRequest(BaseModel):
    task_name: str
    json_payload: str
from contextlib import asynccontextmanager
from models import (
    InteractionRequest,
    InteractionResponse,
    InteractionRequestContext,
    ReactUIAction,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

toefl_tutor_graph: Optional[Any] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Initializes the global memory instance and the graph with its checkpointer.
    """
    global toefl_tutor_graph
    logger.info("--- Application startup: Initializing memory and graph ---")
    
    initialize_memory()
    logger.info("--- Application startup: StudentProfileMemory initialized ---")

    async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        toefl_tutor_graph = build_graph(checkpointer)
        logger.info("--- Application startup: Main graph compiled with SQLite checkpointer ---")
        
        yield
    
    logger.info("--- Application shutdown: Resources cleaned up ---")


app = FastAPI(
    title="ShodhAI Langgraph",
    version="0.1.0",
    description="A LangGraph-based AI service for academic tutoring.",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The graph is now initialized within the lifespan manager
# toefl_tutor_graph = build_graph()

# --- Deepgram Transcription Proxy Endpoint ---
@app.post("/transcribe_audio")
async def transcribe_audio(file: UploadFile = File(...)):
    logger.info("Received request for /transcribe_audio")
    try:
        deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        if not deepgram_api_key:
            logger.error("DEEPGRAM_API_KEY not set in environment.")
            raise HTTPException(status_code=500, detail="Server configuration error: Missing Deepgram API key.")

        # Initialize Deepgram client
        deepgram = DeepgramClient(deepgram_api_key)

        audio_data = await file.read()
        source = {'buffer': audio_data, 'mimetype': file.content_type}

        # Set transcription options
        options = PrerecordedOptions(
            model='nova-2',
            smart_format=True,
            language='en-US'
        )

        # Call Deepgram API
        logger.info("Sending audio to Deepgram for transcription...")
        response = await deepgram.listen.asyncprerecorded.v("1").transcribe_file(source, options)
        
        # Extract transcript
        transcript = response.results.channels[0].alternatives[0].transcript
        logger.info(f"Successfully transcribed audio. Transcript: {transcript[:100]}...")
        
        return {"transcript": transcript}

    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")


# --- User Registration Endpoint (from origin/subgraphs) ---
class UserRegistrationRequest(BaseModel):
    user_id: str = Field(..., description="The unique identifier for the user.")
    name: str
    goal: str
    feeling: str
    confidence: str

async def create_initial_state(request_data: InvokeTaskRequest) -> AgentGraphState:
    """
    Build the state for *this turn only* from the incoming request.
    No attempt is made to load or merge prior state – LangGraph's
    checkpointer handles persistence and merging automatically.
    """
    # 1. Parse the incoming JSON payload safely
    try:
        payload = json.loads(request_data.json_payload)
    except json.JSONDecodeError:
        logger.error("Failed to decode json_payload. Using empty payload.")
        payload = {}

    # 2. Core identifiers - THE FIX
    # The user_id is nested inside the context object in the payload.
    incoming_context = payload.get("current_context", {})
    user_id = incoming_context.get("user_id", "unknown_user") # Correctly extract from context
    session_id = payload.get("session_id", str(uuid.uuid4()))

    # 3. Current turn context (can be empty or partial)
    incoming_context = payload.get("current_context", {})
    # Ensure at minimum the user_id is present for downstream logic
    incoming_context.setdefault("user_id", user_id)

    # 4. Determine task_name for routing. Prefer the explicit `task_name` from the request object,
    #    but fall back to any task_stage present in the incoming_context.
    task_name = request_data.task_name or incoming_context.get("task_stage")

    logger.info(f"[create_initial_state] Building turn state for task: '{task_name}' | session_id: '{session_id}'")

    # 5. Construct the minimal state dict. Fields not set here will be
    #    populated by the graph or restored from the checkpoint after merge.
    initial_state: AgentGraphState = {
        "user_id": user_id,
        "session_id": session_id,
        "task_name": task_name,

        # Turn-specific inputs
        "transcript": payload.get("transcript"),
        "chat_history": payload.get("chat_history", []),
        "incoming_context": incoming_context, # Use the new dedicated key

        # Place-holders / defaults for the remainder of the AgentGraphState keys.
        # They will either be set by nodes or restored by the checkpointer merge.
        "current_context": {}, # Will be populated from checkpoint
        "rag_document_data": [],
        "intermediate_modelling_payload": None,
        "intermediate_teaching_payload": None,
        "intermediate_scaffolding_payload": None,
        "intermediate_feedback_payload": None,
        "intermediate_cowriting_payload": None,
        "intermediate_pedagogy_payload": None,
    }

    return initial_state

@app.post("/invoke_task_streaming")
async def invoke_task_streaming_route(request_data: InvokeTaskRequest):
    """
    Receives a generic task, unpacks the payload, and starts the graph stream.
    This is the new, preferred entry point.
    """
    try:
        initial_graph_state = await create_initial_state(request_data)

        # Create the config for the graph invocation, which is essential for memory
        config = {"configurable": {"thread_id": initial_graph_state.get("session_id")}, "recursion_limit": 50}

        # Call the SSE streamer with the prepared state and config
        return StreamingResponse(
            stream_graph_responses_sse(initial_graph_state, config),
            media_type="text/event-stream"
        )
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode json_payload: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON in payload: {e}")
    except Exception as e:
        logger.error(f"Error in invoke_task_streaming: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/user/register")
async def register_user(registration_data: UserRegistrationRequest):
    logger.info(f"Received registration data for user_id: {registration_data.user_id}")
    try:
        profile_data = registration_data.model_dump(exclude={"user_id"})
        
        memory.memory_stub.update_student_profile(
            user_id=registration_data.user_id,
            profile_data=profile_data
        )
        
        logger.info(f"Successfully stored profile for user_id: {registration_data.user_id}")
        return {"message": "User profile stored successfully."}
    except Exception as e:
        logger.error(f"Failed to store user profile for {registration_data.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to store user profile.")

# --- Streaming Endpoint --- 
# In app.py

# FINAL, PERFECTED app.py

async def stream_graph_responses_sse(initial_graph_state: AgentGraphState, config: dict):
    """
    This final version listens for the output of a finalizing node and
    streams it back as a single, comprehensive "final_response" event.
    """
    try:
        yield f"event: stream_start\ndata: {json.dumps({'message': 'Stream started'})}\n\n"
        
        FINALIZING_NODES = [
            "conversation_handler", "handle_welcome", "acknowledge_interrupt",
            "modelling_output_formatter", "teaching_output_formatter",
            "scaffolding_output_formatter", "feedback_output_formatter",
            "cowriting_output_formatter", "pedagogy_output_formatter",
        ]

        async for event in toefl_tutor_graph.astream_events(initial_graph_state, config=config, stream_mode="values"):
            event_name = event.get("event")
            node_name = event.get("name")
            
            if event_name == "on_chain_end" and node_name in FINALIZING_NODES:
                logger.info(f"SSE Streamer: Captured final output from node '{node_name}'.")
                output_data = event.get("data", {}).get("output", {})
                
                if not output_data:
                    logger.warning(f"Node '{node_name}' finished but produced no output data.")
                    continue

                # --- THIS IS THE FIX ---
                # We package everything into ONE event.
                final_response_payload = {
                    "text_for_tts": output_data.get("text_for_tts"),
                    "final_ui_actions": output_data.get("final_ui_actions", [])
                }

                logger.info("SSE Streamer: Yielding single 'final_response' event.")
                yield f"event: final_response\ndata: {json.dumps(final_response_payload)}\n\n"

    except Exception as e:
        logger.error(f"Error streaming graph responses: {e}", exc_info=True)
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    finally:
        logger.info("SSE stream finished.")
        yield f"event: stream_end\ndata: {json.dumps({'message': 'Stream complete'})}\n\n"


@app.post("/invoke_task", response_model=Dict[str, Any])
async def invoke_task_route(request_data: InvokeTaskRequest):
    """
    Handles non-streaming requests for fast, simple nodes.
    """
    logger.info(f"Received non-streaming task '{request_data.task_name}'.")
    try:
        initial_state = await create_initial_state(request_data)
        config = {"configurable": {"thread_id": initial_state["session_id"]}, "recursion_limit": 50}
        
        # Use ainvoke for a single, final result.
        final_state = await toefl_tutor_graph.ainvoke(initial_state, config=config)
        
        # Return the final output keys directly.
        return {
            "final_text_for_tts": final_state.get("final_text_for_tts"),
            "final_ui_actions": final_state.get("final_ui_actions", [])
        }
    except Exception as e:
        logger.error(f"Error in non-streaming invoke_task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")




@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn


    # Ensure the upload directory exists
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    
    # Run the FastAPI app with uvicorn
    port = int(os.getenv("PORT", "8080")) 
    logger.info(f"Starting Uvicorn server on host 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
