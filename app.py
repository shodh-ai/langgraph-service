# app.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.concurrency import run_in_threadpool # Not explicitly used in new version, can be removed if not needed elsewhere
import logging
import uuid
import json
import asyncio
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List # Keep for models if not fully typed

from graph_builder import build_graph
from state import AgentGraphState
from models import InteractionRequest, InteractionResponse, ReactUIAction # Consolidate model imports

# Load environment variables from .env file
load_dotenv()

# Configure logging
# Using uvicorn's logger for consistency if running with uvicorn
# BasicConfig can be used as a fallback or for non-uvicorn environments
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("uvicorn.error") # Standard for Uvicorn, captures its logs and app logs if configured

app = FastAPI(
    title="TOEFL Tutor AI Service",
    version="0.1.0",
    description="A LangGraph-based AI service for TOEFL tutoring."
)

# Add CORS middleware (from original app.py)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the graph when the application starts
toefl_tutor_graph = build_graph()

async def stream_langgraph_response(request_data: InteractionRequest):
    """An async generator function that will stream the LangGraph output."""
    initial_graph_state = AgentGraphState(
        user_id=request_data.current_context.user_id,
        session_id=request_data.session_id,
        transcript=request_data.transcript,
        current_context=request_data.current_context,
        chat_history=request_data.chat_history,
        # Initialize all other fields from AgentGraphState to None or default values
        # This ensures all keys are present when the graph starts.
        user_token=request_data.usertoken, # from original app.py
        full_submitted_transcript=request_data.transcript if request_data.current_context.task_stage == "speaking_task_submitted" else None, # from original
        question_stage=request_data.current_context.question_stage, # from original
        student_memory_context=None,
        task_stage=request_data.current_context.task_stage, # from current_context
        next_task_details=None,
        diagnosis_result=None, # from original
        error_details=None, # Assuming this might be a field
        document_query_result=None,
        rag_query_result=None,
        feedback_plan=None,
        feedback_output=None,
        feedback_content=None, # from original
        scaffolding_analysis=None,
        scaffolding_retrieval_result=None,
        scaffolding_plan=None,
        scaffolding_output=None,
        teaching_module_state=None, 
        p1_curriculum_navigator_output=None,
        conversation_response=None,
        output_content=None, # As per the user's new app.py
    )
    
    config = {"configurable": {"thread_id": request_data.session_id}}
    
    logger.debug(f"Starting graph astream with initial state for session {request_data.session_id}")
    async for chunk in toefl_tutor_graph.astream(initial_graph_state, config=config):
        logger.debug(f"Graph stream chunk received: {chunk}")
        
        # The key in the chunk will be the name of the node that just ran.
        # We are interested in the output from our designated output formatting node.
        # In graph_builder.py, this is NODE_FORMAT_FINAL_OUTPUT, which is 'format_final_output_node'
        if "format_final_output_node" in chunk: 
            node_output = chunk.get("format_final_output_node", {})
            
            if "streaming_text_chunk" in node_output:
                text_chunk = node_output.get("streaming_text_chunk")
                if text_chunk:
                    # Ensure text_chunk is serializable (e.g. string)
                    sse_formatted_chunk = f"data: {json.dumps({'type': 'tts_chunk', 'content': str(text_chunk)})}\n\n"
                    yield sse_formatted_chunk
                    logger.debug(f"Yielded tts_chunk: {text_chunk}")

            if "final_ui_actions" in node_output:
                ui_actions = node_output.get("final_ui_actions")
                if ui_actions:
                    sse_formatted_chunk = f"data: {json.dumps({'type': 'ui_actions', 'content': ui_actions})}\n\n"
                    yield sse_formatted_chunk
                    logger.debug(f"Yielded ui_actions: {ui_actions}")
        else:
            # Log other node outputs if necessary for debugging, but don't stream them directly
            # unless they are specifically formatted for streaming.
            # Example: logger.debug(f"Node {list(chunk.keys())[0]} output: {list(chunk.values())[0]}")
            pass

    logger.debug(f"Graph astream finished for session {request_data.session_id}.")


@app.post("/process_interaction_streaming", response_model=None) # Response model is handled by StreamingResponse
async def process_interaction_streaming_route(request_data: InteractionRequest):
    logger.info(f"Streaming request received for user '{request_data.current_context.user_id}' session '{request_data.session_id}'")
    try:
        return StreamingResponse(stream_langgraph_response(request_data), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Error in streaming endpoint: {e}", exc_info=True)
        # It's important to raise HTTPException for FastAPI to handle it correctly
        raise HTTPException(status_code=500, detail=f"Error processing streaming request: {str(e)}")

# Renamed original endpoint for non-streaming for clarity and to avoid conflict
@app.post("/process_interaction_non_streaming", response_model=InteractionResponse)
async def process_interaction_non_streaming_route(request_data: InteractionRequest):
    logger.info(f"Non-streaming request received for user '{request_data.current_context.user_id}' session '{request_data.session_id}'")
    try:
        initial_graph_state = AgentGraphState(
            user_id=request_data.current_context.user_id,
            session_id=request_data.session_id,
            transcript=request_data.transcript,
            current_context=request_data.current_context,
            chat_history=request_data.chat_history,
            user_token=request_data.usertoken,
            full_submitted_transcript=request_data.transcript if request_data.current_context.task_stage == "speaking_task_submitted" else None,
            question_stage=request_data.current_context.question_stage,
            student_memory_context=None,
            task_stage=request_data.current_context.task_stage,
            next_task_details=None,
            diagnosis_result=None,
            error_details=None,
            document_query_result=None,
            rag_query_result=None,
            feedback_plan=None,
            feedback_output=None,
            feedback_content=None,
            scaffolding_analysis=None,
            scaffolding_retrieval_result=None,
            scaffolding_plan=None,
            scaffolding_output=None,
            teaching_module_state=None,
            p1_curriculum_navigator_output=None,
            conversation_response=None,
            output_content=None,
        )
        config = {"configurable": {"thread_id": request_data.session_id}}
        
        logger.debug(f"Invoking graph with initial state for non-streaming: {initial_graph_state}")
        final_state = await toefl_tutor_graph.ainvoke(initial_graph_state, config=config)
        logger.debug(f"Graph invocation completed. Final state: {final_state}")
        
        # Logic from original /process_interaction endpoint to construct InteractionResponse
        output_content_val: Optional[Dict[str, Any]] = final_state.get("output_content")
        if output_content_val is None:
            output_content_val = final_state.get("feedback_content", {})

        response_text = ""
        if output_content_val:
            response_text = output_content_val.get(
                "response",
                output_content_val.get("text_for_tts", output_content_val.get("text", "")),
            )

        if not response_text:
            response_text = "No response text was generated. Please check the system logs."
            logger.warning(f"No response text in final_state for session {request_data.session_id}")

        ui_actions_raw = None
        if output_content_val:
            ui_actions_raw = output_content_val.get("ui_actions") or output_content_val.get("dom_actions")
            if ui_actions_raw:
                for action in ui_actions_raw:
                    if "action_type" in action: # Ensure consistent key naming
                        action["action_type_str"] = action.pop("action_type")
                    elif "action_type_str" not in action:
                        action["action_type_str"] = None
        
        next_task = final_state.get("next_task_details")
        if next_task:
            if ui_actions_raw is None: ui_actions_raw = []
            has_task_button = any(a.get("action_type_str") == "DISPLAY_NEXT_TASK_BUTTON" for a in ui_actions_raw)
            if not has_task_button:
                ui_actions_raw.append({
                    "action_type_str": "SHOW_ALERT",
                    "parameters": {"message": f"Next Task: {next_task.get('title', 'N/A')}\n{next_task.get('description', '')}"}
                })

        ui_actions_list: Optional[List[ReactUIAction]] = None
        if ui_actions_raw:
            ui_actions_list = [
                ReactUIAction(
                    action_type=a.get("action_type_str", ""),
                    target_element_id=a.get("target_element_id") or a.get("targetElementId"),
                    parameters=a.get("parameters", {})
                ) for a in ui_actions_raw
            ]

        return InteractionResponse(response=response_text, ui_actions=ui_actions_list, session_id=request_data.session_id)

    except Exception as e:
        logger.error(f"Error in non-streaming endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing non-streaming request: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import os
    # Use PORT from env, default to 8000 (common for dev, original used 5005)
    port = int(os.getenv("PORT", "8000")) 
    logger.info(f"Starting Uvicorn server on host 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
