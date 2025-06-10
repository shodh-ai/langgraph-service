import os
from dotenv import load_dotenv
load_dotenv() # Load .env variables at the very beginning

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import uuid
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List
from graph_builder import build_graph
from state import AgentGraphState
from models import (
    InteractionRequest,
    InteractionResponse,
    InteractionRequestContext,
    ReactUIAction,
)

# Configure basic logging for the application
# This will set the root logger level and format, affecting all module loggers unless they are specifically configured otherwise.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

load_dotenv()

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="TOEFL Tutor AI Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

toefl_tutor_graph = build_graph()


@app.post("/process_interaction", response_model=InteractionResponse)
async def process_interaction_route(request_data: InteractionRequest):
    default_user_id = "default_user_for_testing"
    default_session_id = str(uuid.uuid4())

    user_id = default_user_id
    if request_data.current_context and request_data.current_context.user_id:
        user_id = request_data.current_context.user_id

    session_id = request_data.session_id or default_session_id
    context = request_data.current_context or InteractionRequestContext(user_id=user_id)
    chat_history = request_data.chat_history
    transcript = request_data.transcript
    full_submitted_transcript = None

    if context.task_stage == "speaking_task_submitted":
        full_submitted_transcript = transcript

    initial_graph_state = AgentGraphState(
        user_id=user_id,
        user_token=request_data.usertoken,
        session_id=session_id,
        transcript=transcript,
        full_submitted_transcript=full_submitted_transcript,
        current_context=context,
        chat_history=chat_history,
        question_stage=context.question_stage,
        student_memory_context=None,
        next_task_details=None,
        diagnosis_result=None,
        output_content=None,
        feedback_content=None,
        # Fields from local feedback-system branch
        estimated_overall_english_comfort_level=context.estimated_english_comfort_level,
        initial_impression=context.initial_impression,
        fluency=context.fluency,
        grammar=context.grammar,
        vocabulary=context.vocabulary,
        question_one_answer=context.question_one_answer,
        question_two_answer=context.question_two_answer,
        question_three_answer=context.question_three_answer,
        # Explicitly add modelling context fields to the state
        example_prompt_text=context.example_prompt_text,
        student_goal_context=context.student_goal_context,
        student_confidence_context=context.student_confidence_context,
        teacher_initial_impression=context.teacher_initial_impression,
        student_struggle_context=context.student_struggle_context,
        english_comfort_level=context.english_comfort_level,
    )

    try:
        config = {"configurable": {"thread_id": session_id}}
        final_state = await toefl_tutor_graph.ainvoke(
            initial_graph_state, config=config
        )

        output_content: Optional[Dict[str, Any]] = final_state.get("output_content")
        if output_content is None:
            output_content = final_state.get("feedback_content", {})

        response_text = ""
        if output_content:
            response_text = output_content.get(
                "response",
                output_content.get("text_for_tts", output_content.get("text", "")),
            )

        if not response_text:
            response_text = (
                "No response text was generated. Please check the system logs."
            )

        ui_actions = None
        if output_content:
            ui_actions = output_content.get("ui_actions") or output_content.get(
                "dom_actions"
            )
            if ui_actions:
                for action in ui_actions:
                    if "action_type" in action:
                        action["action_type_str"] = action.get("action_type")
                        del action["action_type"]
                    elif "action_type_str" not in action:
                        action["action_type_str"] = None

        next_task = final_state.get("next_task_details")
        if next_task:
            if ui_actions is None:
                ui_actions = []

            has_task_button = any(
                action.get("action_type_str") == "DISPLAY_NEXT_TASK_BUTTON"
                for action in ui_actions
            )

            if not has_task_button and next_task:
                mapped_action_type_str = "SHOW_ALERT"
                task_title = next_task.get("title", "Unknown Task")
                task_desc = next_task.get("description", "")
                mapped_parameters = {"message": f"Next Task: {task_title}\n{task_desc}"}

                ui_actions.append(
                    {
                        "action_type_str": mapped_action_type_str,
                        "parameters": mapped_parameters,
                    }
                )

        ui_actions_list: Optional[List[ReactUIAction]] = None
        if ui_actions:
            ui_actions_list = []
            for action_dict in ui_actions:
                action_type = action_dict.get("action_type_str", "")

                target_element_id = None
                if "target_element_id" in action_dict:
                    target_element_id = action_dict["target_element_id"]
                elif "targetElementId" in action_dict:
                    target_element_id = action_dict["targetElementId"]

                parameters = action_dict.get("parameters", {})

                ui_actions_list.append(
                    ReactUIAction(
                        action_type=action_type,
                        target_element_id=target_element_id,
                        parameters=parameters,
                    )
                )

        response = InteractionResponse(
            response=response_text, ui_actions=ui_actions_list
        )

        return response

    except Exception as e:
        logger.error(f"Exception in /process_interaction: {e}", exc_info=True)
        # The following import and print_exc are for more verbose console output if needed, 
        # but logger.error with exc_info=True should capture it well.
        # import traceback
        # traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error during AI processing: {str(e)}"
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    import os

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5005")))
