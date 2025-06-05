import json
from state import AgentGraphState
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def query_document_node(state: AgentGraphState) -> dict:
    logger.info(
        f"QueryDocumentNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )

    question_stage = state.get("question_stage", "")
    user_data = state.get("user_data", {})
    if user_data == {}:
        user_level = "No level, so concider beginner"
    else:
        user_level = user_data.get("level", "Beginner")
    primary_error = state.get("primary_error", "")
    logger.info(f"QueryDocumentNode: Question Stage: {question_stage}")
    logger.info(f"QueryDocumentNode: User Level: {user_level}")
    logger.info(f"QueryDocumentNode: Primary Error: {primary_error}")

    df = pd.read_csv("data.csv")
    df = df[df["Task"] == question_stage]
    df = df[df["Proficiency"] == user_level]
    df = df[df["Error"] == primary_error]
    print(df.shape)

    if not df.empty:
        document_data = df.to_dict(orient="records")
        logger.info(f"QueryDocumentNode: Found {len(document_data)} matching documents")
    else:
        document_data = []
        logger.info("QueryDocumentNode: No matching documents found")

    return {"document_data": document_data}
