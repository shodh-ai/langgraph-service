import json
from state import AgentGraphState
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def query_document_node(state: AgentGraphState) -> dict:
    logger.info(
        f"QueryDocumentNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )

    # Extract filters from state
    current_context = state.get("current_context", {})
    task_stage = current_context.get("task_stage", "")
    question_stage = state.get("question_stage", task_stage) # Use task_stage as fallback
    
    user_data = state.get("user_data", {})
    if user_data == {}:
        user_level = "Beginner"
    else:
        user_level = user_data.get("level", "Beginner")
    
    primary_error = state.get("primary_error", "")
    
    logger.info(f"QueryDocumentNode: Question Stage: {question_stage}")
    logger.info(f"QueryDocumentNode: User Level: {user_level}")
    logger.info(f"QueryDocumentNode: Primary Error: {primary_error}")

    data_path = Path(__file__).parent.parent / "data" / "feedback.csv"
    df = pd.read_csv(data_path)
    
    # Apply filters sequentially and keep track of dataframe size
    # Only apply filter if the filter value is not empty
    if question_stage:
        df_filtered = df[df["Task"] == question_stage]
        # If filter results in empty dataframe, revert to the original
        if not df_filtered.empty:
            df = df_filtered
    
    if user_level:
        df_filtered = df[df["Proficiency"] == user_level]
        if not df_filtered.empty:
            df = df_filtered
    
    if primary_error:
        df_filtered = df[df["Error"] == primary_error]
        if not df_filtered.empty:
            df = df_filtered
    
    # If we still have no data, take a fallback approach - get some data regardless of filters
    if df.empty:
        logger.warning("No data matched filters, using fallback data selection")
        df = pd.read_csv(data_path)
        # Limit to at most 5 rows as a fallback
        df = df.head(5)
    
    print(f"Data shape after filtering: {df.shape}")

    # Always return at least the available data
    document_data = df.to_dict(orient="records")
    logger.info(f"QueryDocumentNode: Returning {len(document_data)} documents")

    return {"document_data": document_data}
