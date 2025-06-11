import logging
import pandas as pd
from typing import Dict, Any, Optional
import os
from state import AgentGraphState

logger = logging.getLogger(__name__)

# Define the expected path to the CSV file relative to this script or an absolute path
# For now, let's assume it's in the same directory as the main application or a known path.
# IMPORTANT: Adjust this path if your CSV is located elsewhere.
TEACHING_DATA_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'teaching_data.csv') # Assumes it's in the parent directory of 'agents'
# Or use an absolute path if preferred: 
# TEACHING_DATA_CSV_PATH = r'c:\Users\Adya\OneDrive\Desktop\new\backend_ai_service_langgraph\teaching_data.csv'

async def teaching_rag_node(state: AgentGraphState) -> Dict[str, Any]:
    print("--- DEBUG: Entered teaching_rag_node ---") # ADDED FOR DEBUGGING
    logger.info("TeachingRAGNode: Processing...")
    
    current_context_obj = state.get('current_context') # This should be an InteractionRequestContext instance or None

    if not current_context_obj: # Check if current_context_obj is None
        error_message = "TeachingRAGNode: 'current_context' is missing from the agent state."
        logger.error(error_message)
        logger.debug(f"TeachingRAGNode: Received current_context_obj as: {current_context_obj} (type: {type(current_context_obj)})_DEBUG")
        return {"retrieved_teaching_row": None, "rag_error": error_message}
    
    # Ensure it's the correct type, though the error implies it is.
    # from models import InteractionRequestContext # Ensure this import exists at the top of the file if not already
    # if not isinstance(current_context_obj, InteractionRequestContext):
    #     error_message = f"TeachingRAGNode: 'current_context' is not of type InteractionRequestContext. Got {type(current_context_obj)}"
    #     logger.error(error_message)
    #     return {"retrieved_teaching_row": None, "rag_error": error_message}

    teacher_persona = current_context_obj.teacher_persona
    learning_objective_id = current_context_obj.learning_objective_id
    student_proficiency = current_context_obj.student_proficiency_level
    
    retrieved_teaching_row: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    if not all([teacher_persona, learning_objective_id, student_proficiency]):
        error_message = "Missing one or more required inputs for RAG: teacher_persona, learning_objective_id, student_proficiency_level."
        logger.error(f"TeachingRAGNode: {error_message}")
        return {"retrieved_teaching_row": None, "rag_error": error_message}

    try:
        # Ensure the CSV path is correct for your environment
        # Adjust if TEACHING_DATA_CSV_PATH needs to be different
        # For example, if your script is in backend_ai_service_langgraph/agents/ and csv is in backend_ai_service_langgraph/
        csv_path = r'c:\Users\Adya\OneDrive\Desktop\new\backend_ai_service_langgraph\teaching_data.csv'
        print(f"--- DEBUG: Attempting to read CSV from: {csv_path} ---") # ADDED FOR DEBUGGING
        if not os.path.exists(csv_path):
            error_message = f"Teaching data CSV file not found at {csv_path}"
            logger.error(f"TeachingRAGNode: {error_message}")
            return {"retrieved_teaching_row": None, "rag_error": error_message}
            
        df = pd.read_csv(csv_path)
        logger.info(f"TeachingRAGNode: Loaded {len(df)} rows from {csv_path}")

        # Filter based on the criteria
        # Note: CSV column names must exactly match 'TEACHER_PERSONAS', 'LEARNING_OBJECTIVE', 'STUDENT_PROFICIENCY'
        filtered_df = df[
            (df['TEACHER_PERSONAS'] == teacher_persona) &
            (df['LEARNING_OBJECTIVE'].str.startswith(str(learning_objective_id) + ":")) &
            (df['STUDENT_PROFICIENCY'] == student_proficiency)
        ]
        
        if not filtered_df.empty:
            # Get the first matching row as a dictionary
            # Convert NaN to None for JSON compatibility if needed later, though dict itself is fine
            retrieved_teaching_row = filtered_df.iloc[0].where(pd.notnull(filtered_df.iloc[0]), None).to_dict()
            logger.info(f"TeachingRAGNode: Found matching teaching row for Persona: {teacher_persona}, LO: {learning_objective_id}, Proficiency: {student_proficiency}")
        else:
            error_message = f"No matching teaching row found for Persona: {teacher_persona}, LO: {learning_objective_id}, Proficiency: {student_proficiency}."
            logger.warning(f"TeachingRAGNode: {error_message}")
            
    except FileNotFoundError:
        error_message = f"Teaching data CSV file not found at path: {csv_path}"
        logger.error(f"TeachingRAGNode: {error_message}", exc_info=True)
    except pd.errors.EmptyDataError:
        error_message = f"Teaching data CSV file is empty: {csv_path}"
        logger.error(f"TeachingRAGNode: {error_message}", exc_info=True)
    except KeyError as e:
        error_message = f"Missing expected column in CSV: {e}. Ensure CSV has 'TEACHER_PERSONAS', 'LEARNING_OBJECTIVE', 'STUDENT_PROFICIENCY'."
        logger.error(f"TeachingRAGNode: {error_message}", exc_info=True)
    except Exception as e:
        error_message = f"An unexpected error occurred during teaching RAG: {e}"
        logger.error(f"TeachingRAGNode: {error_message}", exc_info=True)

    return {
        "retrieved_teaching_row": retrieved_teaching_row,
        "rag_error": error_message # To allow downstream nodes to know if RAG failed
    }
