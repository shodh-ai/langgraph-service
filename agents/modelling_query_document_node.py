import logging
import pandas as pd
import os
from state import AgentGraphState

logger = logging.getLogger(__name__)

# Define the expected columns based on the CSV structure
EXPECTED_COLUMNS = [
    'Example_Prompt_Text', 'Student_Goal_Context', 'Student_Confidence_Context',
    'English_Comfort_Level', 'Teacher_Initial_Impression', 'Student_Struggle_Context',
    'pre_modeling_setup_script', 'modeling_and_think_aloud_sequence_json',
    'post_modeling_summary_and_key_takeaways',
    'comprehension_check_or_reflection_prompt_for_student',
    'adaptation_for_student_profile_notes'
]

async def modelling_query_document_node(state: AgentGraphState) -> dict:
    """
    Loads and filters the modelling_data.csv based on student context.
    """
    logger.info(
        f"ModellingQueryDocumentNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )

    english_comfort_level = state.get("english_comfort_level", "")
    student_struggle_context = state.get("student_struggle_context", "")
    
    # Determine the path to modelling_data.csv, assuming it's in the project's root directory
    # (i.e., parent directory of the 'agents' directory)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "modelling_data.csv")

    logger.info(f"ModellingQueryDocumentNode: Attempting to load CSV from: {csv_path}")
    logger.info(f"ModellingQueryDocumentNode: Filtering based on English Comfort Level: '{english_comfort_level}' and Student Struggle Context: '{student_struggle_context}'")

    if not os.path.exists(csv_path):
        logger.error(f"ModellingQueryDocumentNode: modelling_data.csv not found at {csv_path}")
        return {"modelling_document_data": [], "error": f"modelling_data.csv not found at {csv_path}"}

    try:
        # Comma is the default separator for pd.read_csv
        df = pd.read_csv(csv_path)
        logger.info(f"ModellingQueryDocumentNode: Successfully loaded CSV. Shape: {df.shape}. Columns: {df.columns.tolist()}")

        # Validate that all expected columns are present
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            logger.error(f"ModellingQueryDocumentNode: Missing expected columns in CSV: {missing_cols}. Available columns: {df.columns.tolist()}")
            return {"modelling_document_data": [], "error": f"Missing columns: {missing_cols}"}

        # Apply filters
        filtered_df = df.copy()
        if english_comfort_level and 'English_Comfort_Level' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['English_Comfort_Level'].astype(str).str.lower() == str(english_comfort_level).lower()]
        
        if student_struggle_context and 'Student_Struggle_Context' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Student_Struggle_Context'].astype(str).str.contains(str(student_struggle_context), case=False, na=False)]

        logger.info(f"ModellingQueryDocumentNode: DataFrame shape after filtering: {filtered_df.shape}")

        if filtered_df.empty:
            logger.warning("ModellingQueryDocumentNode: No matching documents found after filtering with both criteria.")
            # Fallback: if no specific matches with both, try returning entries for the given English_Comfort_Level only
            if english_comfort_level and 'English_Comfort_Level' in df.columns:
                fallback_df = df[df['English_Comfort_Level'].astype(str).str.lower() == str(english_comfort_level).lower()]
                if not fallback_df.empty:
                    logger.info(f"ModellingQueryDocumentNode: Fallback - returning {fallback_df.shape[0]} entries for English Comfort Level '{english_comfort_level}' only.")
                    return {"modelling_document_data": fallback_df.to_dict(orient='records')}
            logger.info("ModellingQueryDocumentNode: No documents found even with fallback.")
            return {"modelling_document_data": []}

        modelling_document_data = filtered_df.to_dict(orient='records')
        logger.info(f"ModellingQueryDocumentNode: Returning {len(modelling_document_data)} filtered documents.")
        
        return {"modelling_document_data": modelling_document_data}

    except FileNotFoundError:
        logger.error(f"ModellingQueryDocumentNode: modelling_data.csv not found at {csv_path} (exception)." )
        return {"modelling_document_data": [], "error": f"modelling_data.csv not found at {csv_path}"}
    except pd.errors.EmptyDataError:
        logger.error(f"ModellingQueryDocumentNode: modelling_data.csv is empty at {csv_path}")
        return {"modelling_document_data": [], "error": "modelling_data.csv is empty"}
    except Exception as e:
        logger.error(f"ModellingQueryDocumentNode: Error processing CSV: {e}", exc_info=True)
        return {"modelling_document_data": [], "error": f"Error processing CSV: {str(e)}"}

# Example usage (for local testing if needed)
async def main_test():
    # Mock state for testing
    class MockAgentGraphState(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    # Ensure the dummy CSV is created in the correct location (project root)
    test_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dummy_csv_path = os.path.join(test_base_dir, "modelling_data.csv")

    if not os.path.exists(dummy_csv_path):
        print(f"Creating dummy modelling_data.csv at {dummy_csv_path} for testing.")
        dummy_data = {
            'Example_Prompt_Text': ['Test Prompt 1', 'Test Prompt 2', 'Test Prompt 3', 'Test Prompt 4'], 
            'Student_Goal_Context': ['Goal 1', 'Goal 2', 'Goal 3', 'Goal 4'], 
            'Student_Confidence_Context': ['Confident', 'Not Confident', 'Confident', 'Beginner'],
            'English_Comfort_Level': ['Beginner', 'Beginner', 'Conversational', 'Beginner'], 
            'Teacher_Initial_Impression': ['Good', 'Needs work', 'Excellent', 'Okay'], 
            'Student_Struggle_Context': ['difficulty starting tasks', 'grammar', 'difficulty starting tasks', 'Coherence'],
            'pre_modeling_setup_script': ['Setup 1', 'Setup 2', 'Setup 3', 'Setup 4'], 
            'modeling_and_think_aloud_sequence_json': ['JSON 1', 'JSON 2', 'JSON 3', 'JSON 4'],
            'post_modeling_summary_and_key_takeaways': ['Summary 1', 'Summary 2', 'Summary 3', 'Summary 4'],
            'comprehension_check_or_reflection_prompt_for_student': ['Reflection 1', 'Reflection 2', 'Reflection 3', 'Reflection 4'],
            'adaptation_for_student_profile_notes': ['Notes 1', 'Notes 2', 'Notes 3', 'Notes 4']
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv(dummy_csv_path, sep='\t', index=False)
    else:
        print(f"Dummy modelling_data.csv already exists at {dummy_csv_path}")

    # Test case 1: Specific match
    state1 = MockAgentGraphState({
        "user_id": "test_user_1",
        "english_comfort_level": "Beginner",
        "student_struggle_context": "difficulty starting tasks"
    })
    result1 = await modelling_query_document_node(state1)
    print(f"Result 1 (Specific Match - Beginner, starting tasks): {len(result1.get('modelling_document_data', []))} items")
    if result1.get('modelling_document_data'):
        print(f"First item: {result1.get('modelling_document_data')[0]['Example_Prompt_Text']}")

    # Test case 2: Broader match (only comfort level)
    state2 = MockAgentGraphState({
        "user_id": "test_user_2",
        "english_comfort_level": "Conversational",
        "student_struggle_context": "" # No specific struggle
    })
    result2 = await modelling_query_document_node(state2)
    print(f"Result 2 (Comfort Level Match - Conversational): {len(result2.get('modelling_document_data', []))} items")
    if result2.get('modelling_document_data'):
        print(f"First item: {result2.get('modelling_document_data')[0]['Example_Prompt_Text']}")

    # Test case 3: No match
    state3 = MockAgentGraphState({
        "user_id": "test_user_3",
        "english_comfort_level": "Advanced", # Not in dummy data
        "student_struggle_context": "non_existent_struggle"
    })
    result3 = await modelling_query_document_node(state3)
    print(f"Result 3 (No Match - Advanced): {len(result3.get('modelling_document_data', []))} items. Error: {result3.get('error')}")

    # Test case 4: Only struggle context (should not primarily filter on this alone without comfort level)
    state4 = MockAgentGraphState({
        "user_id": "test_user_4",
        "english_comfort_level": "", # No comfort level
        "student_struggle_context": "grammar"
    })
    result4 = await modelling_query_document_node(state4)
    # This will likely return all rows if comfort level is empty, then filter by struggle, or return empty based on logic.
    # Current logic: if comfort level is empty, it won't filter by it. Then it filters by struggle context.
    print(f"Result 4 (Only Struggle - grammar): {len(result4.get('modelling_document_data', []))} items. Error: {result4.get('error')}")
    if result4.get('modelling_document_data'):
        print(f"First item: {result4.get('modelling_document_data')[0]['Example_Prompt_Text']}")

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main_test())
