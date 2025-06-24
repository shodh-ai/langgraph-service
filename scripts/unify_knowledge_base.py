# scripts/unify_knowledge_base.py

import pandas as pd
import os
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_embedding_document(row: pd.Series, category: str) -> str:
    """
    Intelligently creates a single text document for embedding based on the category.
    This function is the heart of the standardization process, tailored to each
    unique CSV schema.
    """
    text_parts = []
    
    # --- Logic for each specific file type ---

    if category == "cowriting":
        # For co-writing, the most important context is the student's writing and the AI's direct feedback on it.
        text_parts.append(f"Learning Objective: {row.get('Learning_Objective_Focus', '')}")
        text_parts.append(f"Student's Original Text: {row.get('Student_Written_Input_Chunk', '')}")
        text_parts.append(f"Assessed Problem: {row.get('Immediate_Assessment_of_Input', '')}")
        text_parts.append(f"AI's Suggested Revision: {row.get('Suggested_AI_Revision_if_Any', '')}")
        text_parts.append(f"AI's Rationale: {row.get('Rationale_for_Intervention_Style', '')}")
        
    elif category == "modelling":
        # For modelling, the key is the task prompt and the expert's thinking process.
        text_parts.append(f"Task Prompt being modeled: {row.get('Example_Prompt_Text', '')}")
        text_parts.append(f"Intended for student with this struggle: {row.get('Student_Struggle_Context', '')}")
        text_parts.append(f"Modeling setup script: {row.get('pre_modeling_setup_script', '')}")
        text_parts.append(f"Key takeaways from the model: {row.get('post_modeling_summary_and_key_takeaways', '')}")

    elif category == "pedagogy":
        # For pedagogy, the student's self-reported answers and the resulting plan are key.
        text_parts.append(f"Student's self-reported answers: {row.get('Answer One', '')} {row.get('Answer Two', '')} {row.get('Answer Three', '')}")
        text_parts.append(f"Teacher's initial impression: {row.get('Initial Impression', '')}")
        # The 'Pedagogy' column contains a JSON string with the expert reasoning. We want that text.
        try:
            pedagogy_plan = json.loads(row.get('Pedagogy', '{}'))
            text_parts.append(f"Expert reasoning for the plan: {pedagogy_plan.get('reasoning', '')}")
        except (json.JSONDecodeError, TypeError):
            text_parts.append(f"Pedagogical Plan: {row.get('Pedagogy', '')}") # Fallback to raw string

    elif category == "scaffolding":
        # For scaffolding, the specific struggle and the type of help are most important.
        text_parts.append(f"Learning Task: {row.get('Learning_Objective_Task', '')}")
        text_parts.append(f"Specific Student Struggle: {row.get('Specific_Struggle_Point', '')}")
        text_parts.append(f"Reason for choosing this scaffold: {row.get('reasoning_for_scaffold_choice', '')}")
        text_parts.append(f"Script to deliver the scaffold: {row.get('scaffold_delivery_script', '')}")
        
    elif category == "feedback":
        # For feedback, the error and the strategy to correct it are central.
        text_parts.append(f"Student Proficiency and Task: {row.get('Proficiency', '')} {row.get('Task', '')}")
        text_parts.append(f"Student Behavior during task: {row.get('Behavior Factor', '')}")
        text_parts.append(f"Specific Error Made: {row.get('Error', '')}")
        text_parts.append(f"Expert Diagnosis of the Error: {row.get('Diagnose', '')}")
        text_parts.append(f"Strategy to Explain the Correction: {row.get('Explain Strategy', '')}")
        
    elif category == "teaching":
        # For teaching, the learning objective and the core explanation are key.
        text_parts.append(f"Learning Objective: {row.get('LEARNING_OBJECTIVE', '')}")
        text_parts.append(f"Core teaching strategy: {row.get('CORE_EXPLANATION_STRATEGY', '')}")
        text_parts.append(f"Key examples used: {row.get('KEY_EXAMPLES', '')}")
        text_parts.append(f"Common misconceptions to address: {row.get('COMMON_MISCONCEPTIONS', '')}")
        
    else:
        logging.warning(f"No specific embedding document creation logic for category: '{category}'. Using fallback.")
        # Create a generic document from the first few columns as a fallback
        text_parts = [str(v) for v in row.iloc[:5].values]

    # Join all the parts into a single, clean string for the embedding model
    return " \n ".join(filter(None, text_parts)).strip()


def unify_and_standardize_cta_data():
    """
    Reads multiple source CSVs with different schemas, standardizes them into a
    unified format, and saves the result as a single JSONL file ready for ingestion.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = os.path.join(project_root, 'data', 'source_cta')
    output_dir = os.path.join(project_root, 'data', 'unified')
    output_path = os.path.join(output_dir, 'unified_knowledge_base.jsonl')

    os.makedirs(output_dir, exist_ok=True)

    # This dictionary maps the filename to the desired category name.
    # It now includes all your files.
    source_files = {
        "cowriting_data.csv": "cowriting",
        "modelling_data.csv": "modelling",
        "pedagogical_data.csv": "pedagogy",
        "scaffolding_data.csv": "scaffolding",
        "teaching_data.csv": "teaching",
        "feedback_data.csv": "feedback",
    }

    final_standardized_records = []
    
    logging.info("Starting unification and standardization process...")

    for filename, category in source_files.items():
        file_path = os.path.join(source_dir, filename)
        
        if not os.path.exists(file_path):
            logging.warning(f"File not found, skipping: {file_path}")
            continue

        try:
            logging.info(f"Processing '{filename}' for category '{category}'...")
            df = pd.read_csv(file_path)

            # Iterate over each row in the dataframe
            for _, row in df.iterrows():
                # Create the clean text document for this row using our new function
                embedding_doc = create_embedding_document(row, category)
                
                # Convert the original row to a dictionary to store as metadata
                # This ensures all original data is preserved
                original_row_dict = row.to_dict()
                # Convert all metadata values to string for safety with ChromaDB
                metadata = {k: str(v) for k, v in original_row_dict.items()}
                metadata['source_file'] = filename # Add source file for traceability

                # Create the final, standardized record
                standardized_record = {
                    "category": category,
                    "document_for_embedding": embedding_doc,
                    "metadata": metadata
                }
                final_standardized_records.append(standardized_record)

            logging.info(f"Successfully processed and standardized {len(df)} rows from '{filename}'.")

        except Exception as e:
            logging.error(f"Failed to process file {filename}. Error: {e}", exc_info=True)
            continue
            
    if not final_standardized_records:
        logging.error("No records were processed. Unification failed.")
        return

    # Save the list of dictionaries to a JSON Lines file
    logging.info(f"Unification complete. Total standardized records: {len(final_standardized_records)}")
    try:
        logging.info(f"Saving unified knowledge base to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in final_standardized_records:
                f.write(json.dumps(record) + '\n')
        logging.info("Successfully saved the unified JSONL file.")
    except Exception as e:
        logging.error(f"Failed to save the unified JSONL file. Error: {e}", exc_info=True)


if __name__ == "__main__":
    unify_and_standardize_cta_data()