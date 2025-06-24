# new graph/scaffolding_generator_node.py
import logging
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def scaffolding_generator_node(state: AgentGraphState) -> dict:
    """
    Generates the scaffolding content directly, using RAG results as expert examples.
    This combines the planning and generation steps into one efficient LLM call.
    """
    logger.info("---Executing Scaffolding Generator Node (Refactored)---")
    
    try:
        # 1. Get RAG data and student context from state
        rag_data = state.get("rag_document_data", [])
        expert_examples = format_rag_for_prompt(rag_data) # A helper to format the list of dicts

        # 2. PROMPT ENGINEERING: Create a powerful, one-shot prompt
        llm_prompt = f"""
You are 'The Encouraging Nurturer', an expert AI Tutor. Your task is to provide scaffolding for a student.

**Student Context:**
- Primary Struggle: {state.get("primary_struggle", "Not specified")}
- Learning Objective: {state.get("learning_objective_id", "Not specified")}
- Student's Written Input (if any): {state.get("Student_Written_Input_Chunk", "N/A")}

**Expert Examples of Scaffolding for Similar Situations:**
{expert_examples}

**Your Task:**
Based on the student's struggle and the expert examples, generate a complete scaffolding interaction. Return a SINGLE JSON object with two keys:

1.  `"text_for_tts"`: (String) A friendly, spoken script that introduces the scaffold and explains how to use it.
2.  `"ui_actions"`: (JSON Array) A list of UI actions to display on the frontend. The most common action here will be 'DISPLAY_SCAFFOLD_TEMPLATE' or 'SHOW_HINT'.

Example for the "ui_actions" array:
[
    {{
        "action_type": "DISPLAY_SCAFFOLD_TEMPLATE",
        "parameters": {{
            "template_name": "OREO Speaking Template",
            "fields": [
                {{"label": "Opinion:", "placeholder": "I believe that..."}},
                {{"label": "Reason:", "placeholder": "One reason is..."}}
            ]
        }}
    }}
]
"""
        # 3. LLM Call (same as your other generator nodes)
        # ... (your genai.generate_content_async logic) ...
        # response_json = json.loads(response.text)
        
        # 4. Return using the STANDARDIZED intermediate payload key
        # return {"intermediate_scaffolding_payload": response_json}
        return {} # Placeholder

    except Exception as e:
        logger.error(f"ScaffoldingGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {
            "error_message": f"Failed to generate scaffolding: {e}",
            "route_to_error_handler": True
        }

def format_rag_for_prompt(rag_data: list) -> str:
    # Helper to convert the list of metadata dicts into a clean string for the prompt
    return json.dumps(rag_data, indent=2)