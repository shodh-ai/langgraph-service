# new graph/teaching_generator_node.py

import logging
import json
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

def format_rag_for_prompt(rag_data_list: list) -> str:
    """Helper to format the list of dicts from RAG into a clean string for the prompt."""
    if not rag_data_list:
        return "No specific expert examples were retrieved. Rely on general pedagogical principles."
    # Use json.dumps for a clean, readable representation of the examples
    return f"Expert Examples for Inspiration:\n{json.dumps(rag_data_list, indent=2)}"

async def teaching_generator_node(state: AgentGraphState) -> dict:
    """
    Generates the raw, structured content for a teaching segment using an LLM,
    guided by student context and RAG-retrieved examples.
    """
    logger.info("---Executing Teaching Generator Node (Refactored)---")

    try:
        # --- 1. Get Context and RAG data ---
        rag_data = state.get("rag_document_data", [])
        expert_examples = format_rag_for_prompt(rag_data)

        # --- 2. PROMPT ENGINEERING: Ask for the raw components, not the final speech ---
        llm_prompt = f"""
You are an expert AI TOEFL Tutor with the persona of 'The Structuralist'. Your task is to generate the core components for a teaching lesson segment.

**Student & Lesson Context:**
- Learning Objective: {state.get('learning_objective', 'Not specified')}
- Student's Stated Goal: {state.get('student_goal_context', 'Not specified')}
- Student's Primary Struggle: {state.get('student_struggle_context', 'Not specified')}
- Student's Current Emotional State: {state.get('current_affective_state', 'Neutral')}

**Expert Examples from Knowledge Base (for inspiration on structure and tone):**
{expert_examples}

**Your Task:**
Generate the fundamental building blocks for this teaching moment. DO NOT write the final script. Instead, provide the raw content for each part. Return a SINGLE JSON object with the following keys:

1.  `"opening_hook"`: (String) A short, engaging sentence to start this lesson segment.
2.  `"core_explanation"`: (String) A clear, concise explanation of the main concept, tailored to the student's context.
3.  `"key_examples"`: (String) One or two clear examples that illustrate the core explanation.
4.  `"comprehension_check"`: (String) A single, open-ended question to check for understanding.
5.  `"visual_aid_instructions"`: (JSON Object or null) If a visual aid would be highly effective, describe it as a JSON object (like in your previous prompt). Otherwise, return null.
"""

        # --- 3. LLM Call ---
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash", # Use a fast, capable model
            generation_config=GenerationConfig(response_mime_type="application/json")
        )
        
        response = await model.generate_content_async(llm_prompt)
        response_json = json.loads(response.text)
        
        logger.info("Teaching Generator Node: Successfully generated raw teaching components.")
        
        # --- 4. Return using the STANDARDIZED intermediate payload key ---
        return {"intermediate_teaching_payload": response_json}

    except Exception as e:
        logger.error(f"TeachingGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {
            "error_message": f"Failed to generate teaching content: {e}",
            "route_to_error_handler": True
        }