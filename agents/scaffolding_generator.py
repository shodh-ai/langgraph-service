# agents/scaffolding_generator.py
import logging
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

def format_rag_for_prompt(rag_data: list) -> str:
    """Helper to format RAG results for the prompt."""
    if not rag_data:
        return "No specific expert examples were retrieved. Rely on general pedagogical principles for scaffolding."
    # Use the top 1-2 examples to guide the LLM
    try:
        examples_str = "\n---\n".join([json.dumps(item, indent=2) for item in rag_data[:2]])
        return f"Follow the patterns in these expert examples:\n{examples_str}"
    except Exception as e:
        logger.warning(f"Could not format RAG example for prompt: {e}")
        return "No valid expert examples were retrieved."

async def scaffolding_generator_node(state: AgentGraphState) -> dict:
    """
    Designs a scaffolding activity, including instructional text and editor content.
    """
    logger.info("---Executing Scaffolding Generator Node---")
    
    try:
        # Get context from the state
        rag_data = state.get("rag_document_data", [])
        expert_examples_str = format_rag_for_prompt(rag_data)
        
        learning_task = state.get("Learning_Objective_Task", "the current writing task")
        struggle_point = state.get("Specific_Struggle_Point", "general difficulty")

        if not learning_task or not struggle_point:
             raise ValueError("Missing 'Learning_Objective_Task' or 'Specific_Struggle_Point' in state.")

        # This prompt asks the LLM to act as an instructional designer.
        llm_prompt = f"""
You are 'The Structuralist', an expert AI TOEFL Tutor who provides clear, structured support. Your task is to design a scaffolding activity for a student.

**Student Context:**
- Learning Task: "{learning_task}"
- Specific Struggle Point: "{struggle_point}"

**Expert Examples of Scaffolding for Similar Situations:**
{expert_examples_str}

**Your Task:**
Generate a JSON object that defines the complete scaffolding experience. The object must have the following keys:

1.  `"prompt_display_text"`: (String) The full essay prompt or task instruction that should be displayed to the student.
2.  `"initial_editor_content"`: (String) The initial HTML content to set in the student's writing editor. This could be a template, a pre-written topic sentence, or an outline for them to fill in. Use `<p>` tags for paragraphs and placeholders like `[Your turn to write here]`.
3.  `"ai_guidance_script"`: (Array of Strings) A list of short, spoken instructions or feedback. The first string will be the main instruction. Subsequent strings can be used for follow-up feedback.

**Example JSON Output:**
{{
  "prompt_display_text": "Some people prefer to work for a large company. Others prefer to work for a small company. Which would you prefer? Use specific reasons and details to support your choice.",
  "initial_editor_content": "<p>When considering my future career, I would prefer to work for a [large/small] company for several reasons.</p><p>Firstly, [Your first reason here].</p><p>Secondly, [Your second reason here].</p>",
  "ai_guidance_script": [
    "Okay, let's work on structuring your essay. I've set up a template for you in the editor with topic sentences. Your task is to fill in the supporting details for each reason.",
    "That's a good start on your first reason! Now, can you add a specific example to make it stronger?",
    "Excellent, you've completed the first paragraph. Let's move on to the second reason."
  ]
}}

Generate the JSON object for the student's context now.
"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("GOOGLE_API_KEY is not set.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(llm_prompt)
        response_json = json.loads(response.text)
        
        logger.info("Scaffolding generator successfully created the activity plan.")
        
        return {"intermediate_scaffolding_payload": response_json}

    except Exception as e:
        logger.error(f"ScaffoldingGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"intermediate_scaffolding_payload": {"error": True, "error_message": str(e)}}