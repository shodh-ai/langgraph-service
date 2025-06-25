# langgraph-service/agents/modelling_generator.py
import logging
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

def format_rag_for_prompt(rag_data: list) -> str:
    """Formats the RAG document data into a string for the LLM prompt."""
    if not rag_data:
        return "No expert examples were found in the knowledge base."
    
    formatted_examples = []
    for i, item in enumerate(rag_data):
        # Assuming each item is a dict with a 'content' key from the knowledge base
        content = item.get('content', 'No content available.')
        formatted_examples.append(f"--- Expert Example {i+1} ---\n{content}\n")
    
    return "\n".join(formatted_examples)

async def modelling_generator_node(state: AgentGraphState) -> dict:
    """
    Generates the full modelling script and associated content by using context
    from the state to create a detailed prompt for the LLM.
    """
    logger.info("---Executing Modelling Generator Node---")

    # --- Defensive checks for required inputs ---
    prompt_to_model = state.get("example_prompt_text")
    if not prompt_to_model:
        logger.warning("ModellingGeneratorNode: 'example_prompt_text' not found in state.")
        return {
            "intermediate_modelling_payload": {
                "error": "Missing prompt",
                "error_message": "I seem to have misplaced the prompt we were going to work on. Could you tell me again?"
            }
        }

    rag_document_data = state.get("rag_document_data")
    if not rag_document_data:
        logger.warning("ModellingGeneratorNode: Received no RAG documents. Cannot generate model.")
        return {
            "intermediate_modelling_payload": {
                "error": "Missing RAG data",
                "error_message": "I'm sorry, I couldn't find any reference materials for this task. Let's try a different approach."
            }
        }
    # --- End defensive checks ---

    try:
        # Now we can safely get the rest of the context
        student_struggle = state.get("student_struggle_context", "general difficulties")
        english_comfort = state.get("english_comfort_level", "not specified")
        student_goal = state.get("student_goal_context", "not specified")
        student_confidence = state.get("student_confidence_context", "not specified")

        expert_examples = format_rag_for_prompt(rag_document_data)

        # --- Prompt Engineering ---
        # This prompt is now rich with context from the state.
        llm_prompt = f"""
_You are 'The Structuralist', an expert AI TOEFL Tutor. Your task is to MODEL how to approach a TOEFL task._

**Student Context:**
- Task Prompt for Student: "{prompt_to_model}"
- Student's Primary Struggle: "{student_struggle}"
- Student's Stated Goal: "{student_goal}"
- Student's Confidence Level: "{student_confidence}"
- Student's English Comfort Level: "{english_comfort}"

**Expert Examples from Knowledge Base:**
{expert_examples}

**Your Task:**
_Generate a response that breaks down the process into logical steps. 
Include a title for the model, a summary of the key takeaway, and the steps themselves._

_Return a SINGLE JSON object with the following keys:_
- _"model_title": (String) A clear, concise title for the example (e.g., 'How to Structure a Paragraph')._
- _"model_steps": (Array of Strings) A list where each string is a distinct step in the process (e.g., ["Step 1: Write a topic sentence.", "Step 2: Provide supporting evidence.", "Step 3: Add a concluding sentence."])_
- _"model_summary": (String) A brief summary of the main lesson from the model._
"""
        logger.debug(f"Modelling Generator Prompt:\n{llm_prompt}")
        
        # --- LLM Call ---
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(llm_prompt)
        response_json = json.loads(response.text)
        logger.info(f"LLM Response for modelling: {response_json}")

        # This node's output is an intermediate payload for the formatter.
        return {"intermediate_modelling_payload": response_json}
        
    except Exception as e:
        logger.error(f"ModellingGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {
            "intermediate_modelling_payload": {
                "error": "Generator failure",
                "error_message": f"I encountered an unexpected issue while preparing your modeling exercise. The error was: {e}"
            }
        }
