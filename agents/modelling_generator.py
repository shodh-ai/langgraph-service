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
    Generates a layered modeling payload including a main explanation, simplified version,
    pre-written clarifications, and an interactive TTS/listen sequence.
    """
    logger.info("---Executing Modelling Generator Node (Layered Content Refactor)---")

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
        student_struggle = state.get("student_struggle_context", "general difficulties")
        expert_examples = format_rag_for_prompt(rag_document_data)

        json_output_example = """
{
  "main_explanation": "The core task of this prompt is to analyze the provided text and identify the author's main argument and the evidence they use to support it. We'll break this down into finding the thesis, locating supporting details, and summarizing the overall point.",
  "simplified_explanation": "Imagine you're a detective. The prompt gives you a story, and your job is to find the big idea the author wants you to believe and the clues they left to convince you.",
  "clarifications": {
    "what_is_a_thesis": "The thesis is the main point or central argument of a piece of writing. It's the one sentence that sums up what the author is trying to prove.",
    "what_counts_as_evidence": "Evidence can be facts, statistics, quotes from experts, or specific examples from the text that back up the main argument. It's the 'how we know' part of the argument."
  },
  "sequence": [
    {"type": "tts", "content": "Great, let's break down how to approach this prompt together."},
    {"type": "tts", "content": "First, let's read the prompt carefully..."},
    {"type": "listen", "expected_intent": "CONFIRMATION", "timeout_ms": 3000},
    {"type": "tts", "content": "The very first step is to identify the key task. Here, it's asking us to analyze an argument."},
    {"type": "listen", "expected_intent": "UNDERSTOOD_STEP", "prompt_if_silent": "Does that first step make sense?", "timeout_ms": 4000}
  ]
}
"""

        llm_prompt = f"""
_You are 'The Structuralist', an expert AI TOEFL Tutor. Your task is to MODEL how to approach a TOEFL task by generating a 'Layered Content' payload._

**Student & Task Context:**
- Task Prompt to Model: "{prompt_to_model}"
- Student's Primary Struggle: "{student_struggle}"

**Expert Examples from Knowledge Base (for inspiration on structure and tone):**
{expert_examples}

**Your Task:**
Generate a single JSON object containing four keys:
1.  `main_explanation`: A clear, concise explanation of how to approach the specific `Task Prompt to Model`.
2.  `simplified_explanation`: A very simple analogy for the approach.
3.  `clarifications`: A JSON object with pre-written answers to 2-3 common questions a student might have about this type of task.
4.  `sequence`: A JSON array for an interactive, step-by-step walkthrough of the process. Use 'tts' and 'listen' types.

Return a SINGLE JSON object with the exact structure shown in this example:
{json_output_example}
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
        logger.info(f"LLM Response for layered modelling content: {response_json}")

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
