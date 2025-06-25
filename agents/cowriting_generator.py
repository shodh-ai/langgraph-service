# langgraph-service/agents/cowriting_generator.py
import logging
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def cowriting_generator_node(state: AgentGraphState) -> dict:
    """
    Acts as a creative co-writer, providing students with suggestions to continue their writing.
    """
    logger.info("---Executing Co-Writing Generator Node---")

    # --- Defensive checks for required inputs ---
    student_writing = state.get("Student_Written_Input_Chunk")
    if not student_writing:
        logger.warning("CoWritingNode: 'Student_Written_Input_Chunk' not found in state.")
        return {
            "intermediate_cowriting_payload": {
                "error": "Missing student input",
                "error_message": "It looks like there's nothing to work with yet. Try writing a sentence or two first!"
            }
        }
    # --- End defensive checks ---

    try:
        # PROMPT ENGINEERING: A prompt for a co-writing partner
        llm_prompt = f"""
You are 'The Creative Collaborator', an AI partner helping a student write.
The student has written this so far:
---
{student_writing}
---

Your task is to provide two distinct suggestions to help the student continue.
For each suggestion, provide the suggested text and a brief, encouraging explanation of why it's a good next step.

Return a SINGLE JSON object with the following keys:
- "suggestion_1_text": (String) The first concrete sentence or two the student could add.
- "suggestion_1_explanation": (String) A brief rationale for the first suggestion (e.g., "This adds more detail to your main point.").
- "suggestion_2_text": (String) A different, second suggestion for what to write next.
- "suggestion_2_explanation": (String) The rationale for the second suggestion (e.g., "This could be a good transition to your next idea.").
"""
        
        # LLM Call (Placeholder)
        # response_json = ...
        
        # Placeholder JSON for development
        response_json = {
            "suggestion_1_text": "To further strengthen your argument, you could add a specific example here.",
            "suggestion_1_explanation": "Adding a concrete example will make your point more convincing and easier for the reader to understand.",
            "suggestion_2_text": "Another option is to introduce a counter-argument and then refute it.",
            "suggestion_2_explanation": "This shows that you have considered different perspectives and strengthens your own position."
        }

        # Return using a STANDARDIZED intermediate payload key
        return {"intermediate_cowriting_payload": response_json}

    except Exception as e:
        logger.error(f"CoWritingGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {
            "intermediate_cowriting_payload": {
                "error": "Generator failure",
                "error_message": f"I encountered an unexpected issue while preparing co-writing suggestions. The error was: {e}"
            }
        }
