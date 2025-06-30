# agents/feedback_generator.py
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
        return "No expert examples were retrieved. Rely on general pedagogical principles for feedback."
    try:
        # Provide a few diverse examples of feedback strategies
        examples_str = "\n---\n".join([json.dumps(item, indent=2) for item in rag_data[:3]])
        return f"Consult these expert examples of giving feedback:\n{examples_str}"
    except Exception as e:
        logger.warning(f"Could not format RAG example for prompt: {e}")
        return "No valid expert examples were retrieved."

async def feedback_generator_node(state: AgentGraphState) -> dict:
    """
    Analyzes student work and generates a rich, structured feedback plan.
    """
    logger.info("---Executing Feedback Generator Node---")
    
    try:
        student_writing = state.get("transcript")
        if not student_writing:
            raise ValueError("'transcript' (the student's work) is missing from the state.")

        rag_data = state.get("rag_document_data", [])
        expert_examples_str = format_rag_for_prompt(rag_data)
        
        # This is a very detailed prompt to get the structured data we need.
        llm_prompt = f"""
You are 'The Structuralist', an expert AI TOEFL Tutor providing detailed feedback on a student's work.

**Student's Written Work:**
"{student_writing}"

**Expert Examples of Feedback for Similar Situations:**
{expert_examples_str}

**Your Task:**
Analyze the student's writing and generate a structured feedback plan as a single JSON object. The plan should identify 1-3 key areas for improvement. For each area, you will provide a spoken explanation, text highlights, and detailed remarks.

The JSON object MUST have two keys:

1.  `"spoken_script"`: (Array of Strings) A list of sentences the AI will say. This script should introduce the feedback, explain each point, and provide a concluding summary.
2.  `"feedback_items"`: (Array of Objects) A list of specific feedback points. Each object in this array MUST have:
    - `"remark"`: (Object) An object containing the detailed feedback card.
        - `"id"`: (String) A unique ID for this remark (e.g., "R1", "R2").
        - `"title"`: (String) A short title for the feedback point (e.g., "Subject-Verb Agreement").
        - `"details"`: (String) A detailed explanation of the error or issue.
        - `"suggestion"`: (String) A concrete suggestion for improvement.
    - `"highlight"`: (Object) An object describing the text to highlight in the student's work that corresponds to this remark.
        - `"start"`: (Integer) The starting character position of the highlight in the student's text.
        - `"end"`: (Integer) The ending character position of the highlight.
        - `"style_class"`: (String) The CSS class for the highlight (e.g., "error_grammar", "suggestion_style").

**Example JSON Output:**
{{
  "spoken_script": [
    "Overall, a good effort on this paragraph. I've noted two areas we can focus on to make it even stronger.",
    "First, let's look at the subject-verb agreement in your opening sentence.",
    "Next, we can improve the clarity of your main example.",
    "Working on these points will greatly enhance the impact of your writing."
  ],
  "feedback_items": [
    {{
      "remark": {{
        "id": "R1",
        "title": "Subject-Verb Agreement",
        "details": "The subject 'The dogs' is plural, but the verb 'run' is in its singular form 'runs'. Plural subjects must have plural verbs.",
        "suggestion": "Change 'runs' to 'run' to match the plural subject 'dogs'."
      }},
      "highlight": {{
        "start": 4,
        "end": 14,
        "style_class": "error_grammar"
      }}
    }},
    {{
      "remark": {{
        "id": "R2",
        "title": "Clarity & Specificity",
        "details": "The phrase 'it was very good' is a bit vague. The reader doesn't know what specific qualities made it good.",
        "suggestion": "Try replacing 'very good' with more descriptive words, like 'exceptionally well-researched and insightful'."
      }},
      "highlight": {{
        "start": 85,
        "end": 99,
        "style_class": "suggestion_style"
      }}
    }}
  ]
}}

Generate the JSON object for the provided student's work now.
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
        
        logger.info("Feedback generator successfully created the structured feedback plan.")
        
        return {"intermediate_feedback_payload": response_json}

    except Exception as e:
        logger.error(f"FeedbackGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"intermediate_feedback_payload": {"error": True, "error_message": str(e)}}