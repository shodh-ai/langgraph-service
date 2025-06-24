# langgraph-service/agents/pedagogy_generator_node.py
import logging
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def pedagogy_generator_node(state: AgentGraphState) -> dict:
    """
    Analyzes the student's state and RAG results to generate a personalized
    pedagogical next step, such as a new task or a concept explanation.
    """
    logger.info("---PEDAGOGY GENERATOR---")

    try:
        rag_results = state.get("pedagogy_rag_results", [])
        rag_context = "\n".join([doc.get("content", "") for doc in rag_results])

        llm_prompt = f"""
        You are an expert AI Tutor. Your role is to determine the best next step for a student.

        **Student's Current Situation:**
        - **Learning Objective:** {state.get('Learning_Objective_Focus', 'Not specified')}
        - **Recent Interaction Transcript:** {state.get('transcript', 'No recent interaction.')}
        - **Student Model Summary:** {state.get('student_model', {}).get('summary', 'No summary available.')}

        **Relevant Pedagogical Strategies from Knowledge Base:**
        {rag_context if rag_context else 'No specific strategies were retrieved.'}

        **Your Task:**
        Based on all the information above, decide on the single best next step for this student. Formulate a response that includes a friendly, encouraging message and the details for the next task.

        **Output Format:**
        Return a SINGLE JSON object with the following keys:
        - "suggestion_text": (String) A friendly, encouraging message to the student explaining the next step.
        - "next_task_details": (Object) An object containing the details for the next task, with the following sub-keys:
            - "title": (String) A clear, concise title for the task (e.g., "Practice: Identifying the Main Idea").
            - "type": (String) The type of task (e.g., 'practice', 'reading', 'writing', 'concept_review').
            - "page_target": (String) The specific UI page or component to load (e.g., 'reading_exercise', 'writing_prompt').
            - "prompt_id": (String) A unique identifier for the specific content or prompt to be used.
        """

        # Configure the generative model
        # model = genai.GenerativeModel('gemini-pro')
        # response = await model.generate_content_async(
        #     llm_prompt,
        #     generation_config=GenerationConfig(response_mime_type="application/json")
        # )
        # generated_content = json.loads(response.text)

        # Placeholder for generated content
        generated_content = {
            "suggestion_text": "Great work on the last task! To build on that, let's try an exercise focused on identifying the main idea in a short paragraph.",
            "next_task_details": {
                "title": "Practice: Main Idea",
                "type": "practice",
                "page_target": "reading_exercise_main_idea",
                "prompt_id": "reading_main_idea_003"
            }
        }

        logger.info("Successfully generated pedagogical content.")
        return {"pedagogy_content": generated_content}

    except Exception as e:
        logger.error(f"PedagogyGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        # Fallback to a safe, generic response
        fallback_content = {
            "suggestion_text": "Let's try something new. Please select a task from the menu.",
            "next_task_details": {}
        }
        return {"pedagogy_content": fallback_content}
