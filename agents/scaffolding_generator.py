import json
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)


async def scaffolding_generator_node(state: AgentGraphState) -> dict:
    """
    Generates the final scaffolding content for the student.
    
    This node creates personalized scaffolding content based on the selected strategy,
    adapting it to the student's specific needs and learning context.
    """
    logger.info(
        f"ScaffoldingGeneratorNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    
    user_data = state.get("user_data", {})
    primary_struggle = state.get("primary_struggle", "")
    learning_objective_id = state.get("learning_objective_id", "")
    selected_scaffold_type = state.get("selected_scaffold_type", "")
    scaffold_adaptation_plan = state.get("scaffold_adaptation_plan", "")
    scaffold_content_type = state.get("scaffold_content_type", "")
    scaffold_content_name = state.get("scaffold_content_name", "")
    scaffold_content = state.get("scaffold_content", {})
    
    logger.info(f"Generator received scaffold type: {selected_scaffold_type}")
    logger.info(f"Generator received content type: {scaffold_content_type}")
    logger.info(f"Generator received content name: {scaffold_content_name}")
    logger.info(f"State keys available: {list(state.keys())}")
    
    new_state = {key: value for key, value in state.items()}
    
    logger.info(f"Starting with state keys in generator: {list(new_state.keys())}")
    
    if not selected_scaffold_type:
        logger.warning("No valid scaffolding strategy selected - using generic fallback")
        
        new_state["selected_scaffold_type"] = "Basic Template"
        new_state["scaffold_content_type"] = "template"
        new_state["scaffold_content_name"] = "Basic TOEFL Speaking Template"
        
        fallback_tts = "Here's a basic template to help structure your TOEFL speaking response."
        
        template_content = {
            "fields": [
                {"label": "Introduction", "placeholder": "I believe that..."},
                {"label": "First Reason", "placeholder": "One reason is..."},
                {"label": "Example", "placeholder": "For example..."},
                {"label": "Second Reason", "placeholder": "Another reason is..."},
                {"label": "Conclusion", "placeholder": "Therefore, I think..."},
            ]
        }
        
        scaffolding_ui = [{
            "type": "scaffold",
            "scaffold_type": "template",
            "content": template_content
        }]
        
        output_ui = [{
            "type": "display_scaffold",
            "scaffold_type": "template",
            "scaffold_name": "Basic TOEFL Speaking Template",
            "content": json.dumps(template_content)
        }]
        
        new_state["scaffolding_output"] = {
            "text_for_tts": fallback_tts,
            "ui_components": scaffolding_ui
        }
        
        new_state["output_content"] = {
            "text_for_tts": fallback_tts,
            "ui_actions": output_ui
        }
        
        new_state["task_suggestion_llm_output"] = {
            "task_suggestion_tts": fallback_tts
        }
        new_state["tts"] = "Let me help you organize your response better. I've created a template you can use to structure your thoughts."
        return new_state
    
    prompt = f"""
    You are 'The Encouraging Nurturer' AI Tutor.
    
    Student Profile: {user_data}
    Primary Struggle: {primary_struggle}
    Learning Objective: {learning_objective_id}
    
    You need to generate scaffolding for this student based on this plan:
    Selected Scaffold Type: {selected_scaffold_type}
    Adaptation Plan: {scaffold_adaptation_plan}
    Scaffold Content Type: {scaffold_content_type}
    Scaffold Content Name: {scaffold_content_name}
    Scaffold Content Structure: {json.dumps(scaffold_content, indent=2)}
    
    Create a complete, personalized scaffolding response that:
    1. Introduces the scaffolding in an encouraging way
    2. Explains how to use it effectively
    3. Includes the actual scaffolding content (template, questions, steps, etc.)
    4. Provides guidance on applying it to their current task
    5. Briefly mentions how this will help with their specific struggle
    
    Return JSON:
    {{
        "text_for_tts": "The spoken introduction to the scaffolding (friendly, encouraging, brief)",
        "ui_components": [
            {{
                "type": "text",
                "content": "Any explanatory text to display"
            }},
            {{
                "type": "scaffold",
                "scaffold_type": "{scaffold_content_type}",
                "content": // The actual scaffold content structure
            }},
            {{
                "type": "guidance",
                "content": "Instructions on using the scaffold"
            }}
        ]
    }}
    """

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set")
        return {
            "scaffolding_output": {
                "text_for_tts": "I'm sorry, but I'm having technical difficulties right now.",
                "ui_components": []
            }
        }

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = model.generate_content(prompt)
        response_json = json.loads(response.text)
        
        text_for_tts = response_json.get("text_for_tts", "")
        ui_components = response_json.get("ui_components", [])
        
        logger.info(f"Generated scaffolding with {len(ui_components)} UI components")
        
        new_state["scaffolding_output"] = {
            "text_for_tts": text_for_tts,
            "ui_components": ui_components
        }
        
        new_state["output_content"] = {
            "text_for_tts": text_for_tts,
            "ui_actions": ui_components
        }
        
        new_state["task_suggestion_llm_output"] = {
            "task_suggestion_tts": text_for_tts
        }
        
        logger.info(f"Generator returning state keys: {list(new_state.keys())}")
        
        return new_state
    except Exception as e:
        logger.error(f"Error generating scaffolding: {e}")
        
        error_message = f"I'm sorry, but I encountered an error: {str(e)}"
        
        new_state["scaffolding_output"] = {
            "text_for_tts": error_message,
            "ui_components": []
        }
        
        new_state["output_content"] = {
            "text_for_tts": error_message,
            "ui_actions": []
        }
        
        new_state["task_suggestion_llm_output"] = {
            "task_suggestion_tts": error_message
        }
        
        logger.info(f"Generator returning state keys (error case): {list(new_state.keys())}")
        
        return new_state
