import json
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)


async def scaffolding_planner_node(state: AgentGraphState) -> dict:
    """
    Plans the scaffolding strategy based on retrieved examples and student needs.
    
    This node evaluates retrieved scaffolding strategies and selects the most
    appropriate approach, considering the student's specific struggles and learning goals.
    """
    logger.info(
        f"ScaffoldingPlannerNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    
    primary_struggle = state.get("primary_struggle", "")
    learning_objective_id = state.get("learning_objective_id", "")
    user_data = state.get("user_data", {})
    transcript = state.get("transcript", "")
    logger.info(f"State type: {type(state)}")
    logger.info(f"State keys available: {list(state.keys())}")
    logger.info(f"State repr: {repr(state)}")
    
    primary_struggle = state.get('primary_struggle', '')
    learning_objective_id = state.get('learning_objective_id', '')
    
    logger.info(f"Primary struggle found: {primary_struggle}")
    logger.info(f"Learning objective ID: {learning_objective_id}")
    
    scaffolding_strategies = state.get("scaffolding_strategies", [])
    logger.info(f"Scaffolding strategies type: {type(scaffolding_strategies)}")
    logger.info(f"Scaffolding strategies repr: {repr(scaffolding_strategies)}")
    
    logger.info(f"Primary struggle found: {primary_struggle}")
    logger.info(f"Learning objective ID: {learning_objective_id}")
    logger.info(f"Scaffolding strategies found: {len(scaffolding_strategies)}")
    
    if not scaffolding_strategies:
        logger.warning("No scaffolding strategies found in state - creating fallback strategy")
        
        new_state = {key: value for key, value in state.items()}
            
        if "organizing" in transcript.lower() or "rambling" in transcript.lower() or "off-topic" in transcript.lower():
            logger.info("Planner returning fallback due to error")
            
            new_state["selected_scaffold_type"] = "structure-focused"
            new_state["scaffold_adaptation_plan"] = "Provide a basic template for organizing thoughts"
            new_state["scaffold_content_type"] = "template"
            new_state["scaffold_content_name"] = "Basic TOEFL Speaking Template"
            new_state["scaffold_content"] = {
                "fields": [
                    {"label": "Introduction:", "placeholder": "I believe that..."},
                    {"label": "Main Point 1:", "placeholder": "First,..."},
                    {"label": "Main Point 2:", "placeholder": "Second,..."},
                    {"label": "Conclusion:", "placeholder": "That's why..."}
                ]
            }
            
            logger.info(f"Returning state with structure-focused fallback and keys: {list(new_state.keys())}")
            return new_state
        else:
            logger.info("Creating vocabulary-focused fallback scaffolding")
            
            new_state["selected_scaffold_type"] = "Sentence Starter Bank"
            new_state["scaffold_adaptation_plan"] = "Provide student with general sentence starters for speaking responses"
            new_state["scaffold_content_type"] = "bank"
            new_state["scaffold_content_name"] = "General TOEFL Speaking Starters"
            new_state["scaffold_content"] = {
                "categories": [
                    {
                        "name": "Introductions", 
                        "items": [
                            "In my opinion...", 
                            "I believe that...", 
                            "From my perspective..."
                        ]
                    },
                    {
                        "name": "Supporting Points", 
                        "items": [
                            "One reason is...", 
                            "An important factor is...", 
                            "To illustrate this point..."
                        ]
                    },
                    {
                        "name": "Conclusions", 
                        "items": [
                            "In conclusion...", 
                            "Therefore, I think...", 
                            "Based on these points..."
                        ]
                    }
                ]
            }
            
            logger.info("Created fallback vocabulary-focused scaffolding strategy")
            logger.info(f"Returning state with vocabulary-focused fallback and keys: {list(new_state.keys())}")
            return new_state

    scaffold_examples = []
    for strategy in scaffolding_strategies:
        try:
            scaffold_type = strategy.get("scaffold_type_selected", "")
            content_type = strategy.get("scaffold_content_delivered_type", "")
            content_name = strategy.get("scaffold_content_delivered_name", "")
            content_json = strategy.get("scaffold_content_delivered_content_json", "{}")
            
            if content_json:
                try:
                    content = json.loads(content_json)
                except:
                    content = {"error": "Could not parse JSON content"}
            else:
                content = {}
                
            scaffold_examples.append({
                "type": scaffold_type,
                "content_type": content_type,
                "content_name": content_name,
                "content": content
            })
        except Exception as e:
            logger.error(f"Error processing scaffold strategy: {e}")
    
    prompt = f"""
    You are a pedagogical planner for an AI Tutor system.
    
    Student Profile: {user_data}
    Primary Struggle: {primary_struggle}
    Learning Objective ID: {learning_objective_id}
    
    You have access to these scaffolding strategy examples:
    {json.dumps(scaffold_examples, indent=2)}
    
    Based on the student's needs and the examples provided, determine:
    1. Which scaffolding type would be most effective
    2. How it should be adapted for this specific student
    3. What specific content should be included in the scaffold
    
    Return JSON:
    {{
        "selected_scaffold_type": "name of the selected scaffold type",
        "scaffold_adaptation_plan": "explanation of how the scaffold should be adapted",
        "scaffold_content_type": "template OR sentence_starters OR guiding_questions OR hint OR simplified_steps_list OR vocabulary_list",
        "scaffold_content_name": "descriptive name for the scaffold",
        "scaffold_content": {{}} // JSON object with the adapted content structure
    }}
    """

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set")
        
        new_state = {key: value for key, value in state.items()}
        
        new_state["selected_scaffold_type"] = "Error"
        new_state["scaffold_adaptation_plan"] = "Error: API key not available"
        
        logger.info(f"Returning state with keys: {list(new_state.keys())}")
        return new_state

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = model.generate_content(prompt)
        response_json = json.loads(response.text)
        
        selected_scaffold_type = response_json.get("selected_scaffold_type", "Unknown")
        scaffold_adaptation_plan = response_json.get("scaffold_adaptation_plan", "")
        scaffold_content_type = response_json.get("scaffold_content_type", "")
        scaffold_content_name = response_json.get("scaffold_content_name", "")
        scaffold_content = response_json.get("scaffold_content", {})
        
        logger.info(f"Selected scaffold type: {selected_scaffold_type}")
        logger.info(f"Selected scaffold content type: {scaffold_content_type}")
        
        new_state = {key: value for key, value in state.items()}
        
        new_state["selected_scaffold_type"] = selected_scaffold_type
        new_state["scaffold_adaptation_plan"] = scaffold_adaptation_plan
        new_state["scaffold_content_type"] = scaffold_content_type
        new_state["scaffold_content_name"] = scaffold_content_name
        new_state["scaffold_content"] = scaffold_content
        
        logger.info(f"Returning state with keys: {list(new_state.keys())}")
        return new_state
        
    except Exception as e:
        logger.error(f"Error in scaffolding planning: {e}")
        
        new_state = {key: value for key, value in state.items()}
        
        new_state["selected_scaffold_type"] = "Error"
        new_state["scaffold_adaptation_plan"] = f"Error in planning: {str(e)}"
        
        logger.info(f"Returning state with keys on error: {list(new_state.keys())}")
        return new_state
