import logging
import os
import copy
from typing import Dict, Any, List
import json

from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

FALLBACK_COWRITING_OUTPUT = {
    "text_for_tts": "I notice you're working on your writing. Would you like me to help you with any specific part of it?",
    "ui_components": [
        {
            "type": "text",
            "content": "I notice you're working on your writing. Would you like me to help you with any specific part of it?"
        },
        {
            "type": "button",
            "content": "Yes, help me improve my text",
            "action": "SUGGEST_IMPROVEMENTS"
        }
    ]
}

def cowriting_generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Entering cowriting_generator_node")
    
    state_copy = copy.deepcopy(state)
    
    try:
        selected_intervention = state_copy.get("selected_cowriting_intervention", {})
        
        if not selected_intervention:
            logger.warning("No selected cowriting intervention available")
            state_copy["cowriting_output"] = FALLBACK_COWRITING_OUTPUT
            state_copy["output_content"] = {
                "text_for_tts": FALLBACK_COWRITING_OUTPUT["text_for_tts"],
                "ui_actions": FALLBACK_COWRITING_OUTPUT["ui_components"]
            }
            state_copy["task_suggestion_llm_output"] = FALLBACK_COWRITING_OUTPUT["text_for_tts"]
            return state_copy
        
        ai_spoken_text = selected_intervention.get("ai_spoken_or_suggested_text", 
                                                 selected_intervention.get("AI_Spoken_or_Suggested_Text", ""))
        intervention_type = selected_intervention.get("intervention_type", 
                                                   selected_intervention.get("Intervention_Type", ""))
        
        ui_action_hints = []
        if "ui_action_hints" in selected_intervention:
            ui_action_hints = selected_intervention.get("ui_action_hints", [])
        elif "Associated_UI_Action_Hints_JSON" in selected_intervention:
            try:
                json_str = selected_intervention.get("Associated_UI_Action_Hints_JSON")
                if json_str and not isinstance(json_str, list):
                    ui_action_hints = json.loads(json_str)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Could not parse UI action hints: {e}")
        
        original_text = selected_intervention.get("original_student_text_if_revision", 
                                                selected_intervention.get("Original_Student_Text_if_Revision", ""))
        suggested_revision = selected_intervention.get("suggested_ai_revision_if_any", 
                                                     selected_intervention.get("Suggested_AI_Revision_if_Any", ""))
        
        tts_text = ai_spoken_text
        
        ui_components = []
        
        ui_components.append({
            "type": "text",
            "content": ai_spoken_text
        })
        
        if intervention_type == "SuggestRephrasing" and original_text and suggested_revision:
            ui_components.append({
                "type": "revision",
                "original": original_text,
                "suggestion": suggested_revision,
                "action": "APPLY_REVISION"
            })
            
        elif intervention_type == "OfferVocab":
            vocab_options = suggested_revision.split(",") if suggested_revision else []
            if vocab_options:
                ui_components.append({
                    "type": "options",
                    "options": [option.strip() for option in vocab_options],
                    "action": "SELECT_VOCABULARY"
                })
                
        elif intervention_type == "AskSocraticQuestion":
            ui_components.append({
                "type": "reflection",
                "prompt": ai_spoken_text,
                "action": "RESPOND_TO_QUESTION"
            })
            
        ui_components.append({
            "type": "button",
            "content": "I need more help",
            "action": "REQUEST_MORE_HELP"
        })
        
        cowriting_output = {
            "text_for_tts": tts_text,
            "ui_components": ui_components
        }
        
        state_copy["cowriting_output"] = cowriting_output
        
        state_copy["output_content"] = {
            "text_for_tts": tts_text,
            "ui_actions": ui_components
        }
        
        state_copy["task_suggestion_llm_output"] = tts_text
        
        logger.info("Successfully generated cowriting output")
        return state_copy
        
    except Exception as e:
        logger.error(f"Error in cowriting_generator_node: {str(e)}")
        
        state_copy["cowriting_output"] = FALLBACK_COWRITING_OUTPUT
        
        state_copy["output_content"] = {
            "text_for_tts": FALLBACK_COWRITING_OUTPUT["text_for_tts"],
            "ui_actions": FALLBACK_COWRITING_OUTPUT["ui_components"]
        }
        
        state_copy["task_suggestion_llm_output"] = FALLBACK_COWRITING_OUTPUT["text_for_tts"]
        
        state_copy["error"] = f"Cowriting generator error: {str(e)}"
        
        return state_copy
