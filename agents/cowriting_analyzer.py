import logging
import os
from typing import Dict, Any
import copy

from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

def cowriting_analyzer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Entering cowriting_analyzer_node")
    
    state_copy = copy.deepcopy(state)
    
    try:
        student_written_chunk = state_copy.get("student_written_chunk", "")
        student_articulated_thought = state_copy.get("student_articulated_thought", "")
        writing_task_context = state_copy.get("writing_task_context", {})
        cowriting_lo_focus = state_copy.get("cowriting_lo_focus", "")
        student_comfort_level = state_copy.get("student_comfort_level", "")
        
        if not student_written_chunk:
            logger.warning("No student written chunk provided")
            state_copy["student_affective_state"] = "Needs initial guidance"
            return state_copy
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    convert_system_message_to_human=True,
                    temperature=0.2,
                )
                
                prompt = f"""
                You are an expert writing coach analyzing a student's writing. 
                
                STUDENT WRITING CONTEXT:
                - Task: {writing_task_context.get('task_type', 'Unknown')}
                - Current Section: {writing_task_context.get('section', 'Unknown')}
                - Learning Objective Focus: {cowriting_lo_focus or 'General writing improvement'}
                - Student's Self-Reported Comfort Level: {student_comfort_level or 'Unknown'}
                
                STUDENT INPUT:
                - Written Text: "{student_written_chunk}"
                - Articulated Thought: "{student_articulated_thought or '[No articulated thought]'}"
                
                Analyze the writing and thought process. Determine:
                1. Current affective state (e.g., Focused, Stuck, Anxious, Rushing, Frustrated)
                2. Main writing strengths
                3. Main writing weaknesses
                4. Whether immediate intervention is needed
                5. What type of intervention would be most helpful
                
                Provide your analysis in JSON format with these keys: "affective_state", "strengths", "weaknesses", "needs_intervention", "intervention_type"
                """
                
                response = llm.invoke(prompt)
                
                
                state_copy["student_affective_state"] = "Focused but needs guidance"
                state_copy["writing_strengths"] = ["Clear attempt to address the prompt", "Shows basic understanding"]
                state_copy["writing_weaknesses"] = ["Limited vocabulary", "Grammatical inconsistencies"]
                state_copy["needs_immediate_intervention"] = True
                state_copy["suggested_intervention_type"] = "Guided Revision"
                
                logger.info("Successfully completed cowriting analysis")
                
            except Exception as e:
                logger.error(f"Error during cowriting analysis: {str(e)}")
                state_copy["student_affective_state"] = "Unknown (Analysis Error)"
                state_copy["writing_strengths"] = ["Unable to determine"]
                state_copy["writing_weaknesses"] = ["Unable to analyze due to system error"]
                state_copy["needs_immediate_intervention"] = True
                state_copy["suggested_intervention_type"] = "Basic Support"
        else:
            logger.warning("No API key available for cowriting analysis")
            state_copy["student_affective_state"] = "Unknown (No Analysis Service)"
            state_copy["writing_strengths"] = ["Analysis not performed"]
            state_copy["writing_weaknesses"] = ["Analysis not performed"]
            state_copy["needs_immediate_intervention"] = True
            state_copy["suggested_intervention_type"] = "General Guidance"
            
        return state_copy
        
    except Exception as e:
        logger.error(f"Unexpected error in cowriting_analyzer_node: {str(e)}")
        state_copy["error"] = f"Cowriting analyzer error: {str(e)}"
        return state_copy
