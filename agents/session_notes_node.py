import logging
import os
import json
import enum
from typing import Dict, Any, List, Optional
from datetime import datetime
from state import AgentGraphState
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)


class SessionNotesNode:
    """
    Generates a concise summary of a completed learning task or tutoring session.
    
    Purpose:
    - Generate summaries of learning sessions or tasks
    - Include key concepts, errors, feedback points, vocabulary, and suggestions
    - Format notes for student viewing on P10 page or as session takeaway
    - Provide data for StudentModelNode to save in Mem0
    """
    
    class SessionType(enum.Enum):
        """Enumeration of session summary types"""
        TASK = "task"
        CONVERSATION = "conversation"
        FEEDBACK = "feedback"
        TEACHING = "teaching"
        DRILL = "drill"
        FULL_SESSION = "full_session"
    
    def __init__(self):
        """Initialize the SessionNotesNode"""
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        else:
            logger.error("GOOGLE_API_KEY not found in environment.")
            
        try:
            self.model = genai.GenerativeModel(
                'gemini-2.5-flash-preview-05-20',
                generation_config=GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            logger.debug("SessionNotesNode: GenerativeModel initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing GenerativeModel: {e}")
            self.model = None
    
    async def _extract_session_data(self, state: AgentGraphState) -> Dict[str, Any]:
        """
        Extract relevant data from the state for session notes generation
        
        Args:
            state: The current agent graph state
            
        Returns:
            Dictionary of extracted session data
        """
        session_data = {
            "user_id": state.get("user_id", "unknown_user"),
            "session_id": state.get("session_id", "unknown_session"),
            "timestamp": datetime.now().isoformat(),
            "page_history": state.get("page_history", []),
            "session_duration_mins": state.get("session_duration_mins", 0)
        }
        
        task_details = state.get("current_task", state.get("next_task_details", {}))
        if task_details:
            session_data["task"] = {
                "type": task_details.get("type", "Unknown"),
                "title": task_details.get("title", "Unknown task"),
                "difficulty": task_details.get("difficulty", "Unknown"),
                "topic": task_details.get("topic", "Unknown"),
                "instructions": task_details.get("instructions", "")
            }
        
        chat_history = state.get("chat_history", [])
        if chat_history:
            session_data["chat_history"] = chat_history[-10:]
        
        diagnosis = state.get("diagnosis_result", {})
        if diagnosis:
            session_data["diagnosis"] = {
                "summary": diagnosis.get("summary", ""),
                "strengths": diagnosis.get("strengths", []),
                "improvement_areas": diagnosis.get("improvement_areas", [])
            }
        
        feedback = state.get("feedback_result", {})
        if feedback:
            session_data["feedback"] = {
                "general_feedback": feedback.get("general_feedback", ""),
                "specific_points": feedback.get("points", []),
                "score": feedback.get("score", None)
            }
        
        teaching_content = state.get("teaching_content", {})
        if teaching_content:
            session_data["teaching"] = {
                "topic": teaching_content.get("topic", ""),
                "concepts": teaching_content.get("concepts", []),
                "examples": teaching_content.get("examples", [])
            }
        
        drill_history = state.get("drill_history", [])
        if drill_history:
            session_data["drills"] = {
                "topic": state.get("drill_topic", ""),
                "difficulty": state.get("drill_difficulty", ""),
                "history": drill_history
            }
            
        return session_data
    
    async def _generate_session_notes(self, 
                                     session_data: Dict[str, Any],
                                     session_type: SessionType) -> Dict[str, Any]:
        """
        Generate session notes using the LLM
        
        Args:
            session_data: Extracted session data
            session_type: Type of session summary to generate
            
        Returns:
            Generated session notes
        """
        if not self.model:
            return {"error": "LLM model not available"}
        
        try:
            instruction = ""
            if session_type == self.SessionType.TASK:
                instruction = "Focus on summarizing this specific task, performance, and key learnings."
            elif session_type == self.SessionType.CONVERSATION:
                instruction = "Focus on summarizing the conversation flow, questions asked, and information exchanged."
            elif session_type == self.SessionType.FEEDBACK:
                instruction = "Focus on summarizing the feedback provided, strengths, and areas for improvement."
            elif session_type == self.SessionType.TEACHING:
                instruction = "Focus on summarizing the teaching content, key concepts, and examples."
            elif session_type == self.SessionType.DRILL:
                instruction = "Focus on summarizing the skill drill session, performance, and practice areas."
            else:
                instruction = "Provide a comprehensive summary of the entire tutoring session across all activities."
                
            system_prompt = f"""
            You are an expert TOEFL tutor summarizing a learning session for a student.
            Create a concise, helpful summary that highlights key points from the session.
            {instruction}
            
            Your response MUST be a JSON object with the following structure:
            {{
                "session_title": "Brief descriptive title for this session",
                "key_concepts": ["List of key concepts covered"],
                "vocabulary": [{{"term": "word/phrase", "definition": "simple definition", "example": "usage example"}}],
                "strengths": ["List of strengths demonstrated"],
                "areas_for_improvement": ["List of areas to work on"],
                "next_steps": ["Recommended next steps or practice activities"],
                "summary": "A concise paragraph summarizing the session"
            }}
            
            Keep the summary brief but informative. Focus on the most valuable insights and feedback.
            """
            
            user_prompt = f"""
            Session data: {json.dumps(session_data)}
            
            Generate concise, helpful session notes based on this information.
            """
            
            response = await self.model.generate_content_async(system_prompt + "\n" + user_prompt)
            response_text = response.text
            
            try:
                notes = json.loads(response_text)
                logger.info(f"Successfully generated session notes for {session_type.value} session")
                return notes
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {response_text}")
                return {"error": "Failed to generate session notes"}
                
        except Exception as e:
            logger.error(f"Error in _generate_session_notes: {e}")
            return {"error": f"Error generating session notes: {str(e)}"}
    
    async def _format_notes_for_display(self, notes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the session notes for display to the student
        
        Args:
            notes: The raw generated notes
            
        Returns:
            Formatted notes ready for display
        """
        if "error" in notes:
            return {
                "title": "Session Summary",
                "content": "Sorry, we couldn't generate a full summary for this session."
            }
            
        formatted_notes = {
            "title": notes.get("session_title", "Session Summary"),
            "sections": [
                {
                    "title": "Summary",
                    "content": notes.get("summary", "No summary available.")
                }
            ]
        }
        
        key_concepts = notes.get("key_concepts", [])
        if key_concepts:
            formatted_notes["sections"].append({
                "title": "Key Concepts Covered",
                "content": "\n• " + "\n• ".join(key_concepts)
            })
            
        vocabulary = notes.get("vocabulary", [])
        if vocabulary:
            vocab_content = ""
            for term in vocabulary:
                vocab_content += f"• **{term.get('term', '')}**: {term.get('definition', '')}"  
                if "example" in term:
                    vocab_content += f" (Example: *{term.get('example', '')}*)"
                vocab_content += "\n"
            
            formatted_notes["sections"].append({
                "title": "New Vocabulary",
                "content": vocab_content
            })
            
        strengths = notes.get("strengths", [])
        if strengths:
            formatted_notes["sections"].append({
                "title": "Strengths",
                "content": "\n• " + "\n• ".join(strengths)
            })
            
        improvements = notes.get("areas_for_improvement", [])
        if improvements:
            formatted_notes["sections"].append({
                "title": "Areas for Improvement",
                "content": "\n• " + "\n• ".join(improvements)
            })
            
        next_steps = notes.get("next_steps", [])
        if next_steps:
            formatted_notes["sections"].append({
                "title": "Recommended Next Steps",
                "content": "\n• " + "\n• ".join(next_steps)
            })
            
        return formatted_notes
        

_session_notes = SessionNotesNode()


async def compile_session_notes_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    LangGraph node for compiling session notes for a completed learning task or session.
    This function replaces the stub implementation.
    
    Args:
        state: The current agent graph state
        
    Returns:
        Updated state with session notes
    """
    try:
        user_id = state.get("user_id", "unknown_user")
        session_id = state.get("session_id", "unknown_session")
        
        logger.info(f"SessionNotesNode: Compiling session notes for user_id: {user_id}, session_id: {session_id}")
        
        session_data = await _session_notes._extract_session_data(state)
        
        session_type = SessionNotesNode.SessionType.FULL_SESSION
        if "teaching" in session_data:
            session_type = SessionNotesNode.SessionType.TEACHING
        elif "diagnosis" in session_data:
            session_type = SessionNotesNode.SessionType.FEEDBACK
        elif "drills" in session_data:
            session_type = SessionNotesNode.SessionType.DRILL
        elif "chat_history" in session_data and len(session_data["chat_history"]) > 0:
            session_type = SessionNotesNode.SessionType.CONVERSATION
        elif "task" in session_data:
            session_type = SessionNotesNode.SessionType.TASK
            
        notes = await _session_notes._generate_session_notes(session_data, session_type)
        
        formatted_notes = await _session_notes._format_notes_for_display(notes)
        
        return {
            "session_notes": notes,
            "output_content": {
                "notes": formatted_notes
            }
        }
        
    except Exception as e:
        logger.error(f"Error in compile_session_notes_node: {e}")
        return {
            "output_content": {
                "response": "Sorry, I couldn't compile session notes at this time."
            }
        }


async def compile_session_notes_stub_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Legacy stub function that now calls the real implementation.
    Maintained for backward compatibility.
    
    Args:
        state: The current agent graph state
        
    Returns:
        Result from the real implementation
    """
    logger.info("Using real implementation instead of stub for session notes compilation")
    return await compile_session_notes_node(state)
