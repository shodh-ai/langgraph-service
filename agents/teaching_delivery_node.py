import logging
import os
import json
import enum
from typing import Dict, Any, Optional, List
from state import AgentGraphState
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)


class TeachingDeliveryNode:
    """
    Delivers lessons (from KnowledgeNode) and manages interactive skill drills.
    Handles content delivery and interaction flow for teaching modules and skill drills.
    
    This class is used across multiple pages: P7 (Teaching Q&A) and P9 (Skill Drills).
    """
    
    class TeachingMode(enum.Enum):
        """Enumeration of teaching delivery modes"""
        LESSON = "lesson"
        INTERACTIVE_DRILL = "drill"
        GUIDED_EXAMPLE = "example"
        CONCEPT_EXPLANATION = "concept"
        FEEDBACK_LESSON = "feedback"
        
    def __init__(self):
        """Initialize the TeachingDeliveryNode"""
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
            logger.debug("TeachingDeliveryNode: GenerativeModel initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing GenerativeModel: {e}")
            self.model = None
            
    async def _format_lesson_content(self, lesson_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format lesson content for delivery to the student
        
        Args:
            lesson_data: Raw lesson data from knowledge node
            
        Returns:
            Formatted lesson content ready for delivery
        """
        try:
            if not self.model:
                return {"error": "LLM model not available"}
                
            system_prompt = """
            You are an expert TOEFL tutor responsible for formatting lesson content.
            Take the raw lesson data and transform it into an engaging, clear, and structured format.
            
            Your response MUST be a JSON object with the following structure:
            {
                "title": "Lesson title",
                "introduction": "Brief engaging introduction to the topic",
                "content_blocks": [
                    {
                        "type": "explanation",
                        "content": "Explanation text"
                    },
                    {
                        "type": "example",
                        "content": "Example with highlighted key points"
                    },
                    {
                        "type": "tip",
                        "content": "Helpful tip for the student"
                    }
                ],
                "summary": "Brief summary of key points",
                "ui_actions": [] // Optional UI actions
            }
            """
            
            user_prompt = f"""
            Format the following lesson content:
            {json.dumps(lesson_data)}
            
            Make sure to maintain all the educational content while making it more engaging and clear.
            """
            
            response = await self.model.generate_content_async(system_prompt + "\n" + user_prompt)
            response_text = response.text
            
            try:
                formatted_content = json.loads(response_text)
                logger.info("Successfully formatted lesson content")
                return formatted_content
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {response_text}")
                return {"error": "Failed to format lesson content"}
                
        except Exception as e:
            logger.error(f"Error in _format_lesson_content: {e}")
            return {"error": f"Error formatting lesson content: {str(e)}"}
    
    async def _create_skill_drill_step(self, topic: str, difficulty: str, previous_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a new skill drill exercise step
        
        Args:
            topic: The topic of the skill drill
            difficulty: The difficulty level (easy, medium, hard)
            previous_steps: Previous steps in this skill drill session
            
        Returns:
            A new skill drill step
        """
        try:
            if not self.model:
                return {"error": "LLM model not available"}
                
            system_prompt = """
            You are an expert TOEFL tutor creating interactive skill drills for students.
            Based on the topic, difficulty level, and previous interactions, create the next step in the skill drill.
            
            Your response MUST be a JSON object with the following structure:
            {
                "step_type": "question|example|challenge|reflection",
                "content": "The main content for this step",
                "options": ["option1", "option2", "option3", "option4"],  // if step_type is question
                "correct_answer": "The correct answer",  // if applicable
                "explanation": "Explanation to provide after student responds",
                "ui_actions": [] // Optional UI actions
            }
            """
            
            user_prompt = f"""
            Topic: {topic}
            Difficulty: {difficulty}
            Previous Steps: {json.dumps(previous_steps)}
            
            Create the next logical step in this skill drill sequence.
            """
            
            response = await self.model.generate_content_async(system_prompt + "\n" + user_prompt)
            response_text = response.text
            
            try:
                drill_step = json.loads(response_text)
                logger.info(f"Created new skill drill step of type: {drill_step.get('step_type', 'unknown')}")
                return drill_step
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {response_text}")
                return {"error": "Failed to create skill drill step"}
                
        except Exception as e:
            logger.error(f"Error in _create_skill_drill_step: {e}")
            return {"error": f"Error creating skill drill step: {str(e)}"}
    
    async def _evaluate_drill_response(self, 
                                      student_response: str, 
                                      current_step: Dict[str, Any],
                                      drill_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a student's response to a skill drill step
        
        Args:
            student_response: The student's response text
            current_step: The current skill drill step
            drill_history: History of the drill session
            
        Returns:
            Evaluation results and next step information
        """
        try:
            if not self.model:
                return {"error": "LLM model not available"}
                
            system_prompt = """
            You are an expert TOEFL tutor evaluating a student's response to a skill drill exercise.
            Analyze the student's response and provide constructive feedback.
            
            Your response MUST be a JSON object with the following structure:
            {
                "is_correct": true|false,
                "feedback": "Specific feedback on the student's response",
                "explanation": "Deeper explanation of the concept if needed",
                "next_action": "continue|complete|remediate",
                "ui_actions": [] // Optional UI actions
            }
            """
            
            user_prompt = f"""
            Current Drill Step: {json.dumps(current_step)}
            Student Response: {student_response}
            Drill History: {json.dumps(drill_history)}
            
            Evaluate the student's response and determine the appropriate next action.
            """
            
            response = await self.model.generate_content_async(system_prompt + "\n" + user_prompt)
            response_text = response.text
            
            try:
                evaluation = json.loads(response_text)
                logger.info(f"Evaluated student response as: {evaluation.get('is_correct', False)}")
                return evaluation
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {response_text}")
                return {"error": "Failed to evaluate response"}
                
        except Exception as e:
            logger.error(f"Error in _evaluate_drill_response: {e}")
            return {"error": f"Error evaluating response: {str(e)}"}


_teaching_delivery = TeachingDeliveryNode()


async def deliver_teaching_module_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Delivers a teaching module with structured lesson content.
    Used primarily on P7 (Teaching Q&A).
    
    Args:
        state: The current agent graph state
        
    Returns:
        Updated state with teaching module content
    """
    logger.info(f"Deliver teaching module for user {state.get('user_id', 'unknown_user')}")
    
    topic = state.get("teaching_request", {}).get("topic")
    knowledge_content = state.get("knowledge_content", {})
    
    if not topic or not knowledge_content:
        logger.warning("Missing topic or knowledge content for teaching module")
        return {"output_content": {"response": "I'm not sure what to teach about. Could you specify a topic?", "ui_actions": []}}
    
    logger.info(f"Delivering teaching module on topic: {topic}")
    
    formatted_lesson = await _teaching_delivery._format_lesson_content(knowledge_content)
    
    if "error" in formatted_lesson:
        logger.error(f"Error formatting lesson: {formatted_lesson['error']}")
        return {"output_content": {"response": "I'm having trouble preparing that lesson right now. Let's try something else.", "ui_actions": []}}
    
    return {
        "output_content": {
            "title": formatted_lesson.get("title", f"Lesson: {topic}"),
            "response": formatted_lesson.get("introduction", "Let's learn about this topic."),
            "content_blocks": formatted_lesson.get("content_blocks", []),
            "summary": formatted_lesson.get("summary", ""),
            "ui_actions": formatted_lesson.get("ui_actions", [])
        },
        "teaching_data": {
            "current_module": topic,
            "module_content": formatted_lesson
        }
    }


async def manage_skill_drill_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Manages interactive skill drill exercises.
    Used primarily on P9 (Skill Drills).
    
    Args:
        state: The current agent graph state
        
    Returns:
        Updated state with skill drill step or evaluation
    """
    logger.info(f"Manage skill drill for user {state.get('user_id', 'unknown_user')}")
    
    drill_info = state.get("skill_drill", {})
    student_response = state.get("transcript", "")
    drill_history = state.get("drill_history", [])
    current_step = state.get("current_drill_step", {})
    
    topic = drill_info.get("topic", "")
    difficulty = drill_info.get("difficulty", "medium")
    
    is_new_drill = not current_step
    is_student_response = bool(student_response) and not is_new_drill
    
    if is_new_drill:
        logger.info(f"Starting new skill drill on topic: {topic}, difficulty: {difficulty}")
        
        first_step = await _teaching_delivery._create_skill_drill_step(topic, difficulty, [])
        
        if "error" in first_step:
            logger.error(f"Error creating first drill step: {first_step['error']}")
            return {"output_content": {"response": "I'm having trouble setting up this skill drill. Let's try again later.", "ui_actions": []}}
        
        return {
            "output_content": {
                "title": f"Skill Drill: {topic}",
                "response": first_step.get("content", "Let's practice this skill."),
                "options": first_step.get("options", []),
                "ui_actions": first_step.get("ui_actions", [])
            },
            "current_drill_step": first_step,
            "drill_history": []
        }
    
    elif is_student_response:
        logger.info(f"Processing student response to skill drill: {student_response}")
        
        evaluation = await _teaching_delivery._evaluate_drill_response(
            student_response, current_step, drill_history
        )
        
        if "error" in evaluation:
            logger.error(f"Error evaluating drill response: {evaluation['error']}")
            return {"output_content": {"response": "I couldn't evaluate your response properly. Let's continue.", "ui_actions": []}}
        
        updated_history = drill_history + [{
            "step": current_step,
            "student_response": student_response,
            "evaluation": evaluation
        }]
        
        next_action = evaluation.get("next_action", "continue")
        
        if next_action == "complete":
            return {
                "output_content": {
                    "response": f"Great job completing this drill! {evaluation.get('feedback', '')}",
                    "summary": "Skill drill completed",
                    "ui_actions": evaluation.get("ui_actions", [])
                },
                "drill_history": updated_history,
                "drill_complete": True
            }
            
        elif next_action == "remediate":
            remediation_step = await _teaching_delivery._create_skill_drill_step(
                topic, "easy" if difficulty != "easy" else difficulty, updated_history
            )
            
            return {
                "output_content": {
                    "response": f"{evaluation.get('feedback', '')} {evaluation.get('explanation', '')}",
                    "next_step": remediation_step.get("content", "Let's try something a bit different."),
                    "options": remediation_step.get("options", []),
                    "ui_actions": remediation_step.get("ui_actions", [])
                },
                "current_drill_step": remediation_step,
                "drill_history": updated_history
            }
            
        else:
            next_step = await _teaching_delivery._create_skill_drill_step(
                topic, difficulty, updated_history
            )
            
            return {
                "output_content": {
                    "response": f"{evaluation.get('feedback', '')}",
                    "next_step": next_step.get("content", "Let's continue with the next question."),
                    "options": next_step.get("options", []),
                    "ui_actions": next_step.get("ui_actions", [])
                },
                "current_drill_step": next_step,
                "drill_history": updated_history
            }
    
    else:
        logger.warning("Unclear skill drill state - no current step and no response")
        return {"output_content": {"response": "I'm not sure where we are in the drill. Let's start over.", "ui_actions": []}}
