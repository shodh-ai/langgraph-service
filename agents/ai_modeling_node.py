import logging
import os
import json
import enum
from typing import Dict, Any, Optional, List
from state import AgentGraphState
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)


class AIModelingNode:
    """
    Generates high-quality model responses and think-aloud explanations for TOEFL tasks.
    
    Purpose:
    - Generate model responses (spoken or written) for TOEFL tasks
    - Prepare think-aloud explanations revealing the expert thought process
    - Output content for modeling pages (P8A for Speaking, P8C for Writing)
    """
    
    class ModelingType(enum.Enum):
        """Enumeration of modeling response types"""
        SPEAKING = "speaking"
        WRITING = "writing"
    
    class AnnotationType(enum.Enum):
        """Enumeration of annotation/explanation types"""
        INLINE = "inline"
        SECTIONED = "sectioned"
        NARRATIVE = "narrative"
        TIMESTAMPED = "timestamped"
    
    def __init__(self):
        """Initialize the AIModelingNode"""
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
            logger.debug("AIModelingNode: GenerativeModel initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing GenerativeModel: {e}")
            self.model = None
    
    async def _analyze_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the TOEFL task to determine key requirements and approach
        
        Args:
            task_data: The task data including prompt, instructions, and constraints
            
        Returns:
            Analysis of the task with key points and strategy
        """
        if not self.model:
            return {"error": "LLM model not available"}
        
        try:
            system_prompt = """
            You are an expert TOEFL instructor analyzing a TOEFL task.
            Break down the task requirements, key points to address, and optimal approach.
            
            Your response MUST be a JSON object with the following structure:
            {
                "task_type": "speaking_independent|speaking_integrated|writing_independent|writing_integrated",
                "key_requirements": ["list of key requirements"],
                "content_points": ["list of content points to address"],
                "structure_recommendation": "recommended structure for the response",
                "time_management": "how to allocate time for this task",
                "common_pitfalls": ["list of common mistakes to avoid"]
            }
            """
            
            user_prompt = f"""
            Task: {json.dumps(task_data)}
            
            Provide a thorough analysis of this TOEFL task to guide creating a model response.
            """
            
            response = await self.model.generate_content_async(system_prompt + "\n" + user_prompt)
            response_text = response.text
            
            try:
                task_analysis = json.loads(response_text)
                logger.info("Successfully analyzed task")
                return task_analysis
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {response_text}")
                return {"error": "Failed to analyze task"}
                
        except Exception as e:
            logger.error(f"Error in _analyze_task: {e}")
            return {"error": f"Error analyzing task: {str(e)}"}
    
    async def _generate_model_response(self, 
                                      task_data: Dict[str, Any], 
                                      task_analysis: Dict[str, Any],
                                      modeling_type: ModelingType) -> Dict[str, Any]:
        """
        Generate a model response for the specified TOEFL task
        
        Args:
            task_data: The task data including prompt and instructions
            task_analysis: Analysis of the task from _analyze_task
            modeling_type: The type of modeling (speaking or writing)
            
        Returns:
            A model response for the task
        """
        if not self.model:
            return {"error": "LLM model not available"}
        
        try:
            is_speaking = modeling_type == self.ModelingType.SPEAKING
            
            system_prompt = f"""
            You are an expert TOEFL instructor creating a model {'speaking' if is_speaking else 'writing'} response.
            Create a high-quality {'spoken' if is_speaking else 'written'} response that would score highly on the TOEFL exam.
            
            Your response MUST be a JSON object with the following structure:
            {{
                "response_text": "The full {'spoken' if is_speaking else 'written'} response",
                "word_count": 0, // Approximate word count
                "estimated_score": "0-30", // Estimated TOEFL score this response would receive
                "strengths": ["list of strengths of this response"]
            }}
            
            Task analysis: {json.dumps(task_analysis)}
            """
            
            user_prompt = f"""
            Task: {json.dumps(task_data)}
            
            Generate a model {'speaking' if is_speaking else 'writing'} response for this task.
            {'Keep the response around 1 minute when spoken aloud.' if is_speaking else 'For integrated writing, aim for 300-350 words. For independent writing, aim for 400-450 words.'}
            """
            
            response = await self.model.generate_content_async(system_prompt + "\n" + user_prompt)
            response_text = response.text
            
            try:
                model_response = json.loads(response_text)
                logger.info(f"Successfully generated model {modeling_type.value} response")
                return model_response
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {response_text}")
                return {"error": "Failed to generate model response"}
                
        except Exception as e:
            logger.error(f"Error in _generate_model_response: {e}")
            return {"error": f"Error generating model response: {str(e)}"}
    
    async def _generate_think_aloud(self,
                                   task_data: Dict[str, Any],
                                   task_analysis: Dict[str, Any],
                                   model_response: Dict[str, Any],
                                   modeling_type: ModelingType,
                                   annotation_type: AnnotationType) -> Dict[str, Any]:
        """
        Generate think-aloud explanations for the model response
        
        Args:
            task_data: The task data including prompt and instructions
            task_analysis: Analysis of the task from _analyze_task
            model_response: The model response generated
            modeling_type: The type of modeling (speaking or writing)
            annotation_type: The type of annotations/explanations to generate
            
        Returns:
            Think-aloud explanations for the model response
        """
        if not self.model:
            return {"error": "LLM model not available"}
        
        try:
            is_speaking = modeling_type == self.ModelingType.SPEAKING
            response_text = model_response.get("response_text", "")
            
            annotation_instructions = ""
            if annotation_type == self.AnnotationType.INLINE:
                annotation_instructions = "Insert explanatory notes directly within the response text using [NOTE: explanation] format."
            elif annotation_type == self.AnnotationType.SECTIONED:
                annotation_instructions = "Break down your explanation by sections (Introduction, Body, Conclusion, etc.)."
            elif annotation_type == self.AnnotationType.NARRATIVE:
                annotation_instructions = "Provide a narrative explaining the entire thought process from start to finish."
            elif annotation_type == self.AnnotationType.TIMESTAMPED:
                annotation_instructions = "For each major point or section, provide a timestamp and explanation (e.g. '0:10 - Introduction begins...')."
            
            system_prompt = f"""
            You are an expert TOEFL instructor explaining the thought process behind a model {'speaking' if is_speaking else 'writing'} response.
            Create a detailed think-aloud explanation that reveals expert thinking behind the response.
            {annotation_instructions}
            
            Your response MUST be a JSON object with the following structure:
            {{
                "explanations": [ // Array of explanation elements
                    {{
                        "type": "section|point|strategy|language",
                        "reference": "Text from the response this explanation refers to",
                        "explanation": "The explanation or annotation"
                    }},
                    // more explanation elements
                ],
                "overall_strategy": "Overall strategy explanation",
                "key_techniques": ["list of key techniques demonstrated"]
            }}
            
            Task analysis: {json.dumps(task_analysis)}
            """
            
            user_prompt = f"""
            Task: {json.dumps(task_data)}
            
            Model Response: {response_text}
            
            Generate a detailed think-aloud explanation for this {'speaking' if is_speaking else 'writing'} response.
            Focus on explaining the choices made, structure, language use, and how it fulfills task requirements.
            """
            
            response = await self.model.generate_content_async(system_prompt + "\n" + user_prompt)
            response_text = response.text
            
            try:
                think_aloud = json.loads(response_text)
                logger.info(f"Successfully generated think-aloud for {modeling_type.value} response")
                return think_aloud
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {response_text}")
                return {"error": "Failed to generate think-aloud explanation"}
                
        except Exception as e:
            logger.error(f"Error in _generate_think_aloud: {e}")
            return {"error": f"Error generating think-aloud explanation: {str(e)}"}
            
    async def generate_model_with_explanation(self,
                                           task_data: Dict[str, Any],
                                           modeling_type: ModelingType,
                                           annotation_type: AnnotationType = None) -> Dict[str, Any]:
        """
        Generate both a model response and its explanation
        
        Args:
            task_data: The task data including prompt and instructions
            modeling_type: The type of modeling (speaking or writing)
            annotation_type: The type of annotations/explanations to generate (default to SECTIONED if None)
            
        Returns:
            Complete modeling package with both response and explanation
        """
        try:
            if annotation_type is None:
                if modeling_type == self.ModelingType.SPEAKING:
                    annotation_type = self.AnnotationType.TIMESTAMPED
                else:
                    annotation_type = self.AnnotationType.SECTIONED
                
            task_analysis = await self._analyze_task(task_data)
            if "error" in task_analysis:
                return task_analysis
                
            model_response = await self._generate_model_response(
                task_data, task_analysis, modeling_type
            )
            if "error" in model_response:
                return model_response
                
            think_aloud = await self._generate_think_aloud(
                task_data, task_analysis, model_response, modeling_type, annotation_type
            )
            if "error" in think_aloud:
                return think_aloud
                
            result = {
                "task_analysis": task_analysis,
                "model_response": model_response,
                "think_aloud": think_aloud,
                "modeling_type": modeling_type.value,
                "annotation_type": annotation_type.value
            }
            
            logger.info(f"Successfully generated complete {modeling_type.value} model with explanations")
            return result
            
        except Exception as e:
            logger.error(f"Error in generate_model_with_explanation: {e}")
            return {"error": f"Failed to generate complete model with explanation: {str(e)}"}


_ai_modeling = AIModelingNode()


async def generate_speaking_model_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    LangGraph node for generating a speaking model response with explanations
    for the TOEFL Speaking task on page P8A.
    
    Args:
        state: The current state of the agent graph
        
    Returns:
        Updated state with the speaking model response and explanations
    """
    try:
        task_data = state.get("current_task", {})
        if not task_data:
            logger.warning("No task data found in state for speaking model generation")
            return {
                "output_content": {
                    "response": "I don't have enough information about the speaking task to provide a model response."
                }
            }
            
        result = await _ai_modeling.generate_model_with_explanation(
            task_data,
            AIModelingNode.ModelingType.SPEAKING,
            AIModelingNode.AnnotationType.TIMESTAMPED
        )
        
        if "error" in result:
            logger.error(f"Error in speaking model generation: {result['error']}")
            return {
                "output_content": {
                    "response": "Sorry, I encountered an error while generating the speaking model. Please try again later."
                }
            }
        
        model_response = result["model_response"]
        think_aloud = result["think_aloud"]
        
        formatted_response = {
            "output_content": {
                "model_response": model_response.get("response_text", ""),
                "estimated_score": model_response.get("estimated_score", "N/A"),
                "explanations": think_aloud.get("explanations", []),
                "overall_strategy": think_aloud.get("overall_strategy", ""),
                "key_techniques": think_aloud.get("key_techniques", []),
                "response_type": "speaking_model"
            },
            "modeling_result": result
        }
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in generate_speaking_model_node: {e}")
        return {
            "output_content": {
                "response": "Sorry, something went wrong while generating the speaking model. Please try again later."
            }
        }


async def generate_writing_model_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    LangGraph node for generating a writing model response with explanations
    for the TOEFL Writing task on page P8C.
    
    Args:
        state: The current state of the agent graph
        
    Returns:
        Updated state with the writing model response and explanations
    """
    try:
        task_data = state.get("current_task", {})
        if not task_data:
            logger.warning("No task data found in state for writing model generation")
            return {
                "output_content": {
                    "response": "I don't have enough information about the writing task to provide a model response."
                }
            }
            
        result = await _ai_modeling.generate_model_with_explanation(
            task_data,
            AIModelingNode.ModelingType.WRITING,
            AIModelingNode.AnnotationType.SECTIONED
        )
        
        if "error" in result:
            logger.error(f"Error in writing model generation: {result['error']}")
            return {
                "output_content": {
                    "response": "Sorry, I encountered an error while generating the writing model. Please try again later."
                }
            }
        
        model_response = result["model_response"]
        think_aloud = result["think_aloud"]
        
        formatted_response = {
            "output_content": {
                "model_response": model_response.get("response_text", ""),
                "word_count": model_response.get("word_count", 0),
                "estimated_score": model_response.get("estimated_score", "N/A"),
                "explanations": think_aloud.get("explanations", []),
                "overall_strategy": think_aloud.get("overall_strategy", ""),
                "key_techniques": think_aloud.get("key_techniques", []),
                "response_type": "writing_model"
            },
            "modeling_result": result
        }
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in generate_writing_model_node: {e}")
        return {
            "output_content": {
                "response": "Sorry, something went wrong while generating the writing model. Please try again later."
            }
        }
