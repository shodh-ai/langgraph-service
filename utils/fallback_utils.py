"""
Fallback utilities for when Google Cloud authentication is not available.
This module provides simple fallback implementations for the agent nodes
that would normally use Vertex AI Gemini models.
"""

import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class FallbackGenerativeModel:
    """A fallback implementation of the Vertex AI GenerativeModel class."""
    
    def __init__(self, model_name="fallback-model"):
        self.model_name = model_name
        logger.info(f"Using fallback generative model: {model_name}")
    
    async def generate_content_async(self, contents, generation_config=None):
        """Generate content without using Vertex AI."""
        logger.info(f"Generating fallback content with config: {generation_config}")
        
        # Extract prompt from contents
        prompt = ""
        if isinstance(contents, str):
            prompt = contents
        elif isinstance(contents, list):
            for item in contents:
                if isinstance(item, str):
                    prompt += item
                elif hasattr(item, 'text') and item.text:
                    prompt += item.text
        
        # Log a truncated version of the prompt for debugging
        truncated_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info(f"Prompt (truncated): {truncated_prompt}")
        
        # Return a fallback response based on the task
        return FallbackResponse(self._generate_fallback_response(prompt))
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a fallback response based on the prompt content."""
        # Welcome greeting fallback
        if "welcome" in prompt.lower() or "greeting" in prompt.lower():
            return """
            Hello! Welcome to the TOEFL tutor. I'm here to help you prepare for your TOEFL exam.
            We'll work together on improving your English skills, focusing on the areas that will
            help you succeed in the test. What would you like to work on today?
            """
        
        # Curriculum navigation fallback
        elif "curriculum" in prompt.lower() or "next step" in prompt.lower():
            return json.dumps({
                "next_step": "introduction",
                "rationale": "Starting with an introduction to establish baseline understanding.",
                "learning_objectives": ["Understand TOEFL test format", "Identify personal strengths and areas for improvement"]
            })
        
        # Socratic questioning fallback
        elif "question" in prompt.lower() or "socratic" in prompt.lower():
            return """
            That's an interesting point. Have you considered how this approach might work in different contexts?
            What evidence supports your reasoning? How might this connect to other aspects of language learning?
            """
        
        # Default fallback
        else:
            return """
            I understand this is an important part of your learning journey. Let's continue working
            on developing your skills in this area. Would you like to try a practice exercise or
            would you prefer some additional explanation?
            """


class FallbackResponse:
    """A fallback implementation of the Vertex AI response object."""
    
    def __init__(self, text):
        self.text = text.strip()
    
    def __str__(self):
        return self.text


def get_model_with_fallback(model_name="gemini-2.5-flash-preview-05-20"):
    """
    Attempt to load the Vertex AI model, but fall back to the fallback implementation
    if Google Cloud authentication is not available.
    """
    try:
        # Try to import and initialize Vertex AI
        import vertexai
        from vertexai.generative_models import GenerativeModel
        
        # Try to initialize Vertex AI using Application Default Credentials
        try:
            # No need to specify GOOGLE_APPLICATION_CREDENTIALS
            # ADC will be used automatically if you've run 'gcloud auth application-default login'
            project_id = "windy-orb-460108-t0"
            location = "us-central1"
            vertexai.init(project=project_id, location=location)
            model = GenerativeModel(model_name)
            logger.info(f"Successfully loaded Vertex AI model: {model_name} using Application Default Credentials")
            return model
        except Exception as e:
            logger.warning(f"Failed to initialize Vertex AI with Application Default Credentials: {e}")
            return FallbackGenerativeModel(model_name)
            
    except ImportError as e:
        logger.warning(f"Vertex AI not available: {e}")
        return FallbackGenerativeModel(model_name)
