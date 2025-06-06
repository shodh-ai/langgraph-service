import logging
import yaml
import os
import json
import enum
from typing import Dict, Any, Optional, List
from state import AgentGraphState
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)


class ConversationalTurnManagerNode:
    """
    Manages general turn-by-turn dialogue when the AI is in a direct conversation with the student.
    Handles NLU for student utterances and routes to appropriate knowledge/action nodes or generates conversational replies.
    
    This class is used across multiple pages: P1 (Rox Welcome), P6 (Feedback Q&A), P7 (Teaching Q&A), P8 (Modeling Q&A).
    """
    
    class IntentType(enum.Enum):
        """Enumeration of possible intents detected in student utterances"""
        GENERAL_QUESTION = "general_question"
        CLARIFICATION = "clarification"
        FEEDBACK_QUESTION = "feedback_question"
        TEACHING_QUESTION = "teaching_question"
        EXERCISE_QUESTION = "exercise_question"
        OFF_TOPIC = "off_topic"
        UNKNOWN = "unknown"
        
    def __init__(self):
        """Initialize the ConversationalTurnManagerNode"""
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
            logger.debug("ConversationalTurnManagerNode: GenerativeModel initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing GenerativeModel: {e}")
            self.model = None
            
    async def detect_intent(self, utterance: str, context: Dict[str, Any]) -> IntentType:
        """
        Analyzes student utterance to determine the intent
        
        Args:
            utterance: The student's utterance text
            context: Additional context about the conversation and current page
            
        Returns:
            The detected intent type
        """
        if not self.model or not utterance.strip():
            return self.IntentType.UNKNOWN
        
        try:
            system_prompt = """
            You are an intent classifier for an educational AI assistant named Rox.
            Analyze the student's message and classify it into one of these categories:
            - general_question: General questions about content or information
            - clarification: Student asking for clarification about something
            - feedback_question: Questions specifically about feedback received
            - teaching_question: Questions related to teaching methods or lessons
            - exercise_question: Questions about specific exercises or tasks
            - off_topic: Queries unrelated to the educational context
            - unknown: When intent cannot be determined
            
            Return ONLY the category name as a string, nothing else.
            """
            
            user_prompt = f"""Current page: {context.get('current_page', 'unknown')}
            Previous context: {context.get('previous_utterances', [])}
            Student utterance: {utterance}
            """
            
            response = await self.model.generate_content_async(system_prompt + "\n" + user_prompt)
            intent_text = response.text.strip().lower()
            
            for intent_type in self.IntentType:
                if intent_type.value == intent_text:
                    logger.info(f"Detected intent: {intent_type} for utterance: {utterance}")
                    return intent_type
                    
            logger.warning(f"Unknown intent detected: {intent_text}. Defaulting to UNKNOWN")
            return self.IntentType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Error during intent detection: {e}")
            return self.IntentType.UNKNOWN
    
    async def route_to_knowledge_node(self, intent: IntentType, state: AgentGraphState) -> Dict[str, Any]:
        """
        Routes the conversation to the appropriate knowledge or action node based on intent
        
        Args:
            intent: The detected intent type
            state: The current agent graph state
            
        Returns:
            Updated state with routing information
        """
        current_page = state.get("current_context", {}).get("page", "unknown")
        logger.info(f"Routing on page {current_page} with intent {intent}")
        
        if intent == self.IntentType.FEEDBACK_QUESTION:
            return {"routing": {"next_node": "feedback_knowledge_node"}}
            
        elif intent == self.IntentType.TEACHING_QUESTION:
            return {"routing": {"next_node": "teaching_knowledge_node"}}
            
        elif intent == self.IntentType.EXERCISE_QUESTION:
            return {"routing": {"next_node": "exercise_knowledge_node"}}
            
        else:
            return {"routing": {"next_node": "general_conversation_node"}}
    
    async def generate_conversational_response(self, utterance: str, state: AgentGraphState) -> Dict[str, Any]:
        """
        Generates a direct conversational response when no specialized knowledge node is needed
        
        Args:
            utterance: The student's utterance
            state: The current agent graph state
            
        Returns:
            Response content to be sent to the student
        """
        if not self.model:
            return {"output_content": {"response": "I'm sorry, I'm having trouble processing that right now.", "ui_actions": []}}
            
        try:
            chat_history = state.get("chat_history", [])
            student_memory = state.get("student_memory_context", {})
            current_page = state.get("current_context", {}).get("page", "unknown")
            
            system_prompt = """
            You are Rox, a friendly and encouraging AI guide for TOEFL preparation.
            Respond to the student's message in a helpful and supportive way.
            Keep your responses brief and focused on the student's question.
            
            Your response MUST be a JSON object with the following structure:
            {
                "response": "Your conversational response text",
                "ui_actions": [] // Optional array of UI action objects if needed
            }
            """
            
            context_str = f"Current page: {current_page}\n"
            
            if chat_history:
                context_str += "Recent conversation:\n"
                for turn in chat_history[-3:]:
                    if "student" in turn:
                        context_str += f"Student: {turn['student']}\n"
                    if "ai" in turn:
                        context_str += f"Rox: {turn['ai']}\n"
            
            student_info = ""
            if student_memory:
                student_info = f"Student knowledge: {student_memory.get('knowledge_areas', {})}\n"
                student_info += f"Student challenges: {student_memory.get('challenge_areas', {})}\n"
            
            user_prompt = f"{context_str}\n{student_info}\nStudent message: {utterance}"
            
            response = await self.model.generate_content_async(system_prompt + "\n" + user_prompt)
            response_text = response.text
            
            try:
                response_json = json.loads(response_text)
                
                ai_response = response_json.get("response", "I'm sorry, could you rephrase that?")
                ui_actions = response_json.get("ui_actions", [])
                
                new_chat_history = chat_history + [
                    {"student": utterance},
                    {"ai": ai_response}
                ]
                
                return {
                    "output_content": {"response": ai_response, "ui_actions": ui_actions},
                    "chat_history": new_chat_history
                }
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {response_text}")
                return {"output_content": {"response": "I'm having trouble understanding. Could you rephrase that?", "ui_actions": []}}
                
        except Exception as e:
            logger.error(f"Error generating conversational response: {e}")
            return {"output_content": {"response": "I'm sorry, something went wrong. Let's try again.", "ui_actions": []}}

PROMPTS_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'llm_prompts.yaml')

PROMPTS = {}
try:
    with open(PROMPTS_FILE_PATH, 'r') as f:
        loaded_prompts = yaml.safe_load(f)
        if loaded_prompts and 'PROMPTS' in loaded_prompts:
            PROMPTS = loaded_prompts['PROMPTS']
        else:
            logger.error(f"Could not find 'PROMPTS' key in {PROMPTS_FILE_PATH}")
except FileNotFoundError:
    logger.error(f"Prompts file not found at {PROMPTS_FILE_PATH}")
except yaml.YAMLError as e:
    logger.error(f"Error parsing YAML from {PROMPTS_FILE_PATH}: {e}")

_conversation_manager = ConversationalTurnManagerNode()

async def process_conversational_turn_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Manages general turn-by-turn dialogue when the AI is in a direct conversation with the student.
    Handles NLU for student utterances and routes to appropriate knowledge/action nodes or generates conversational replies.
    
    Args:
        state: The current agent graph state
        
    Returns:
        Updated state with either routing information or a direct conversational response
    """
    logger.info(f"Process conversational turn for user {state.get('user_id', 'unknown_user')}")
    
    utterance = state.get("transcript", "")
    if not utterance:
        logger.warning("Empty transcript in process_conversational_turn_node")
        return {"output_content": {"response": "I didn't catch that. Could you say it again?", "ui_actions": []}}
    
    current_context = state.get("current_context", {})
    current_page = current_context.get("page", "unknown")
    
    logger.info(f"Processing utterance on page {current_page}: {utterance}")
    
    intent_context = {
        "current_page": current_page,
        "previous_utterances": [turn.get("student", "") for turn in state.get("chat_history", [])[-3:] if "student" in turn]
    }
    
    intent = await _conversation_manager.detect_intent(utterance, intent_context)
    logger.info(f"Detected intent: {intent} for utterance: {utterance}")
    
    if intent in [
        _conversation_manager.IntentType.FEEDBACK_QUESTION,
        _conversation_manager.IntentType.TEACHING_QUESTION, 
        _conversation_manager.IntentType.EXERCISE_QUESTION
    ]:
        routing_info = await _conversation_manager.route_to_knowledge_node(intent, state)
        logger.info(f"Routing to node: {routing_info.get('routing', {}).get('next_node')}")
        return routing_info
    else:
        response_data = await _conversation_manager.generate_conversational_response(utterance, state)
        logger.info(f"Generated direct conversational response: {response_data.get('output_content', {}).get('response')}")
        return response_data

async def handle_home_greeting_node(state: AgentGraphState) -> dict:
    """
    Generates a personalized welcome greeting using an LLM based on the student's name and persona.
    The LLM is expected to return a JSON object with a 'greeting_tts' field.
    
    Args:
        state: The current agent graph state
        
    Returns:
        Dict with output_content update, including the LLM-generated greeting.
    """
    student_name = "Harshit"
    logger.info(f"ConversationalManagerNode: Using hardcoded student_name: '{student_name}' for testing LLM call.")


    greeting_prompt_config = PROMPTS.get('welcome_greeting')
    if not greeting_prompt_config:
        logger.error("Welcome greeting prompt configuration not found. Falling back to default.")
        return {"output_content": {"response": f"Hello {student_name}! Welcome.", "ui_actions": []}}

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment. Falling back to default greeting.")
        return {"output_content": {"response": f"Hello {student_name}! Welcome.", "ui_actions": []}}

    try:
        logger.debug("Attempting genai.configure()...")
        genai.configure(api_key=api_key)
        logger.debug("genai.configure() successful.")
        
        logger.debug(f"Attempting to initialize GenerativeModel: gemini-2.5-flash-preview-05-20")
        model = genai.GenerativeModel(
            'gemini-2.5-flash-preview-05-20',
            generation_config=GenerationConfig(
                response_mime_type="application/json"
            )
        )
        logger.debug("GenerativeModel initialized successfully.")

        persona_details = "Your friendly and encouraging AI guide, Rox."
        system_prompt_text = greeting_prompt_config.get('system_prompt', '').format(
            student_name=student_name,
            persona_details=persona_details
        )
        user_prompt_text = greeting_prompt_config.get('user_prompt', '')
        full_prompt = f"{system_prompt_text}\n\n{user_prompt_text}"
        
        logger.info(f"GOOGLE_API_KEY loaded: {api_key[:5]}...{api_key[-5:] if len(api_key) > 10 else ''}")
        logger.info(f"Full prompt for greeting LLM: {full_prompt}")
        
        raw_llm_response_text = ""
        try:
            logger.debug("Attempting model.generate_content_async()...")
            response = await model.generate_content_async(full_prompt)
            logger.debug("model.generate_content_async() successful.")
            raw_llm_response_text = response.text
            logger.info(f"Raw LLM Response for greeting: {raw_llm_response_text}")
        except Exception as gen_err:
            logger.error(f"Error during model.generate_content_async(): {gen_err}", exc_info=True)
            fallback_tts = f"Hello {student_name}! Welcome. (LLM Generation Error)"
            logger.info(f"ConversationalManagerNode: Using fallback greeting due to generation error: {fallback_tts}")
            return {"greeting_data": {"greeting_tts": fallback_tts}}

        try:
            llm_output_json = json.loads(raw_llm_response_text)
            greeting_tts = llm_output_json.get("greeting_tts", f"Hello {student_name}! I'm Rox, welcome! (JSON Key Missing)")
        except json.JSONDecodeError as json_err:
            logger.error(f"JSONDecodeError parsing LLM greeting response. Error: {json_err}. Raw text: {raw_llm_response_text}")
            greeting_tts = f"Hello {student_name}! I'm Rox. (JSON Parse Error)"
        except Exception as parse_err:
            logger.error(f"Unexpected error parsing LLM greeting response. Error: {parse_err}. Raw text: {raw_llm_response_text}")
            greeting_tts = f"Hello {student_name}! I'm Rox. (Unexpected Parse Error)"

        logger.info(f"ConversationalManagerNode: LLM-generated greeting: {greeting_tts}")
        return {"greeting_data": {"greeting_tts": greeting_tts}}

    except Exception as e:
        logger.error(f"Outer error in handle_home_greeting_node (e.g., config, model init): {e}", exc_info=True)
        fallback_tts = f"Hello {student_name}! Welcome. (LLM Setup Error)"
        logger.info(f"ConversationalManagerNode: Using fallback greeting due to setup error: {fallback_tts}")
        return {"greeting_data": {"greeting_tts": fallback_tts}}
