import logging
import yaml
import os
import json
import enum
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from state import AgentGraphState
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)

PROMPTS_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'llm_prompts.yaml')

PROMPTS = {}
try:
    with open(PROMPTS_FILE_PATH, 'r') as f:
        loaded_prompts = yaml.safe_load(f)
        if loaded_prompts and 'PROMPTS' in loaded_prompts:
            PROMPTS = loaded_prompts['PROMPTS']
        else:
            logger.error(f"Could not find 'PROMPTS' key in {PROMPTS_FILE_PATH} for practice_selector_node")
except FileNotFoundError:
    logger.error(f"Prompts file not found at {PROMPTS_FILE_PATH} for practice_selector_node")
except yaml.YAMLError as e:
    logger.error(f"Error parsing YAML from {PROMPTS_FILE_PATH} for practice_selector_node: {e}")


class PracticeSelectorNode:
    """
    Selects or generates appropriate follow-up practice exercises or skill drills based on student needs.
    
    Purpose:
    - Analyze student strengths and weaknesses from previous interactions
    - Select appropriate practice exercises that target areas needing improvement
    - Generate custom drills when predefined ones don't match specific needs
    - Support the curriculum flow by suggesting relevant practice activities
    """
    
    class PracticeType(enum.Enum):
        """Enumeration of practice exercise types"""
        VOCABULARY = "vocabulary"
        GRAMMAR = "grammar"
        PRONUNCIATION = "pronunciation"
        ORGANIZATION = "organization"
        FLUENCY = "fluency"
        COMPREHENSION = "comprehension"
        ARGUMENTATION = "argumentation"
        TRANSITIONS = "transitions"
        PARAPHRASING = "paraphrasing"
        NOTE_TAKING = "note_taking"
    
    class SkillArea(enum.Enum):
        """Enumeration of TOEFL skill areas"""
        SPEAKING = "speaking"
        WRITING = "writing"
        READING = "reading"
        LISTENING = "listening"
        INTEGRATED = "integrated"
    
    def __init__(self):
        """Initialize the PracticeSelectorNode"""
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
            logger.debug("PracticeSelectorNode: GenerativeModel initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing GenerativeModel: {e}")
            self.model = None
            
        self._instance = self
        
        self.practice_database = {
            "speaking": {
                "pronunciation": [
                    {
                        "id": "PRN_MINIMAL_PAIRS",
                        "title": "Minimal Pairs Practice",
                        "description": "Practice distinguishing and pronouncing commonly confused sound pairs in English.",
                        "difficulty": "beginner",
                        "estimated_time_minutes": 5,
                        "instructions": "Listen to each pair of words and repeat them, focusing on the difference in pronunciation."
                    },
                    {
                        "id": "PRN_STRESS_PATTERNS",
                        "title": "Word Stress Patterns",
                        "description": "Practice correct stress placement in multi-syllable words.",
                        "difficulty": "intermediate",
                        "estimated_time_minutes": 8,
                        "instructions": "For each word, identify and emphasize the stressed syllable."
                    }
                ],
                "fluency": [
                    {
                        "id": "FLU_QUICK_RESPONSES",
                        "title": "Quick Response Drill",
                        "description": "Practice giving quick, fluent responses to common questions.",
                        "difficulty": "intermediate",
                        "estimated_time_minutes": 10,
                        "instructions": "Respond to each question with a complete answer within 15 seconds."
                    }
                ],
                "organization": [
                    {
                        "id": "ORG_RESPONSE_STRUCTURE",
                        "title": "Response Structure Practice",
                        "description": "Practice organizing speaking responses with clear introductions, body points, and conclusions.",
                        "difficulty": "intermediate",
                        "estimated_time_minutes": 15,
                        "instructions": "For each prompt, outline your response structure before speaking."
                    }
                ]
            },
            "writing": {
                "organization": [
                    {
                        "id": "ORG_THESIS_DEVELOPMENT",
                        "title": "Thesis Statement Development",
                        "description": "Practice writing clear, strong thesis statements.",
                        "difficulty": "intermediate",
                        "estimated_time_minutes": 12,
                        "instructions": "For each topic, write a thesis statement that clearly states your position and previews your main points."
                    }
                ],
                "transitions": [
                    {
                        "id": "TRN_PARAGRAPH_CONNECTIONS",
                        "title": "Paragraph Transition Practice",
                        "description": "Practice connecting paragraphs with smooth transitions.",
                        "difficulty": "advanced",
                        "estimated_time_minutes": 15,
                        "instructions": "For each pair of paragraphs, write a transition sentence that logically connects them."
                    }
                ]
            },
            "integrated": {
                "note_taking": [
                    {
                        "id": "NOT_LECTURE_NOTES",
                        "title": "Lecture Note-Taking Practice",
                        "description": "Practice taking organized notes from a short lecture.",
                        "difficulty": "intermediate",
                        "estimated_time_minutes": 20,
                        "instructions": "Listen to the lecture and take notes capturing main ideas and key details.",
                        "materials": {
                            "audio": "sample_lecture.mp3"
                        }
                    }
                ]
            }
        }
        
    def _extract_student_needs(self, state: AgentGraphState) -> Dict[str, Any]:
        """
        Extract information about student needs from the agent state
        
        Args:
            state: The current agent graph state
            
        Returns:
            A dictionary containing student needs and skill gaps
        """
        student_needs = {}
        
        if 'student_data' in state and state['student_data']:
            student_data = state['student_data']
            if 'skill_levels' in student_data:
                student_needs['skill_levels'] = student_data['skill_levels']
            if 'learning_history' in student_data:
                student_needs['learning_history'] = student_data['learning_history']
            if 'focus_areas' in student_data:
                student_needs['focus_areas'] = student_data['focus_areas']
        
        if 'diagnosis' in state and state['diagnosis']:
            student_needs['diagnosis'] = state['diagnosis']
        
        if 'feedback' in state and state['feedback']:
            student_needs['feedback'] = state['feedback']
        
        if not student_needs:
            student_needs = {
                'skill_levels': {
                    'speaking': 'intermediate',
                    'writing': 'intermediate',
                    'reading': 'intermediate',
                    'listening': 'intermediate'
                },
                'focus_areas': ['organization', 'fluency'],
                'skill_gaps': ['pronunciation', 'transitions']
            }
            logger.warning("No student needs data found in state. Using default values.")
            
        return student_needs
        
    def _identify_skill_gaps(self, student_needs: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Identify skill gaps based on student needs
        
        Args:
            student_needs: Dictionary containing student needs and skill data
            
        Returns:
            List of tuples (skill_area, practice_type) representing skill gaps
        """
        skill_gaps = []
        
        if 'skill_gaps' in student_needs:
            for gap in student_needs['skill_gaps']:
                if gap in ['pronunciation', 'fluency', 'organization']:
                    skill_gaps.append(('speaking', gap))
                elif gap in ['organization', 'transitions', 'argumentation']:
                    skill_gaps.append(('writing', gap))
                elif gap in ['comprehension', 'vocabulary']:
                    skill_gaps.append(('reading', gap))
                    skill_gaps.append(('listening', gap))
                elif gap in ['note_taking', 'paraphrasing']:
                    skill_gaps.append(('integrated', gap))
        
        if 'feedback' in student_needs:
            feedback = student_needs['feedback']
            if isinstance(feedback, dict) and 'areas_for_improvement' in feedback:
                for area in feedback['areas_for_improvement']:
                    if 'pronunciation' in area.lower():
                        skill_gaps.append(('speaking', 'pronunciation'))
                    if 'structure' in area.lower() or 'organization' in area.lower():
                        skill_gaps.append(('speaking', 'organization'))
                        skill_gaps.append(('writing', 'organization'))
                    if 'transition' in area.lower() or 'flow' in area.lower():
                        skill_gaps.append(('writing', 'transitions'))
                    if 'grammar' in area.lower():
                        skill_gaps.append(('speaking', 'grammar'))
                        skill_gaps.append(('writing', 'grammar'))
        
        if not skill_gaps and 'skill_levels' in student_needs:
            skill_levels = student_needs['skill_levels']
            min_skill = min(skill_levels.items(), key=lambda x: {'beginner': 0, 'intermediate': 1, 'advanced': 2}.get(x[1], 1))
            if min_skill[0] == 'speaking':
                skill_gaps = [('speaking', 'fluency'), ('speaking', 'pronunciation')]
            elif min_skill[0] == 'writing':
                skill_gaps = [('writing', 'organization'), ('writing', 'transitions')]
            elif min_skill[0] == 'reading':
                skill_gaps = [('reading', 'comprehension'), ('integrated', 'paraphrasing')]
            elif min_skill[0] == 'listening':
                skill_gaps = [('listening', 'comprehension'), ('integrated', 'note_taking')]
                
        unique_gaps = []
        for gap in skill_gaps:
            if gap not in unique_gaps:
                unique_gaps.append(gap)
                
        return unique_gaps
    
    def _select_practice_exercises(self, skill_gaps: List[Tuple[str, str]], 
                                   difficulty_preference: str = 'intermediate') -> List[Dict[str, Any]]:
        """
        Select appropriate practice exercises based on identified skill gaps
        
        Args:
            skill_gaps: List of tuples (skill_area, practice_type)
            difficulty_preference: Preferred difficulty level
            
        Returns:
            List of practice exercises
        """
        selected_exercises = []
        
        priority_gaps = skill_gaps[:3] if len(skill_gaps) > 3 else skill_gaps
        
        for skill_area, practice_type in priority_gaps:
            if skill_area in self.practice_database and practice_type in self.practice_database[skill_area]:
                available_exercises = self.practice_database[skill_area][practice_type]
                
                matching_exercises = [ex for ex in available_exercises if ex.get('difficulty') == difficulty_preference]
                
                if not matching_exercises and available_exercises:
                    matching_exercises = available_exercises
                
                if matching_exercises:
                    exercise = matching_exercises[0].copy()
                    exercise['skill_area'] = skill_area
                    exercise['practice_type'] = practice_type
                    selected_exercises.append(exercise)
        
        return selected_exercises
    
    async def _generate_custom_drill(self, skill_area: str, practice_type: str, 
                                   difficulty: str, state: AgentGraphState) -> Dict[str, Any]:
        """
        Generate a custom drill when no pre-made drills match the student's needs
        
        Args:
            skill_area: Target skill area (speaking, writing, etc.)
            practice_type: Type of practice (pronunciation, organization, etc.)
            difficulty: Difficulty level
            state: Current agent graph state
            
        Returns:
            A dictionary containing the custom drill details
        """
        logger.info(f"Generating custom {practice_type} drill for {skill_area} at {difficulty} level")
        
        if not self.model or not self.api_key:
            logger.error("Cannot generate custom drill: LLM model not available")
            return self._create_fallback_drill(skill_area, practice_type, difficulty)
        
        student_context = ""
        if 'student_data' in state and state['student_data']:
            if 'name' in state['student_data']:
                student_context += f"Student name: {state['student_data']['name']}\n"
            if 'learning_goals' in state['student_data']:
                student_context += f"Learning goals: {state['student_data']['learning_goals']}\n"
        
        recent_topics = ""
        if 'chat_history' in state and state['chat_history']:
            recent_topics = "Recent conversation topics: "
            for i, entry in enumerate(state['chat_history'][-5:]):
                if 'content' in entry:
                    recent_topics += entry['content'][:100] + " "
        
        prompt_config = PROMPTS.get('custom_drill_generation')
        if not prompt_config:
            logger.error("Custom drill generation prompt configuration not found")
            return self._create_fallback_drill(skill_area, practice_type, difficulty)
        
        system_prompt_text = prompt_config.get('system_prompt', '').format(
            skill_area=skill_area,
            practice_type=practice_type,
            difficulty=difficulty,
            student_context=student_context,
            recent_topics=recent_topics
        )
        user_prompt_text = prompt_config.get('user_prompt', '')
        full_prompt = f"{system_prompt_text}\n\n{user_prompt_text}"
        
        try:
            response = await self.model.generate_content_async(full_prompt)
            raw_response = response.text
            logger.info(f"Raw LLM response for custom drill: {raw_response[:200]}...")
            
            try:
                drill_data = json.loads(raw_response)
                
                required_fields = ['id', 'title', 'description', 'instructions', 'difficulty']
                for field in required_fields:
                    if field not in drill_data:
                        drill_data[field] = "" if field != 'difficulty' else difficulty
                        logger.warning(f"Required field {field} missing in generated drill, using default")
                
                drill_data['skill_area'] = skill_area
                drill_data['practice_type'] = practice_type
                drill_data['custom_generated'] = True
                
                return drill_data
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {raw_response[:200]}")
                return self._create_fallback_drill(skill_area, practice_type, difficulty)
                
        except Exception as e:
            logger.error(f"Error generating custom drill: {e}")
            return self._create_fallback_drill(skill_area, practice_type, difficulty)
    
    def _create_fallback_drill(self, skill_area: str, practice_type: str, difficulty: str) -> Dict[str, Any]:
        """
        Create a fallback drill when custom generation fails
        
        Args:
            skill_area: Target skill area
            practice_type: Type of practice
            difficulty: Difficulty level
            
        Returns:
            A dictionary containing the fallback drill details
        """
        logger.info(f"Creating fallback drill for {practice_type} in {skill_area}")
        
        practice_type_readable = practice_type.replace('_', ' ').title()
        skill_area_readable = skill_area.title()
        
        fallback_drill = {
            "id": f"FALLBACK_{skill_area[:3].upper()}_{practice_type[:3].upper()}",
            "title": f"{practice_type_readable} Practice for {skill_area_readable}",
            "description": f"Practice your {practice_type_readable.lower()} skills for {skill_area_readable.lower()} tasks.",
            "difficulty": difficulty,
            "estimated_time_minutes": 10,
            "instructions": f"Complete these {practice_type_readable.lower()} exercises to improve your {skill_area_readable.lower()} performance.",
            "skill_area": skill_area,
            "practice_type": practice_type,
            "fallback": True
        }
        
        return fallback_drill

    async def select_next_practice(self, state: AgentGraphState) -> List[Dict[str, Any]]:
        """
        Main method to select appropriate practice exercises based on student needs
        
        Args:
            state: The current agent graph state
            
        Returns:
            List of selected practice exercises
        """
        logger.info("PracticeSelectorNode: Selecting next practice exercises")
        
        student_needs = self._extract_student_needs(state)
        
        skill_gaps = self._identify_skill_gaps(student_needs)
        logger.info(f"PracticeSelectorNode: Identified skill gaps: {skill_gaps}")
        
        difficulty_preference = 'intermediate'
        if 'student_data' in state and state['student_data'] and 'difficulty_preference' in state['student_data']:
            difficulty_preference = state['student_data']['difficulty_preference']
        
        selected_exercises = self._select_practice_exercises(skill_gaps, difficulty_preference)
        
        if not selected_exercises and skill_gaps:
            skill_area, practice_type = skill_gaps[0]
            custom_drill = await self._generate_custom_drill(skill_area, practice_type, difficulty_preference, state)
            selected_exercises.append(custom_drill)
        
        logger.info(f"PracticeSelectorNode: Selected {len(selected_exercises)} practice exercises")
        return selected_exercises


_practice_selector = PracticeSelectorNode()


async def select_next_practice_or_drill_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Selects appropriate follow-up practice exercises or skill drills based on student needs.
    
    This node analyzes the student's performance, identifies skill gaps, and selects 
    or generates appropriate practice exercises to address those gaps.
    
    Args:
        state: The current agent graph state
        
    Returns:
        Dictionary with selected practices and ui_actions for routing
    """
    logger.info("PracticeSelectorNode: Selecting next practice or drill")
    
    try:
        selected_practices = await _practice_selector.select_next_practice(state)
        
        if not selected_practices:
            logger.warning("PracticeSelectorNode: No practices selected, using fallback")
            selected_practices = [_practice_selector._create_fallback_drill(
                'speaking', 'organization', 'intermediate')]
        
        next_practice = selected_practices[0]
        
        ui_actions = [
            {
                "action_type": "DISPLAY_PRACTICE_OPTION",
                "parameters": next_practice
            },
            {
                "action_type": "ENABLE_START_PRACTICE_BUTTON",
                "parameters": {"button_id": "start_practice_button_id"}
            }
        ]
        
        practice_title = next_practice.get('title', 'practice exercise')
        practice_description = next_practice.get('description', '')
        practice_suggestion_tts = f"I recommend working on {practice_title} to improve your skills. {practice_description}"
        
        return {
            "selected_practices": selected_practices,
            "next_practice": next_practice,
            "practice_suggestion": {
                "tts": practice_suggestion_tts,
                "title": practice_title,
                "description": practice_description
            },
            "output_content": {
                "response": practice_suggestion_tts,
                "ui_actions": ui_actions
            }
        }
    
    except Exception as e:
        logger.error(f"Error in select_next_practice_or_drill_node: {e}", exc_info=True)
        error_response = {
            "selected_practices": [],
            "next_practice": {},
            "practice_suggestion": {
                "tts": "I'm having trouble selecting an appropriate practice for you right now. Let's continue with our regular session.",
                "title": "Practice Unavailable",
                "description": "Practice selection temporarily unavailable."
            },
            "output_content": {
                "response": "I'm having trouble selecting an appropriate practice for you right now. Let's continue with our regular session.",
                "ui_actions": []
            }
        }
        return error_response
