#not tested this

import os
import yaml
import json
import enum
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from state import AgentGraphState

logger = logging.getLogger(__name__)

PROMPTS_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'llm_prompts.yaml')

PROMPTS = {}
try:
    with open(PROMPTS_FILE_PATH, 'r') as f:
        loaded_prompts = yaml.safe_load(f)
        if loaded_prompts and 'PROMPTS' in loaded_prompts:
            PROMPTS = loaded_prompts['PROMPTS']
        else:
            logger.error(f"Could not find 'PROMPTS' key in {PROMPTS_FILE_PATH} for scaffolding_provider_node")
except FileNotFoundError:
    logger.error(f"Prompts file not found at {PROMPTS_FILE_PATH} for scaffolding_provider_node")
except yaml.YAMLError as e:
    logger.error(f"Error parsing YAML from {PROMPTS_FILE_PATH} for scaffolding_provider_node: {e}")


class ScaffoldingProviderNode:
    """
    Provides various types of temporary support to students based on their current needs.
    
    Purpose:
    - Provide structural templates (e.g., for essays, speaking responses)
    - Offer sentence starters and hints
    - Break down complex tasks into manageable steps
    - Provide partial solutions for the student to complete
    - Guide students through educational processes
    - Support tasks within the student's Zone of Proximal Development (ZPD)
    """
    
    class ScaffoldingType(enum.Enum):
        """Enumeration of scaffolding types"""
        TEMPLATE = "template"
        SENTENCE_STARTER = "sentence_starter"
        HINT = "hint"
        TASK_BREAKDOWN = "task_breakdown"
        PARTIAL_SOLUTION = "partial_solution"
        PROCESS_GUIDE = "process_guide"
    
    class TaskType(enum.Enum):
        """Enumeration of task types that may need scaffolding"""
        SPEAKING = "speaking"
        WRITING = "writing"
        READING = "reading"
        LISTENING = "listening"
        INTEGRATED = "integrated"
    
    def __init__(self):
        """Initialize the ScaffoldingProviderNode"""
        if 'GOOGLE_API_KEY' in os.environ:
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        else:
            logger.error("GOOGLE_API_KEY not found in environment.")
        
        try:
            self.model = genai.GenerativeModel(
                'gemini-2.5-flash-preview-05-20',
                generation_config=GenerationConfig(
                    temperature=0.4,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048
                )
            )
            logger.info("ScaffoldingProviderNode: LLM model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GenerativeModel: {e}")
            self.model = None
        
        self._instance = self
        
        self.scaffolding_database = self._initialize_scaffolding_database()
        
    def _initialize_scaffolding_database(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Initialize the database of scaffolding templates.
        
        This database contains pre-defined scaffolding elements organized by task type and scaffolding type.
        Each scaffolding element contains content, description, and other metadata.
        
        Returns:
            Dictionary of scaffolding templates organized by task type and scaffolding type
        """
        return {
            "speaking": {
                "template": [
                    {
                        "id": "speaking_template_independent",
                        "title": "Independent Speaking Response Template",
                        "description": "A template for organizing independent speaking responses",
                        "content": {
                            "introduction": "In my opinion...",
                            "reason1": "First, [state your first reason]...",
                            "reason2": "Second, [state your second reason]...",
                            "example": "For example, [provide a specific example]...",
                            "conclusion": "That's why I believe..."
                        },
                        "difficulty": "intermediate"
                    },
                    {
                        "id": "speaking_template_integrated",
                        "title": "Integrated Speaking Response Template",
                        "description": "A template for organizing integrated speaking responses",
                        "content": {
                            "introduction": "The [reading/lecture] discusses...",
                            "key_point1": "The first key point is...",
                            "key_point2": "The second key point is...",
                            "connection": "These points relate to the [reading/lecture] by...",
                            "conclusion": "In conclusion, the main idea is..."
                        },
                        "difficulty": "intermediate"
                    }
                ],
                "sentence_starter": [
                    {
                        "id": "speaking_starter_opinion",
                        "title": "Opinion Statement Starters",
                        "description": "Sentence starters for expressing opinions",
                        "content": [
                            "In my opinion...",
                            "I believe that...",
                            "From my perspective...",
                            "Based on my experience...",
                            "I would argue that..."
                        ],
                        "difficulty": "beginner"
                    },
                    {
                        "id": "speaking_starter_examples",
                        "title": "Example Introduction Starters",
                        "description": "Sentence starters for introducing examples",
                        "content": [
                            "For instance...",
                            "To illustrate this point...",
                            "One clear example is...",
                            "This is evident in...",
                            "We can see this when..."
                        ],
                        "difficulty": "intermediate"
                    }
                ],
                "hint": [
                    {
                        "id": "speaking_hint_organization",
                        "title": "Organization Hints",
                        "description": "Hints for organizing speaking responses",
                        "content": [
                            "Start with a clear position or main idea",
                            "Use transitional phrases between points",
                            "Support your points with specific examples",
                            "Summarize your main points at the end"
                        ],
                        "difficulty": "intermediate"
                    }
                ],
                "task_breakdown": [
                    {
                        "id": "speaking_breakdown_independent",
                        "title": "Independent Speaking Task Breakdown",
                        "description": "Step-by-step breakdown for independent speaking tasks",
                        "content": [
                            "Step 1: Read and understand the question (15 seconds)",
                            "Step 2: Brainstorm your position and 2-3 supporting points (30 seconds)",
                            "Step 3: Organize your response with intro, body, and conclusion (15 seconds)",
                            "Step 4: Deliver your response clearly within the time limit (45 seconds)"
                        ],
                        "difficulty": "intermediate"
                    }
                ]
            },
            
            "writing": {
                "template": [
                    {
                        "id": "writing_template_essay",
                        "title": "TOEFL Essay Template",
                        "description": "A template for organizing a standard TOEFL essay",
                        "content": {
                            "introduction": {
                                "hook": "[Interesting opening statement about the topic]",
                                "background": "[Brief context about the issue]",
                                "thesis": "[Clear statement of your position]"
                            },
                            "body_paragraph_1": {
                                "topic_sentence": "[First main supporting point]",
                                "explanation": "[Explain why this point supports your thesis]",
                                "example": "[Specific example that illustrates this point]",
                                "significance": "[Why this point matters to your argument]"
                            },
                            "body_paragraph_2": {
                                "topic_sentence": "[Second main supporting point]",
                                "explanation": "[Explain why this point supports your thesis]",
                                "example": "[Specific example that illustrates this point]",
                                "significance": "[Why this point matters to your argument]"
                            },
                            "conclusion": {
                                "restatement": "[Restate your thesis in different words]",
                                "summary": "[Briefly summarize your main points]",
                                "final_thought": "[Concluding statement that leaves an impression]"
                            }
                        },
                        "difficulty": "intermediate"
                    }
                ],
                "sentence_starter": [
                    {
                        "id": "writing_starter_introduction",
                        "title": "Essay Introduction Starters",
                        "description": "Sentence starters for essay introductions",
                        "content": [
                            "In today's society, the issue of [topic] has become increasingly important...",
                            "Many people believe that [common perspective], while others argue that...",
                            "The question of whether [restate question] is a complex one that deserves careful consideration...",
                            "Over the past [timeframe], [topic] has emerged as a significant concern..."
                        ],
                        "difficulty": "intermediate"
                    },
                    {
                        "id": "writing_starter_conclusion",
                        "title": "Essay Conclusion Starters",
                        "description": "Sentence starters for essay conclusions",
                        "content": [
                            "In conclusion, it is clear that...",
                            "Based on the arguments presented, I firmly believe that...",
                            "After examining the various aspects of this issue, it is evident that...",
                            "To summarize the main points discussed above..."
                        ],
                        "difficulty": "intermediate"
                    }
                ],
                "partial_solution": [
                    {
                        "id": "writing_partial_integrated",
                        "title": "Partial Integrated Essay",
                        "description": "A partially completed integrated essay for students to extend",
                        "content": {
                            "introduction": "The reading and lecture both discuss [topic]. The author of the reading believes that [main point from reading]. However, the lecturer challenges this view by arguing that [main point from lecture].",
                            "body_paragraph_1": "According to the reading, [first point from reading]. This suggests that [implication]. The lecturer, however, counters this point by stating that [student to complete]...",
                            "body_paragraph_2": "The reading also claims that [second point from reading]. [Student to complete]...",
                            "body_paragraph_3": "[Student to complete]...",
                            "conclusion": "[Student to complete]..."
                        },
                        "difficulty": "advanced"
                    }
                ]
            },
            
            "reading": {
                "process_guide": [
                    {
                        "id": "reading_process_guide",
                        "title": "Strategic Reading Guide",
                        "description": "A step-by-step guide for approaching TOEFL reading passages",
                        "content": [
                            "Step 1: Preview the passage (skim headings, first/last paragraphs, 30 seconds)",
                            "Step 2: Read the questions to know what information to look for (1 minute)",
                            "Step 3: Read the full passage with purpose, noting key ideas (3-4 minutes)",
                            "Step 4: Answer questions by referring back to specific parts of the text (rather than from memory)",
                            "Step 5: For vocabulary questions, look at the context around the word"
                        ],
                        "difficulty": "intermediate"
                    }
                ],
                "hint": [
                    {
                        "id": "reading_hint_inference",
                        "title": "Inference Question Hints",
                        "description": "Hints for approaching inference questions in reading passages",
                        "content": [
                            "Look for clues in the text rather than relying on outside knowledge",
                            "Eliminate answer choices that contradict the passage",
                            "Choose answers that logically follow from the information given",
                            "Pay attention to tone words and author attitude"
                        ],
                        "difficulty": "advanced"
                    }
                ]
            },
            
            "listening": {
                "template": [
                    {
                        "id": "listening_template_notes",
                        "title": "Listening Notes Template",
                        "description": "A template for organizing notes while listening to lectures",
                        "content": {
                            "main_topic": "[Write the main subject of the lecture]",
                            "key_points": "[List 3-5 main points with bullet points]",
                            "examples": "[Note specific examples mentioned]",
                            "connections": "[How ideas connect to each other]",
                            "terminology": "[New terms or concepts introduced]"
                        },
                        "difficulty": "intermediate"
                    }
                ],
                "hint": [
                    {
                        "id": "listening_hint_cues",
                        "title": "Listening Cue Hints",
                        "description": "Hints for recognizing important information in lectures",
                        "content": [
                            "Pay attention to emphasized words or repeated ideas",
                            "Notice when the speaker says phrases like 'The important thing is...' or 'The main reason...'",
                            "Listen for transitional phrases like 'however', 'on the other hand', 'in contrast'",
                            "Note when examples or illustrations are introduced with phrases like 'for instance' or 'to illustrate'"
                        ],
                        "difficulty": "intermediate"
                    }
                ]
            },
            
            "integrated": {
                "task_breakdown": [
                    {
                        "id": "integrated_breakdown_speakingq6",
                        "title": "Integrated Speaking Task 6 Breakdown",
                        "description": "Step-by-step breakdown for the campus situation speaking task",
                        "content": [
                            "Step 1: Listen carefully to identify the problem/situation (who, what, where)",
                            "Step 2: Identify the two solutions/opinions presented",
                            "Step 3: Note the pros and cons of each option",
                            "Step 4: Organize response: describe situation → explain options → state preference with reasons",
                            "Step 5: Deliver response within time limit (60 seconds)"
                        ],
                        "difficulty": "advanced"
                    }
                ],
                "template": [
                    {
                        "id": "integrated_template_speakingq3",
                        "title": "Integrated Speaking Task 3 Template",
                        "description": "Response template for campus situation with reading and listening",
                        "content": {
                            "introduction": "The reading and listening passage discuss [topic]. The announcement/article presents [main point from reading]. In the conversation, the students discuss [reaction to the reading].",
                            "point1": "According to the reading, [first key point]. The [male/female] student responds to this by saying [reaction to first point].",
                            "point2": "The reading also mentions that [second key point]. In response, the student [reaction to second point].",
                            "conclusion": "Overall, the student [agrees/disagrees] with the announcement/article because [summary of main reasons]."
                        },
                        "difficulty": "advanced"
                    }
                ]
            }
        }
    
    def _analyze_student_needs(self, state: AgentGraphState) -> Dict[str, Any]:
        """
        Analyze the agent state to determine the student's current needs and context
        
        Args:
            state: The current agent graph state
            
        Returns:
            Dictionary containing student needs, current task, and other context
        """
        student_needs = {}
        
        if 'current_task' in state:
            student_needs['current_task'] = state['current_task']
            
        if 'task_type' in state:
            student_needs['task_type'] = state['task_type']
        elif 'current_task' in state and 'type' in state['current_task']:
            student_needs['task_type'] = state['current_task']['type']
        
        if 'student_data' in state and state['student_data']:
            student_data = state['student_data']
            if 'skill_levels' in student_data:
                student_needs['skill_levels'] = student_data['skill_levels']
            if 'learning_goals' in student_data:
                student_needs['learning_goals'] = student_data['learning_goals']
            if 'scaffolding_preference' in student_data:
                student_needs['scaffolding_preference'] = student_data['scaffolding_preference']
        
        if 'diagnosis' in state and state['diagnosis']:
            student_needs['diagnosis'] = state['diagnosis']
        
        if 'feedback' in state and state['feedback']:
            student_needs['feedback'] = state['feedback']
        
        if 'scaffolding_request' in state:
            student_needs['explicit_request'] = state['scaffolding_request']
        
        if 'task_type' not in student_needs:
            student_needs['task_type'] = 'speaking'
            
            if 'current_page' in state:
                page = state['current_page']
                if 'speaking' in page.lower():
                    student_needs['task_type'] = 'speaking'
                elif 'writing' in page.lower():
                    student_needs['task_type'] = 'writing'
                elif 'reading' in page.lower():
                    student_needs['task_type'] = 'reading'
                elif 'listening' in page.lower():
                    student_needs['task_type'] = 'listening'
        
        if 'explicit_request' in student_needs:
            request = student_needs['explicit_request'].lower()
            if 'template' in request or 'structure' in request:
                student_needs['scaffolding_type'] = 'template'
            elif 'sentence starter' in request or 'how to begin' in request:
                student_needs['scaffolding_type'] = 'sentence_starter'
            elif 'hint' in request or 'tips' in request:
                student_needs['scaffolding_type'] = 'hint'
            elif 'steps' in request or 'breakdown' in request:
                student_needs['scaffolding_type'] = 'task_breakdown'
            elif 'example' in request or 'start for me' in request:
                student_needs['scaffolding_type'] = 'partial_solution'
            elif 'guide' in request or 'how to' in request:
                student_needs['scaffolding_type'] = 'process_guide'
        
        return student_needs
    
    def _select_scaffolding(self, student_needs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select appropriate scaffolding based on student needs
        
        Args:
            student_needs: Dictionary containing student needs analysis
            
        Returns:
            Selected scaffolding item or None if no suitable scaffolding found
        """
        difficulty_preference = 'intermediate'
        if 'skill_levels' in student_needs:
            skill_levels = student_needs['skill_levels']
            if 'overall' in skill_levels:
                difficulty_preference = skill_levels['overall']
            
        task_type = student_needs.get('task_type', 'speaking')
        scaffolding_type = student_needs.get('scaffolding_type')
        
        if not scaffolding_type and 'current_task' in student_needs:
            current_task = student_needs['current_task']
            
            if current_task.get('status') == 'new' or current_task.get('status') == 'assigned':
                scaffolding_type = 'task_breakdown'
            elif current_task.get('status') == 'in_progress' and student_needs.get('explicit_request'):
                if 'just started' in student_needs.get('explicit_request', '').lower():
                    scaffolding_type = 'template'
                else:
                    scaffolding_type = 'hint'
            elif current_task.get('status') == 'feedback' or current_task.get('status') == 'revision':
                scaffolding_type = 'partial_solution'
        
        if not scaffolding_type:
            default_scaffolding = {
                'speaking': 'template',
                'writing': 'template',
                'reading': 'process_guide',
                'listening': 'hint',
                'integrated': 'task_breakdown'
            }
            scaffolding_type = default_scaffolding.get(task_type, 'hint')
        
        if task_type in self.scaffolding_database and scaffolding_type in self.scaffolding_database[task_type]:
            available_scaffolds = self.scaffolding_database[task_type][scaffolding_type]
            
            matching_scaffolds = [s for s in available_scaffolds if s.get('difficulty') == difficulty_preference]
            
            if not matching_scaffolds and available_scaffolds:
                matching_scaffolds = available_scaffolds
            
            if matching_scaffolds:
                scaffold = matching_scaffolds[0].copy()
                scaffold['task_type'] = task_type
                scaffold['scaffolding_type'] = scaffolding_type
                return scaffold
        
        if scaffolding_type in ['hint', 'process_guide', 'task_breakdown']:
            for t_type in self.scaffolding_database:
                if scaffolding_type in self.scaffolding_database[t_type]:
                    available_scaffolds = self.scaffolding_database[t_type][scaffolding_type]
                    if available_scaffolds:
                        scaffold = available_scaffolds[0].copy()
                        scaffold['task_type'] = t_type
                        scaffold['scaffolding_type'] = scaffolding_type
                        scaffold['note'] = f"This is a general {scaffolding_type.replace('_', ' ')} that might help with your {task_type} task."
                        return scaffold
        
        return None
    
    async def _generate_custom_scaffolding(self, task_type: str, scaffolding_type: str, 
                                         student_context: Dict[str, Any], state: AgentGraphState) -> Dict[str, Any]:
        """
        Generate custom scaffolding using LLM when pre-defined scaffolding is not available
        
        Args:
            task_type: Type of task (speaking, writing, etc.)
            scaffolding_type: Type of scaffolding needed
            student_context: Student context and needs
            state: The current agent graph state
            
        Returns:
            Generated scaffolding item
        """
        if not self.model:
            logger.error("Cannot generate custom scaffolding: LLM model not available")
            return self._create_fallback_scaffolding(task_type, scaffolding_type)
        
        student_info = ""
        if 'student_data' in state and state['student_data']:
            student_data = state['student_data']
            if 'name' in student_data:
                student_info += f"Student name: {student_data['name']}\n"
            if 'skill_levels' in student_data:
                student_info += f"Skill levels: {json.dumps(student_data['skill_levels'])}\n"
            if 'learning_goals' in student_data:
                student_info += f"Learning goals: {student_data['learning_goals']}\n"
        
        task_details = ""
        if 'current_task' in state:
            task = state['current_task']
            if 'prompt' in task:
                task_details += f"Task prompt: {task['prompt']}\n"
            if 'instructions' in task:
                task_details += f"Task instructions: {task['instructions']}\n"
            if 'criteria' in task:
                task_details += f"Evaluation criteria: {task['criteria']}\n"
        
        recent_topics = ""
        if 'chat_history' in state and state['chat_history']:
            recent_topics = "Recent conversation topics: "
            for i, entry in enumerate(state['chat_history'][-3:]):
                if 'content' in entry:
                    recent_topics += entry['content'][:100] + " "
        
        prompt_config = PROMPTS.get('custom_scaffolding_generation')
        if not prompt_config:
            logger.error("Custom scaffolding generation prompt configuration not found")
            return self._create_fallback_scaffolding(task_type, scaffolding_type)
        
        system_prompt_text = prompt_config.get('system', "")
        user_prompt_template = prompt_config.get('user', "")
        
        scaffolding_type_readable = scaffolding_type.replace('_', ' ').title()
        task_type_readable = task_type.title()
        
        user_prompt_text = user_prompt_template.format(
            task_type=task_type_readable,
            scaffolding_type=scaffolding_type_readable,
            student_info=student_info,
            task_details=task_details,
            recent_topics=recent_topics,
        )
        
        full_prompt = f"{system_prompt_text}\n\n{user_prompt_text}"
        
        try:
            response = await self.model.generate_content_async(full_prompt)
            raw_response = response.text
            logger.info(f"Raw LLM response for custom scaffolding: {raw_response[:200]}...")
            
            try:
                scaffold_data = json.loads(raw_response)
                
                required_fields = ['id', 'title', 'description', 'content']
                for field in required_fields:
                    if field not in scaffold_data:
                        scaffold_data[field] = f"Generated {scaffolding_type_readable} for {task_type_readable}"
                        logger.warning(f"Required field {field} missing in generated scaffolding, using default")
                
                scaffold_data['task_type'] = task_type
                scaffold_data['scaffolding_type'] = scaffolding_type
                scaffold_data['difficulty'] = student_context.get('difficulty_preference', 'intermediate')
                scaffold_data['custom_generated'] = True
                scaffold_data['generation_timestamp'] = datetime.now().isoformat()
                
                return scaffold_data
            
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {raw_response[:100]}...")
                return self._create_fallback_scaffolding(task_type, scaffolding_type)
        
        except Exception as e:
            logger.error(f"Error generating custom scaffolding with LLM: {e}")
            return self._create_fallback_scaffolding(task_type, scaffolding_type)
    
    def _create_fallback_scaffolding(self, task_type: str, scaffolding_type: str) -> Dict[str, Any]:
        """
        Create a simple fallback scaffolding when LLM generation fails
        
        Args:
            task_type: Type of task (speaking, writing, etc.)
            scaffolding_type: Type of scaffolding needed
            
        Returns:
            Fallback scaffolding item
        """
        logger.info(f"Creating fallback scaffolding for {scaffolding_type} in {task_type}")
        
        scaffolding_type_readable = scaffolding_type.replace('_', ' ').title()
        task_type_readable = task_type.title()
        
        content = None
        if scaffolding_type == 'template':
            content = {
                "introduction": "Start with a clear introduction of your main point or argument.",
                "body": "Develop your ideas with 2-3 supporting points and specific examples.",
                "conclusion": "Summarize your main points and restate your position."
            }
        elif scaffolding_type == 'sentence_starter':
            content = [
                "In my opinion...",
                "One important aspect to consider is...",
                "For example...",
                "To conclude..."
            ]
        elif scaffolding_type == 'hint':
            content = [
                "Make sure to address all parts of the question",
                "Use specific examples to support your points",
                "Organize your response with a clear structure",
                "Use transitional phrases between ideas"
            ]
        elif scaffolding_type == 'task_breakdown':
            content = [
                "Step 1: Understand the task requirements",
                "Step 2: Organize your main ideas",
                "Step 3: Develop each point with examples",
                "Step 4: Review and refine your response"
            ]
        elif scaffolding_type == 'partial_solution':
            content = {
                "introduction": "Here's how you might begin: [Example introduction provided]",
                "development": "Continue by developing your ideas with specific examples.",
                "conclusion": "Finally, summarize your main points and restate your position."
            }
        elif scaffolding_type == 'process_guide':
            content = [
                "First, carefully analyze what the task is asking you to do",
                "Next, brainstorm relevant ideas and examples",
                "Then, organize these ideas into a coherent structure",
                "Finally, express your thoughts clearly and concisely"
            ]
        else:
            content = ["Focus on addressing the task requirements clearly and thoroughly."]
        
        fallback_scaffolding = {
            "id": f"{task_type}_{scaffolding_type}_fallback",
            "title": f"Basic {scaffolding_type_readable} for {task_type_readable}",
            "description": f"A simple {scaffolding_type_readable} to help with your {task_type_readable} task",
            "content": content,
            "task_type": task_type,
            "scaffolding_type": scaffolding_type,
            "difficulty": "intermediate",
            "fallback": True
        }
        
        return fallback_scaffolding
        
    async def provide_scaffolding(self, state: AgentGraphState) -> Dict[str, Any]:
        """
        Main method to provide appropriate scaffolding based on student needs and task context
        
        Args:
            state: The current agent graph state
            
        Returns:
            Dictionary containing selected scaffolding and metadata
        """
        logger.info("ScaffoldingProviderNode: Analyzing student needs for scaffolding")
        
        student_needs = self._analyze_student_needs(state)
        
        selected_scaffolding = self._select_scaffolding(student_needs)
        
        if not selected_scaffolding:
            task_type = student_needs.get('task_type', 'speaking')
            scaffolding_type = student_needs.get('scaffolding_type', 'hint')
            selected_scaffolding = await self._generate_custom_scaffolding(task_type, scaffolding_type, student_needs, state)
        
        logger.info(f"ScaffoldingProviderNode: Selected {selected_scaffolding['scaffolding_type']} for {selected_scaffolding['task_type']}")
        return selected_scaffolding


_scaffolding_provider = ScaffoldingProviderNode()


async def provide_scaffolding_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Provides various types of temporary support to the student based on their current needs.
    
    This node analyzes the student's current needs, task, diagnosed weaknesses, or explicit requests
    to provide appropriate scaffolding such as templates, sentence starters, hints, task breakdowns,
    partial solutions, or process guides.
    
    Args:
        state: The current agent graph state
        
    Returns:
        Dictionary with selected scaffolding and ui_actions for display
    """
    logger.info("ScaffoldingProviderNode: Providing scaffolding support")
    
    try:
        selected_scaffolding = await _scaffolding_provider.provide_scaffolding(state)
        
        scaffolding_type = selected_scaffolding['scaffolding_type']
        task_type = selected_scaffolding['task_type']
        scaffolding_type_readable = scaffolding_type.replace('_', ' ').title()
        
        ui_actions = [
            {
                "action_type": "DISPLAY_SCAFFOLDING",
                "parameters": selected_scaffolding
            }
        ]
        
        if scaffolding_type == 'template':
            ui_actions.append({
                "action_type": "SHOW_TEMPLATE_OVERLAY",
                "parameters": {
                    "template_id": selected_scaffolding['id'],
                    "content": selected_scaffolding['content']
                }
            })
        elif scaffolding_type == 'task_breakdown':
            ui_actions.append({
                "action_type": "SHOW_STEPS_SIDEBAR",
                "parameters": {
                    "steps": selected_scaffolding['content']
                }
            })
        elif scaffolding_type == 'sentence_starter':
            ui_actions.append({
                "action_type": "SHOW_SENTENCE_STARTER_BUTTONS",
                "parameters": {
                    "starters": selected_scaffolding['content']
                }
            })
        
        scaffolding_title = selected_scaffolding.get('title', f'{scaffolding_type_readable}')
        scaffolding_description = selected_scaffolding.get('description', '')
        
        if scaffolding_type == 'hint':
            tts_message = f"Here are some hints that might help you: {', '.join(selected_scaffolding['content'][:2])}, and more."
        elif scaffolding_type == 'template':
            tts_message = f"I've provided a {task_type} template to help structure your response. {scaffolding_description}"
        elif scaffolding_type == 'task_breakdown':
            tts_message = f"Let me break down this task for you into manageable steps. {scaffolding_description}"
        elif scaffolding_type == 'sentence_starter':
            tts_message = f"Here are some ways you could begin your response. {scaffolding_description}"
        elif scaffolding_type == 'process_guide':
            tts_message = f"Let me guide you through this process. {scaffolding_description}"
        elif scaffolding_type == 'partial_solution':
            tts_message = f"I've started a solution for you. Try to complete it using your own ideas and examples."
        else:
            tts_message = f"Here's some {scaffolding_type_readable} to help with your {task_type} task."
        
        return {
            "selected_scaffolding": selected_scaffolding,
            "scaffolding_suggestion": {
                "tts": tts_message,
                "title": scaffolding_title,
                "description": scaffolding_description
            },
            "output_content": {
                "response": tts_message,
                "ui_actions": ui_actions
            }
        }
    
    except Exception as e:
        logger.error(f"Error in provide_scaffolding_node: {e}", exc_info=True)
        error_response = {
            "selected_scaffolding": {},
            "scaffolding_suggestion": {
                "tts": "I'm having trouble providing appropriate scaffolding right now. Let's continue with our session.",
                "title": "Scaffolding Unavailable",
                "description": "Scaffolding temporarily unavailable."
            },
            "output_content": {
                "response": "I'm having trouble providing appropriate scaffolding right now. Let's continue with our session.",
                "ui_actions": []
            }
        }
        return error_response
