import json
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)


async def struggle_analyzer_node(state: AgentGraphState) -> dict:
    """
    Analyzes the student's input to identify specific learning struggles.
    
    This node uses AI to analyze user responses and determine the specific
    challenges the student is facing, which will guide the scaffolding strategy.
    """
    logger.info(
        f"StruggleAnalyzerNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    transcript = state.get("transcript", "")
    user_data = state.get("user_data", {})
    task_context = state.get("task_context", {})
    
    if not transcript:
        logger.warning("No transcript provided for struggle analysis")
        return {"struggle_points": ["No specific struggles identified due to missing input"]}

    prompt = f"""
    You are a pedagogical analyst for an AI Tutor system.
    
    Student Profile: {user_data}
    Current Task: {task_context.get('current_task', 'Unknown task')}
    Student's Input: '{transcript}'
    
    Analyze the student's input and identify specific struggle points related to language learning.
    
    Consider the following categories of struggles:
    - Organization/Structure issues
    - Vocabulary limitations
    - Grammar errors
    - Fluency challenges
    - Comprehension difficulties
    - Task understanding issues
    - Confidence/anxiety factors
    
    Return JSON: {{
        "primary_struggle": "brief description of main struggle point",
        "secondary_struggles": ["list", "of", "other", "struggles"],
        "learning_objective_id": "ID of the most relevant learning objective to address"
    }}
    
    Learning objective IDs to choose from:
    - "S_Q1_Fluency": Speaking Fluently for TOEFL tasks
    - "S_Q1_Structure": Structuring a Response for TOEFL Speaking
    - "W_Ind_Thesis": Writing a Clear Thesis Statement
    - "W_Ind_BodyPara_PEE": Developing a Body Paragraph
    - "G_SVA": Subject-Verb Agreement
    - "G_PastTense": Using Past Tense correctly
    - "V_AcademicWords": Using Academic Vocabulary
    """

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY environment variable is not set - using mock analysis")
        
        if "organizing" in transcript.lower() or "structure" in transcript.lower() or "rambling" in transcript.lower():
            mock_analysis = {
                "primary_struggle": "Difficulty organizing thoughts in a structured response",
                "secondary_struggles": ["Going off-topic during response", "Not maintaining clear structure"],
                "learning_objective_id": "S_Q1_Structure"
            }
        elif "vocabulary" in transcript.lower() or "words" in transcript.lower():
            mock_analysis = {
                "primary_struggle": "Limited vocabulary range",
                "secondary_struggles": ["Repetition of same words", "Lack of advanced expressions"],
                "learning_objective_id": "S_Q1_Vocabulary"
            }
        else:
            mock_analysis = {
                "primary_struggle": "Difficulty organizing thoughts in a structured response",
                "secondary_struggles": ["Going off-topic during response"],
                "learning_objective_id": "S_Q1_Structure"
            }
            
        logger.info(f"Mock analysis determined: {mock_analysis['primary_struggle']}")
        
        return {
            "primary_struggle": mock_analysis["primary_struggle"],
            "secondary_struggles": mock_analysis["secondary_struggles"],
            "learning_objective_id": mock_analysis["learning_objective_id"]
        }

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.5-flash-preview-05-20",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = model.generate_content(prompt)
        response_json = json.loads(response.text)
        
        primary_struggle = response_json.get("primary_struggle", "Unknown struggle")
        secondary_struggles = response_json.get("secondary_struggles", [])
        learning_objective_id = response_json.get("learning_objective_id", "Unknown")
        
        logger.info(f"Identified primary struggle: {primary_struggle}")
        logger.info(f"Relevant learning objective: {learning_objective_id}")
        
        new_state = {key: value for key, value in state.items()}
        
        new_state["primary_struggle"] = primary_struggle
        new_state["secondary_struggles"] = secondary_struggles
        new_state["learning_objective_id"] = learning_objective_id
        
        logger.info(f"Struggle analyzer returning fields: primary_struggle, secondary_struggles, learning_objective_id")
        logger.info(f"Returning state with keys: {list(new_state.keys())}")
        
        return new_state
    except Exception as e:
        logger.error(f"Error in struggle analysis: {e}")
        new_state = {key: value for key, value in state.items()}
        
        new_state["primary_struggle"] = "Error analyzing struggles"
        new_state["secondary_struggles"] = []
        new_state["learning_objective_id"] = "S_Q1_Structure"
        
        logger.info("Struggle analyzer returning fallback fields due to error")
        logger.info(f"Returning state with keys: {list(new_state.keys())}")
        
        return new_state
