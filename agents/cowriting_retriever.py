import logging
import os
import copy
from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path
import numpy as np

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings_available = True
except ImportError:
    embeddings_available = False

logger = logging.getLogger(__name__)

SAMPLE_COWRITING_STRATEGIES = [
    {
        "persona_name": "The Encouraging Nurturer",
        "learning_objective_focus": "Developing well-supported body paragraphs with topic sentences",
        "writing_task_context_section": "Developing the First Body Paragraph (Topic Sentence)",
        "decision_to_intervene": True,
        "intervention_type": "SuggestRephrasing",
        "ai_spoken_or_suggested_text": "I notice you're working on your topic sentence. Your idea is clear, but it could be even stronger if you make the central claim more specific. Would you like to try making it more focused?",
        "rationale_for_intervention_style": "Building confidence while offering constructive guidance on a specific skill",
        "ui_action_hints": [{"action_type_suggestion": "HIGHLIGHT_TEXT_RANGE"}]
    },
    {
        "persona_name": "The Structuralist",
        "learning_objective_focus": "Ensuring a clear and arguable thesis statement",
        "writing_task_context_section": "Crafting the Introduction and Thesis Statement",
        "decision_to_intervene": True,
        "intervention_type": "CorrectGrammar",
        "ai_spoken_or_suggested_text": "I notice your thesis statement contains a run-on sentence. Let's break this into two clear statements to strengthen your argument.",
        "rationale_for_intervention_style": "Direct focus on structural clarity in academic writing",
        "ui_action_hints": [{"action_type_suggestion": "SHOW_INLINE_SUGGESTION"}]
    },
    {
        "persona_name": "The Interactive Explorer",
        "learning_objective_focus": "Using precise and academic vocabulary",
        "writing_task_context_section": "Developing the First Body Paragraph (Evidence/Examples)",
        "decision_to_intervene": True,
        "intervention_type": "OfferVocab",
        "ai_spoken_or_suggested_text": "I see you used 'good' here. Since this is academic writing, what other words might convey your meaning more precisely? Perhaps 'beneficial', 'advantageous', or 'favorable'?",
        "rationale_for_intervention_style": "Encouraging critical thinking about word choice through options",
        "ui_action_hints": [{"action_type_suggestion": "DISPLAY_TOOLTIP"}]
    }
]

def cowriting_retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Entering cowriting_retriever_node")
    
    state_copy = copy.deepcopy(state)
    
    try:
        writing_task_context = state_copy.get("writing_task_context", {})
        task_type = writing_task_context.get("task_type", "")
        section = writing_task_context.get("section", "")
        student_affective_state = state_copy.get("student_affective_state", "")
        cowriting_lo_focus = state_copy.get("cowriting_lo_focus", "")
        student_comfort_level = state_copy.get("student_comfort_level", "")
        
        test_mode = os.getenv("COWRITING_TEST_MODE", "false").lower() == "true"
        logger.debug(f"COWRITING_TEST_MODE: {test_mode}")
        
        base_path = Path(__file__).parent.parent
        csv_file = base_path / "data" / "cowriting_cta_data.csv"
        
        strategies = []
        
        if csv_file.exists():
            logger.debug(f"Loading cowriting strategies from {csv_file}")
            try:
                df = pd.read_csv(csv_file)
                logger.debug(f"Loaded {len(df)} cowriting strategies")
                
                if task_type and section:
                    search_term = f"{section}"
                    task_section_filter = df['Writing_Task_Context_Section'].str.contains(search_term, case=False, na=False)
                    filtered_df = df[task_section_filter]
                    
                    if len(filtered_df) >= 3:
                        logger.debug(f"Found {len(filtered_df)} strategies matching section: {section}")
                        strategies = filtered_df.to_dict('records')
                    else:
                        broader_filter = df['Writing_Task_Context_Section'].notna()
                        filtered_df = df[broader_filter].sample(min(5, len(df)))
                        logger.debug(f"Found {len(filtered_df)} strategies using broader match")
                        strategies = filtered_df.to_dict('records')
                
                if not strategies and cowriting_lo_focus:
                    lo_filter = df['Learning_Objective_Focus'].str.contains(cowriting_lo_focus, case=False, na=False)
                    filtered_df = df[lo_filter]
                    logger.debug(f"Found {len(filtered_df)} strategies matching learning objective: {cowriting_lo_focus}")
                    strategies = filtered_df.to_dict('records')
                
                if not strategies and embeddings_available and len(df) > 0:
                    logger.debug("Using semantic search to find relevant strategies")
                    try:
                        
                        strategies = df.sample(min(3, len(df))).to_dict('records')
                        logger.debug(f"Selected {len(strategies)} strategies via semantic search simulation")
                    except Exception as e:
                        logger.error(f"Error during semantic search: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error reading cowriting strategies CSV: {str(e)}")
        
        if not strategies:
            logger.warning("Using fallback sample cowriting strategies")
            strategies = SAMPLE_COWRITING_STRATEGIES
        
        strategies = strategies[:5]
        
        state_copy["cowriting_strategies"] = strategies
        logger.info(f"Retrieved {len(strategies)} cowriting strategies")
        
        return state_copy
        
    except Exception as e:
        logger.error(f"Unexpected error in cowriting_retriever_node: {str(e)}")
        state_copy["error"] = f"Cowriting retriever error: {str(e)}"
        state_copy["cowriting_strategies"] = SAMPLE_COWRITING_STRATEGIES
        return state_copy
