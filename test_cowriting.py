"""
Test harness for the LangGraph Cowriting System

This module tests the cowriting LangGraph system by creating a mock state
and running it through the cowriting workflow.
"""

import asyncio
import json
import logging
import os
from types import SimpleNamespace
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from state import AgentGraphState
from agents.cowriting_student_data import cowriting_student_data_node
from agents.cowriting_analyzer import cowriting_analyzer_node
from agents.cowriting_retriever import cowriting_retriever_node
from agents.cowriting_planner import cowriting_planner_node
from agents.cowriting_generator import cowriting_generator_node

load_dotenv()

os.environ['COWRITING_TEST_MODE'] = 'true'
logging.info("Set COWRITING_TEST_MODE=true to use smaller dataset")

api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    logging.info(f"Google API key found with length: {len(api_key)}")
else:
    logging.warning("Google API key not found in environment variables")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_cowriting_graph():
    """
    Build the cowriting state graph with all required nodes.
    
    Returns:
        A StateGraph with the cowriting workflow
    """
    builder = StateGraph(AgentGraphState)
    
    # Add nodes
    builder.add_node("student_data", cowriting_student_data_node)
    builder.add_node("analyzer", cowriting_analyzer_node)
    builder.add_node("retriever", cowriting_retriever_node)
    builder.add_node("planner", cowriting_planner_node)
    builder.add_node("generator", cowriting_generator_node)
    
    # Add edges
    builder.add_edge("student_data", "analyzer")
    builder.add_edge("analyzer", "retriever")
    builder.add_edge("retriever", "planner")
    builder.add_edge("planner", "generator")
    builder.add_edge("generator", END)
    
    # Set the entry point
    builder.set_entry_point("student_data")
    
    return builder.compile()

async def test_cowriting_system():
    """
    Test function for running the cowriting flow of the LangGraph system.
    """
    logger.info("Building the cowriting graph...")
    graph = build_cowriting_graph()
    
    test_context = SimpleNamespace(
        task_stage="COWRITING_GENERATION",
        current_written_text="I think technology is very good and have many benefits for society.",
        articulated_thought="I'm not sure if my thesis is strong enough.",
        writing_task_type="Independent Essay",
        writing_section="Crafting the Introduction and Thesis Statement",
        learning_objective="Ensuring a clear and arguable thesis statement",
        comfort_level="Conversational"
    )
    
    test_state = AgentGraphState(
        user_id="test_user_123",
        current_context=test_context,
        user_token="test_token",
        session_id="test_session_123",
        transcript="",
        chat_history=[],
        question_stage="COWRITING",
        student_memory_context={
            "learning_style": "prefers feedback with suggestions",
            "language_background": "native Spanish speaker"
        }
    )
    
    logger.info(f"Initial state keys: {list(test_state.keys())}")
    logger.info(f"Initial context: {test_state.get('current_context', 'None')}")
    
    logger.info("Running the graph with test inputs...")
    result = await graph.ainvoke(test_state)
    
    logger.info("Graph execution completed!")
    print("\n" + "="*50)
    print("COWRITING TEST RESULTS")
    print("="*50)
    
    if "cowriting_output" in result:
        print("\nCowriting Output:")
        print(f"\nText for TTS:\n{result['cowriting_output'].get('text_for_tts', 'No TTS text found')}")
        print(f"\nUI Components:\n{json.dumps(result['cowriting_output'].get('ui_components', []), indent=2)}")
        
        print("\nOutput Content (for formatter node):")
        if "output_content" in result:
            print(f"\nText for TTS:\n{result['output_content'].get('text_for_tts', 'No TTS text found')}")
            print(f"\nUI Actions:\n{json.dumps(result['output_content'].get('ui_actions', []), indent=2)}")
        else:
            print("No output_content found in result")
    else:
        print("\nFull Result:")
        print(json.dumps(result, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x), indent=2))
    
    with open('cowriting_test_output.json', 'w') as f:
        json.dump(result, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x), fp=f, indent=2)
    print(f"\nComplete results saved to 'cowriting_test_output.json'\n")


if __name__ == "__main__":
    asyncio.run(test_cowriting_system())
