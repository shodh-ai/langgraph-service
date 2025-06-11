import asyncio
import logging
import json
import os
from types import SimpleNamespace
from graph_builder import (
    NODE_SCAFFOLDING_STUDENT_DATA,
    NODE_STRUGGLE_ANALYZER,
    NODE_SCAFFOLDING_RETRIEVER,
    NODE_SCAFFOLDING_PLANNER,
    NODE_SCAFFOLDING_GENERATOR
)
from agents import (
    scaffolding_student_data_node,
    struggle_analyzer_node,
    scaffolding_retriever_node,
    scaffolding_planner_node,
    scaffolding_generator_node
)
from langgraph.graph import END, StateGraph
from state import AgentGraphState
from dotenv import load_dotenv

load_dotenv()

os.environ['SCAFFOLDING_TEST_MODE'] = 'true'
logging.info("Set SCAFFOLDING_TEST_MODE=true to use smaller dataset")

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


import pytest

def build_scaffolding_graph():
    """
    Build a focused scaffolding state graph with only the required nodes.
    
    Returns:
        A StateGraph with the scaffolding workflow
    """
    builder = StateGraph(AgentGraphState)
    
    # Add nodes
    builder.add_node("student_data", scaffolding_student_data_node)
    builder.add_node("analyzer", struggle_analyzer_node)
    builder.add_node("retriever", scaffolding_retriever_node)
    builder.add_node("planner", scaffolding_planner_node)
    builder.add_node("generator", scaffolding_generator_node)
    
    # Add edges
    builder.add_edge("student_data", "analyzer")
    builder.add_edge("analyzer", "retriever")
    builder.add_edge("retriever", "planner")
    builder.add_edge("planner", "generator")
    builder.add_edge("generator", END)
    
    # Set the entry point
    builder.set_entry_point("student_data")
    
    return builder.compile()

@pytest.mark.asyncio
async def test_scaffolding_system():
    """
    Test function for running the scaffolding flow of the LangGraph system.
    """
    logger.info("Building the graph...")
    graph = build_scaffolding_graph()
    
    # Use a dictionary instead of SimpleNamespace for serialization compatibility
    test_context = {
        "task_stage": "SCAFFOLDING_GENERATION"
    }
    
    test_state = AgentGraphState(
        user_id="test_user_123",
        current_context=test_context,
        transcript="I'm having trouble organizing my TOEFL speaking response. I know what I want to say but I keep rambling and going off-topic.",
        primary_struggle="Difficulty organizing thoughts in a structured response",
        secondary_struggles=["Going off-topic during response", "Not maintaining clear structure"],
        learning_objective_id="S_Q1_Structure",
        user_data={
            "name": "Harshit",
            "level": "Beginner",
            "goal": "Improve TOEFL speaking score",
            "confidence": "Low confidence in speaking under time constraints",
            "attitude": "Eager but anxious about performance"
        },
        task_context={
            "current_task": "TOEFL Speaking Question 1 Practice",
            "task_description": "Give opinion on whether you agree or disagree with statement: Technology has improved education."
        }
    )
    
    logger.info(f"Initial state keys: {list(test_state.keys())}")
    logger.info(f"Initial primary struggle: {test_state.get('primary_struggle', 'None')}")
    logger.info(f"Initial learning objective: {test_state.get('learning_objective_id', 'None')}")
    
    
    logger.info("Running the graph with test inputs...")
    # Simple graph doesn't need checkpointer configuration
    result = await graph.ainvoke(test_state)
    
    logger.info("Graph execution completed!")
    print("\n" + "="*50)
    print("SCAFFOLDING TEST RESULTS")
    print("="*50)
    
    if "scaffolding_output" in result:
        print("\nScaffolding Output:")
        print(f"\nText for TTS:\n{result['scaffolding_output'].get('text_for_tts', 'No TTS text found')}")
        print(f"\nUI Components:\n{json.dumps(result['scaffolding_output'].get('ui_components', []), indent=2)}")
        
        print("\nOutput Content (for formatter node):")
        if "output_content" in result:
            print(f"\nText for TTS:\n{result['output_content'].get('text_for_tts', 'No TTS text found')}")
            print(f"\nUI Actions:\n{json.dumps(result['output_content'].get('ui_actions', []), indent=2)}")
        else:
            print("No output_content found in result")
    else:
        print("\nFull Result:")
        print(json.dumps(result, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x), indent=2))
        
    with open('scaffolding_test_output.json', 'w') as f:
        json.dump(result, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x), fp=f, indent=2)
    print(f"\nComplete results saved to 'scaffolding_test_output.json'\n")


if __name__ == "__main__":
    asyncio.run(test_scaffolding_system())
