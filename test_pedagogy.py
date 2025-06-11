import asyncio
import logging
import json
import os
import pytest
from langgraph.graph import END, StateGraph

from state import AgentGraphState
from agents import pedagogy_generator_node
from graph_builder import NODE_PEDAGOGY_GENERATION

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_pedagogy_graph():
    """
    Build a focused pedagogy state graph with only the required nodes.
    
    Returns:
        A StateGraph with the pedagogy workflow
    """
    builder = StateGraph(AgentGraphState)
    
    # Add the pedagogy generator node
    builder.add_node("pedagogy_generator", pedagogy_generator_node)
    
    # Set the entry point
    builder.set_entry_point("pedagogy_generator")
    
    # Add edge to END
    builder.add_edge("pedagogy_generator", END)
    
    return builder.compile()

@pytest.mark.asyncio
async def test_pedagogy_system():
    """
    Test function for running the pedagogy flow of the LangGraph system.
    """
    logger.info("Building the pedagogy graph...")
    graph = build_pedagogy_graph()
    
    test_state = AgentGraphState(
        user_id="test_user_123",
        current_context={"task_stage": "PEDAGOGY_GENERATION"},
        transcript="I want to improve my English speaking skills for the TOEFL test.",
        user_data={
            "name": "Maria",
            "level": "Intermediate",
            "goal": "Improve TOEFL speaking score",
            "past_tasks": ["Speaking Task 1", "Listening Exercise 3"],
            "learning_history": {
                "completed_tasks": 5,
                "recent_focus": "Speaking fluency"
            }
        }
    )
    
    logger.info(f"Initial state keys: {list(test_state.keys())}")
    logger.info("Running the graph with test inputs...")
    
    result = await graph.ainvoke(test_state)
    
    logger.info("Graph execution completed!")
    print("\n" + "="*50)
    print("PEDAGOGY TEST RESULTS")
    print("="*50)
    
    # Save and display results
    print("\nFull Result:")
    print(json.dumps(result, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x), indent=2))
    
    with open('pedagogy_test_output.json', 'w') as f:
        json.dump(result, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x), fp=f, indent=2)
    print(f"\nComplete results saved to 'pedagogy_test_output.json'\n")

if __name__ == "__main__":
    asyncio.run(test_pedagogy_system())
