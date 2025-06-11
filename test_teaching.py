import asyncio
import logging
import json
import os
import pytest
from langgraph.graph import END, StateGraph

from state import AgentGraphState
from models import InteractionRequestContext
from agents import (
    teaching_rag_node,
    teaching_generator_node
)
from graph_builder import (
    NODE_TEACHING_RAG,
    NODE_TEACHING_DELIVERY,
    NODE_TEACHING_GENERATOR
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_teaching_graph():
    """
    Build a focused teaching state graph with only the required nodes.
    
    Returns:
        A StateGraph with the teaching workflow
    """
    builder = StateGraph(AgentGraphState)
    
    # Add nodes
    builder.add_node("rag", teaching_rag_node)
    builder.add_node("generator", teaching_generator_node)
    
    # Add edges
    builder.add_edge("rag", "generator")
    builder.add_edge("generator", END)
    
    # Set the entry point
    builder.set_entry_point("rag")
    
    return builder.compile()

@pytest.mark.asyncio
async def test_teaching_system():
    """
    Test function for running the teaching flow of the LangGraph system.
    """
    logger.info("Building the teaching graph...")
    graph = build_teaching_graph()
    
    user_data_for_test = {
        "name": "Sofia",
        "level": "Intermediate",
        "goal": "Learn essay structure for TOEFL writing",
        "learning_history": {
            "completed_tasks": 4,
            "recent_focus": "Essay planning"
        }
    }

    context = InteractionRequestContext(
        user_id="test_user_123",
        task_stage="TEACHING",
        learning_objective_id="W_Structure_Independent",
        teacher_persona="Nurturer",  # Provide a default persona for the test
        student_proficiency_level=user_data_for_test.get("level")
    )

    test_state = AgentGraphState(
        user_id="test_user_123",
        current_context=context,
        transcript="I need help understanding how to structure a TOEFL independent essay.",
        learning_objective_id="W_Structure_Independent",
        user_data=user_data_for_test,
        task_context={
            "current_task": "TOEFL Independent Essay Structure",
            "task_description": "Learn how to structure a TOEFL independent essay with proper introduction, body paragraphs, and conclusion."
        }
    )
    
    logger.info(f"Initial state keys: {list(test_state.keys())}")
    logger.info("Running the graph with test inputs...")
    
    result = await graph.ainvoke(test_state)
    
    logger.info("Graph execution completed!")
    print("\n" + "="*50)
    print("TEACHING TEST RESULTS")
    print("="*50)
    
    # Save and display results
    print("\nFull Result:")
    print(json.dumps(result, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x), indent=2))
    
    with open('teaching_test_output.json', 'w') as f:
        json.dump(result, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x), fp=f, indent=2)
    print(f"\nComplete results saved to 'teaching_test_output.json'\n")

if __name__ == "__main__":
    asyncio.run(test_teaching_system())
