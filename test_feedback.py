import asyncio
import logging
import json
import os
import pytest
from langgraph.graph import END, StateGraph

from state import AgentGraphState
from agents import (
    feedback_student_data_node,
    error_generator_node,
    query_document_node,
    RAG_document_node,
    feedback_planner_node,
    feedback_generator_node
)
from graph_builder import (
    NODE_FEEDBACK_STUDENT_DATA,
    NODE_ERROR_GENERATION,
    NODE_QUERY_DOCUMENT,
    NODE_RAG_DOCUMENT,
    NODE_FEEDBACK_PLANNER,
    NODE_FEEDBACK_GENERATOR
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_feedback_graph():
    """
    Build a focused feedback state graph with only the required nodes.
    
    Returns:
        A StateGraph with the feedback workflow
    """
    builder = StateGraph(AgentGraphState)
    
    # Add nodes
    builder.add_node("student_data", feedback_student_data_node)
    builder.add_node("error_generation", error_generator_node)
    builder.add_node("query_document", query_document_node)
    builder.add_node("rag_document", RAG_document_node)
    builder.add_node("planner", feedback_planner_node)
    builder.add_node("generator", feedback_generator_node)
    
    # Add edges
    builder.add_edge("student_data", "error_generation")
    builder.add_edge("error_generation", "query_document")
    builder.add_edge("query_document", "rag_document")
    builder.add_edge("rag_document", "planner")
    builder.add_edge("planner", "generator")
    builder.add_edge("generator", END)
    
    # Set the entry point
    builder.set_entry_point("student_data")
    
    return builder.compile()

@pytest.mark.asyncio
async def test_feedback_system():
    """
    Test function for running the feedback flow of the LangGraph system.
    """
    logger.info("Building the feedback graph...")
    graph = build_feedback_graph()
    
    test_state = AgentGraphState(
        user_id="test_user_123",
        current_context={"task_stage": "FEEDBACK_GENERATION"},
        transcript="My recent TOEFL speaking response was: Technology has improved education because students can access information quickly on the internet. Also, online classes make learning convenient. However, technology can be distracting with social media.",
        student_submission={
            "content": "Technology has improved education because students can access information quickly on the internet. Also, online classes make learning convenient. However, technology can be distracting with social media.",
            "submission_type": "speaking",
            "topic": "Technology's impact on education",
            "task_id": "TOEFL_Speaking_1",
            "timestamp": "2025-06-11T12:00:00Z"
        },
        user_data={
            "name": "Carlos",
            "level": "Intermediate",
            "goal": "Improve TOEFL speaking score",
            "learning_history": {
                "completed_tasks": 7,
                "recent_focus": "Speaking coherence and organization"
            }
        },
        task_context={
            "current_task": "TOEFL Speaking Question 1 Practice",
            "task_description": "Give opinion on whether you agree or disagree with statement: Technology has improved education."
        }
    )
    
    logger.info(f"Initial state keys: {list(test_state.keys())}")
    logger.info("Running the graph with test inputs...")
    
    result = await graph.ainvoke(test_state)
    
    logger.info("Graph execution completed!")
    print("\n" + "="*50)
    print("FEEDBACK TEST RESULTS")
    print("="*50)
    
    # Save and display results
    print("\nFull Result:")
    print(json.dumps(result, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x), indent=2))
    
    with open('feedback_test_output.json', 'w') as f:
        json.dump(result, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x), fp=f, indent=2)
    print(f"\nComplete results saved to 'feedback_test_output.json'\n")

if __name__ == "__main__":
    asyncio.run(test_feedback_system())
