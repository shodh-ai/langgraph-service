import asyncio
import logging
import json
import os
import pytest
from langgraph.graph import END, StateGraph

from state import AgentGraphState
from agents import (
    modelling_query_document_node,
    modelling_RAG_document_node,
    modelling_generator_node,
    modelling_output_formatter_node
)
from graph_builder import (
    NODE_MODELLING_QUERY_DOCUMENT,
    NODE_MODELLING_RAG_DOCUMENT,
    NODE_MODELLING_GENERATOR, 
    NODE_MODELLING_OUTPUT_FORMATTER
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_modeling_graph():
    """
    Build a focused modeling state graph with only the required nodes.
    
    Returns:
        A StateGraph with the modeling workflow
    """
    builder = StateGraph(AgentGraphState)
    
    # Add nodes
    builder.add_node("query_document", modelling_query_document_node)
    builder.add_node("rag_document", modelling_RAG_document_node)
    builder.add_node("generator", modelling_generator_node)
    builder.add_node("output_formatter", modelling_output_formatter_node)
    
    # Add edges
    builder.add_edge("query_document", "rag_document")
    builder.add_edge("rag_document", "generator")
    builder.add_edge("generator", "output_formatter")
    builder.add_edge("output_formatter", END)
    
    # Set the entry point
    builder.set_entry_point("query_document")
    
    return builder.compile()

@pytest.mark.asyncio
async def test_modeling_system():
    """
    Test function for running the modeling flow of the LangGraph system.
    """
    logger.info("Building the modeling graph...")
    graph = build_modeling_graph()
    
    test_state = AgentGraphState(
        user_id="test_user_123",
        current_context={"task_stage": "MODELING_GENERATION"},
        transcript="Can you explain how to organize the introduction section of a TOEFL independent essay?",
        user_data={
            "name": "Lee",
            "level": "Advanced",
            "goal": "Get a high TOEFL writing score",
            "learning_history": {
                "completed_tasks": 8,
                "recent_focus": "Essay structure"
            }
        },
        task_context={
            "current_task": "TOEFL Independent Essay Practice",
            "task_description": "Write an essay on whether students should be required to attend classes."
        }
    )
    
    logger.info(f"Initial state keys: {list(test_state.keys())}")
    logger.info("Running the graph with test inputs...")
    
    result = await graph.ainvoke(test_state)
    
    logger.info("Graph execution completed!")
    print("\n" + "="*50)
    print("MODELING TEST RESULTS")
    print("="*50)
    
    # Save and display results
    print("\nFull Result:")
    print(json.dumps(result, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x), indent=2))
    
    with open('modeling_test_output.json', 'w') as f:
        json.dump(result, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x), fp=f, indent=2)
    print(f"\nComplete results saved to 'modeling_test_output.json'\n")

if __name__ == "__main__":
    asyncio.run(test_modeling_system())
