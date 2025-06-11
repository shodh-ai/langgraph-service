from langgraph.graph import StateGraph, END

try:
    from state import AgentGraphState
except ImportError:
    print("Warning: Could not import AgentGraphState from '..state'. Using a placeholder.")
    class AgentGraphState(dict): pass


# Import the actual agent node functions for the new teaching flow
from agents import (
    teaching_rag_node,
    teaching_generator_node,
)

# Define node names for clarity
NODE_TEACHING_RAG = "teaching_RAG"
NODE_TEACHING_GENERATOR = "teaching_generator"

# Import the actual agent node functions for the teaching flow
# The nodes deliver_lesson_step_node and process_student_qa_on_lesson_node
# are not implemented. Using placeholders to allow the graph to build.

def create_teaching_subgraph():
    """
    Creates a LangGraph subgraph for the LLM-based teaching module.

    This subgraph defines a flow for generating and delivering dynamic lesson content.
    """
    workflow = StateGraph(AgentGraphState)

    # Add nodes to the subgraph
    workflow.add_node(NODE_TEACHING_RAG, teaching_rag_node)
    workflow.add_node(NODE_TEACHING_GENERATOR, teaching_generator_node)

    # Set the entry point for the subgraph
    workflow.set_entry_point(NODE_TEACHING_RAG)

    # Define the sequential flow
    workflow.add_edge(NODE_TEACHING_RAG, NODE_TEACHING_GENERATOR)
    workflow.add_edge(NODE_TEACHING_GENERATOR, END) # End of the teaching subgraph

    # Compile the subgraph definition into a runnable graph and return it.
    return workflow.compile()

