# graph/modeling_flow.py
from langgraph.graph import StateGraph, END
try:
    from state import AgentGraphState
except ImportError:
    print("Warning: Could not import AgentGraphState from 'state'. Using a placeholder.")
    class AgentGraphState(dict): pass

# Import the actual agent node functions for the modeling flow
from agents import (
    modelling_RAG_document_node,
    modelling_generator_node,
    modelling_output_formatter_node,
)

# Define node names for clarity
NODE_MODELLING_RAG_DOCUMENT = "modelling_RAG_document"
NODE_MODELLING_GENERATOR = "modelling_generator"
NODE_MODELLING_OUTPUT_FORMATTER = "modelling_output_formatter"

def create_modeling_subgraph():
    """
    Creates a LangGraph subgraph for the modeling flow.
    This flow generates a model response based on a student's request, using RAG and a generator.
    """
    workflow = StateGraph(AgentGraphState)

    # Add nodes to the subgraph
    workflow.add_node(NODE_MODELLING_RAG_DOCUMENT, modelling_RAG_document_node)
    workflow.add_node(NODE_MODELLING_GENERATOR, modelling_generator_node)
    workflow.add_node(NODE_MODELLING_OUTPUT_FORMATTER, modelling_output_formatter_node)

    # Set the entry point for the subgraph
    workflow.set_entry_point(NODE_MODELLING_RAG_DOCUMENT)

    # Define the sequential flow
    workflow.add_edge(NODE_MODELLING_RAG_DOCUMENT, NODE_MODELLING_GENERATOR)
    workflow.add_edge(NODE_MODELLING_GENERATOR, NODE_MODELLING_OUTPUT_FORMATTER)
    workflow.add_edge(NODE_MODELLING_OUTPUT_FORMATTER, END) # End of the modeling subgraph

    return workflow.compile()

if __name__ == "__main__":
    print("Attempting to compile the modeling subgraph...")
    modeling_graph_compiled = create_modeling_subgraph()
    print("Modeling subgraph compiled successfully.")
