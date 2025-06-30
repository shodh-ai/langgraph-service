# graph/modeling_flow.py
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# 1. Import the agent node functions
from agents import (
    modelling_RAG_document_node,
    modelling_generator,
    modelling_output_formatter,
)

# 2. Define standardized node names
NODE_MODELING_RAG = "modelling_rag"
NODE_MODELING_GENERATOR = "modelling_generator"
NODE_MODELING_OUTPUT_FORMATTER = "modelling_output_formatter"


def create_modeling_subgraph():
    """
    Creates a LangGraph subgraph for the modeling flow.
    This flow follows the standard RAG -> Generator -> Formatter architecture.
    """
    workflow = StateGraph(AgentGraphState)

    # 3. Add the nodes to the subgraph
    workflow.add_node(NODE_MODELING_RAG, modelling_RAG_document_node)
    workflow.add_node(NODE_MODELING_GENERATOR, modelling_generator)
    workflow.add_node(NODE_MODELING_OUTPUT_FORMATTER, modelling_output_formatter)

    # 4. Define the entry point and the sequential flow
    workflow.set_entry_point(NODE_MODELING_RAG)
    workflow.add_edge(NODE_MODELING_RAG, NODE_MODELING_GENERATOR)
    workflow.add_edge(NODE_MODELING_GENERATOR, NODE_MODELING_OUTPUT_FORMATTER)
    # The flow-specific formatter is the final step in this subgraph
    workflow.add_edge(NODE_MODELING_OUTPUT_FORMATTER, END)

    # 5. Compile and return the subgraph
    return workflow.compile()
