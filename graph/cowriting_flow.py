# graph/cowriting_flow.py
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# 1. Import the agent node functions
from agents import (
    cowriting_RAG_document_node,
    cowriting_generator_node,
    cowriting_output_formatter_node,

)

# 2. Define standardized node names
NODE_COWRITING_RAG = "cowriting_rag"
NODE_COWRITING_GENERATOR = "cowriting_generator"
NODE_COWRITING_OUTPUT_FORMATTER = "cowriting_output_formatter"



def create_cowriting_subgraph():
    """
    Creates a LangGraph subgraph for the co-writing flow.
    This flow follows the standard RAG -> Generator -> Formatter architecture.
    """
    workflow = StateGraph(AgentGraphState)

    # 3. Add the nodes to the subgraph
    workflow.add_node(NODE_COWRITING_RAG, cowriting_RAG_document_node)
    workflow.add_node(NODE_COWRITING_GENERATOR, cowriting_generator_node)
    workflow.add_node(NODE_COWRITING_OUTPUT_FORMATTER, cowriting_output_formatter_node)


    # 4. Define the entry point and the sequential flow
    workflow.set_entry_point(NODE_COWRITING_RAG)
    workflow.add_edge(NODE_COWRITING_RAG, NODE_COWRITING_GENERATOR)
    workflow.add_edge(NODE_COWRITING_GENERATOR, NODE_COWRITING_OUTPUT_FORMATTER)
    # The flow-specific formatter is the final step in this subgraph
    workflow.add_edge(NODE_COWRITING_OUTPUT_FORMATTER, END)

    # 5. Compile and return the subgraph
    return workflow.compile()
