# graph/teaching_flow.py
from langgraph.graph import StateGraph, END
from state import AgentGraphState

# 1. Import the agent node functions
from agents import (
    teaching_RAG_document_node,
    teaching_generator_node,
    teaching_output_formatter_node,

)

# 2. Define standardized node names
NODE_TEACHING_RAG = "teaching_RAG"
NODE_TEACHING_GENERATOR = "teaching_generator"
NODE_TEACHING_OUTPUT_FORMATTER = "teaching_output_formatter"



def create_teaching_subgraph():
    """
    Creates a LangGraph subgraph for the LLM-based teaching module.
    This flow follows the standard RAG -> Generator -> Formatter architecture.
    """
    workflow = StateGraph(AgentGraphState)

    # 3. Add the nodes to the subgraph
    workflow.add_node(NODE_TEACHING_RAG, teaching_RAG_document_node)
    workflow.add_node(NODE_TEACHING_GENERATOR, teaching_generator_node)
    workflow.add_node(NODE_TEACHING_OUTPUT_FORMATTER, teaching_output_formatter_node)


    # 4. Define the entry point and the sequential flow
    workflow.set_entry_point(NODE_TEACHING_RAG)
    workflow.add_edge(NODE_TEACHING_RAG, NODE_TEACHING_GENERATOR)
    workflow.add_edge(NODE_TEACHING_GENERATOR, NODE_TEACHING_OUTPUT_FORMATTER)
    # The flow-specific formatter is the final step in this subgraph
    workflow.add_edge(NODE_TEACHING_OUTPUT_FORMATTER, END)

    # 5. Compile and return the subgraph
    return workflow.compile()

