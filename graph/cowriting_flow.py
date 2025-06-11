# graph/cowriting_flow.py
from langgraph.graph import StateGraph, END
try:
    from state import AgentGraphState
except ImportError:
    print("Warning: Could not import AgentGraphState from 'state'. Using a placeholder.")
    class AgentGraphState(dict): pass

# Import the actual agent node functions
from agents import (
    cowriting_student_data_node,
    cowriting_analyzer_node,
    cowriting_retriever_node,
    cowriting_planner_node,
    cowriting_generator_node,
)

# Define node names for clarity
NODE_COWRITING_STUDENT_DATA = "cowriting_student_data"
NODE_COWRITING_ANALYZER = "cowriting_analyzer"
NODE_COWRITING_RETRIEVER = "cowriting_retriever"
NODE_COWRITING_PLANNER = "cowriting_planner"
NODE_COWRITING_GENERATOR = "cowriting_generator"

def create_cowriting_subgraph():
    """
    Creates a LangGraph subgraph for the cowriting flow.
    This flow assists the student with their writing task through a series of analytical and generative steps.
    """
    workflow = StateGraph(AgentGraphState)

    # Add nodes to the subgraph
    workflow.add_node(NODE_COWRITING_STUDENT_DATA, cowriting_student_data_node)
    workflow.add_node(NODE_COWRITING_ANALYZER, cowriting_analyzer_node)
    workflow.add_node(NODE_COWRITING_RETRIEVER, cowriting_retriever_node)
    workflow.add_node(NODE_COWRITING_PLANNER, cowriting_planner_node)
    workflow.add_node(NODE_COWRITING_GENERATOR, cowriting_generator_node)

    # Set the entry point for the subgraph
    workflow.set_entry_point(NODE_COWRITING_STUDENT_DATA)

    # Define the sequential flow
    workflow.add_edge(NODE_COWRITING_STUDENT_DATA, NODE_COWRITING_ANALYZER)
    workflow.add_edge(NODE_COWRITING_ANALYZER, NODE_COWRITING_RETRIEVER)
    workflow.add_edge(NODE_COWRITING_RETRIEVER, NODE_COWRITING_PLANNER)
    workflow.add_edge(NODE_COWRITING_PLANNER, NODE_COWRITING_GENERATOR)
    workflow.add_edge(NODE_COWRITING_GENERATOR, END) # End of the cowriting subgraph

    return workflow.compile()

if __name__ == "__main__":
    print("Attempting to compile the cowriting subgraph...")
    cowriting_graph_compiled = create_cowriting_subgraph()
    print("Cowriting subgraph compiled successfully.")
