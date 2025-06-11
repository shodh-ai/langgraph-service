# graph/pedagogy_flow.py
from langgraph.graph import StateGraph, END
from state import AgentGraphState
from agents import pedagogy_generator_node

# Define node names for clarity
NODE_PEDAGOGY_GENERATOR = "pedagogy_generator"

def create_pedagogy_subgraph():
    """
    Creates a LangGraph subgraph for the pedagogy flow.
    This flow generates the next task suggestion for the student.
    """
    workflow = StateGraph(AgentGraphState)

    workflow.add_node(NODE_PEDAGOGY_GENERATOR, pedagogy_generator_node)

    workflow.set_entry_point(NODE_PEDAGOGY_GENERATOR)

    # The subgraph ends after the generator has run
    workflow.add_edge(NODE_PEDAGOGY_GENERATOR, END)

    return workflow.compile()
