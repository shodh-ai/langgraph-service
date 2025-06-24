# graph/welcome_flow.py

from langgraph.graph import StateGraph, END
from state import AgentGraphState
from agents import handle_welcome_node # Import the specific node

def create_welcome_subgraph():
    """
    Creates a simple subgraph for the welcome flow.
    This ensures its state output is handled consistently before being passed
    back to the main graph.
    """
    workflow = StateGraph(AgentGraphState)

    workflow.add_node("welcome_entry", handle_welcome_node)
    
    workflow.set_entry_point("welcome_entry")
    workflow.add_edge("welcome_entry", END)
    
    return workflow.compile()
