# graph/conversation_flow.py

from langgraph.graph import StateGraph, END
from state import AgentGraphState
from agents import conversation_handler_node # Import the node

def create_conversation_subgraph():
    """
    Creates a simple subgraph for the conversation flow.
    This ensures its state output is handled consistently.
    """
    workflow = StateGraph(AgentGraphState)

    workflow.add_node("conversation_entry", conversation_handler_node)
    
    workflow.set_entry_point("conversation_entry")
    workflow.add_edge("conversation_entry", END)
    
    return workflow.compile()