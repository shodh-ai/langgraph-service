# agents/cowriting_initializer_node.py
import logging
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def cowriting_initializer_node(state: AgentGraphState) -> dict:
    """
    Initializes the state for a co-writing turn.

    It takes the raw student writing from the 'transcript' key and maps it
    to the specific keys needed by the co-writing RAG node. It also
    retrieves any other relevant context.
    """
    logger.info("--- Initializing Co-Writing Flow State ---")

    # The student's latest writing chunk comes from the standard 'transcript' key.
    student_writing = state.get("transcript", "")
    
    # Context can come from the initial payload from the frontend.
    current_context = state.get("current_context", {})
    learning_objective = current_context.get("Learning_Objective_Focus", "General writing assistance")
    
    # Here, you could eventually add a small LLM call to get an
    # 'Immediate_Assessment_of_Input' if needed, but for now, we can pass it as empty.

    return {
        "Student_Written_Input_Chunk": student_writing,
        "Learning_Objective_Focus": learning_objective,
        "Immediate_Assessment_of_Input": "Initial analysis pending.", # Placeholder
        "Student_Articulated_Thought": "" # This would come from a different user action
    }
