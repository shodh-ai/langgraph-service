from .mem0_memory import StudentProfileMemory, Mem0Checkpointer
import logging

# Global instance of the Mem0Memory client.
# It's initialized to None to prevent circular import issues.
memory_stub = None

logger = logging.getLogger(__name__)

def initialize_memory():
    """
    Initializes the global memory_stub instance.
    This function should be called once at application startup.
    """
    global memory_stub
    logger.info("--- [initialize_memory] START ---")
    logger.info(f"--- [initialize_memory] memory_stub before init: {type(memory_stub)} ---")
    
    try:
        instance = StudentProfileMemory()
        logger.info(f"--- [initialize_memory] Successfully created StudentProfileMemory instance. Type: {type(instance)} ---")
        memory_stub = instance
    except Exception as e:
        logger.error(f"--- [initialize_memory] FAILED to create instance: {e} ---", exc_info=True)
        memory_stub = None 
        raise

    logger.info(f"--- [initialize_memory] memory_stub after init: {type(memory_stub)} ---")
    logger.info("--- [initialize_memory] END ---")

__all__ = ["memory_stub", "StudentProfileMemory", "Mem0Checkpointer", "initialize_memory"]