from .simple_memory_stub import SimpleMemoryStub
# Temporarily commenting out Mem0Memory to get the server running
# from .mem0_memory import Mem0Memory

# Global instance of the simple memory stub
memory_stub = SimpleMemoryStub()

# Global instance of Mem0Memory (temporarily commented out)
# mem0_memory = Mem0Memory()

__all__ = ["memory_stub", "SimpleMemoryStub"]  # Temporarily removed Mem0Memory