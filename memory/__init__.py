from .simple_memory_stub import SimpleMemoryStub
from .mem0_memory import Mem0Memory

# Global instance of the simple memory stub
memory_stub = SimpleMemoryStub()

# Global instance of Mem0Memory
mem0_memory = Mem0Memory()

__all__ = ["memory_stub", "SimpleMemoryStub", "Mem0Memory", "mem0_memory"]
