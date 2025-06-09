from .simple_memory_stub import SimpleMemoryStub
from .mem0_memory import Mem0Memory

# Global instance of the Mem0Memory client.
# Files importing 'memory_stub' will now use Mem0Memory.
memory_stub = Mem0Memory()

# You can remove SimpleMemoryStub if it's no longer needed directly elsewhere,
# or keep it if you plan to switch between them for testing.

__all__ = ["memory_stub", "Mem0Memory", "SimpleMemoryStub"]