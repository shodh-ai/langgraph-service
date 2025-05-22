from .simple_memory_stub import SimpleMemoryStub

# Global instance of the memory stub, as imported by agent nodes
memory_stub = SimpleMemoryStub()

__all__ = ["memory_stub", "SimpleMemoryStub"]
