import logging
import pickle
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint as BaseCheckpointer
from mem0 import Memory

logger = logging.getLogger(__name__)

load_dotenv()

class Mem0Memory(BaseCheckpointer):
    """
    A LangGraph checkpointer that stores the entire graph state in Mem0.
    It uses a singleton pattern to ensure the Mem0 client is initialized only once,
    preventing file lock errors on Windows.
    """
    _instance: Optional['Mem0Memory'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'is_initialized') and self.is_initialized:
            return

        with self._lock:
            if hasattr(self, 'is_initialized') and self.is_initialized:
                return

            logger.info("Initializing Mem0Memory singleton checkpointer...")
            super().__init__(serde=pickle) # Use pickle for serialization
            try:
                self.mem0_instance = Memory()
                logger.info("Successfully initialized Mem0 client for checkpointer.")
            except Exception as e:
                logger.error(f"Failed to initialize Mem0 client: {e}", exc_info=True)
                raise
            self.is_initialized = True
            logger.info("Mem0Memory singleton checkpointer initialized.")

    def get(self, config: RunnableConfig) -> Optional[Dict[str, Any]]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        memories = self.mem0_instance.get_all(user_id=thread_id)
        checkpoints = [m for m in memories if m.metadata.get("type") == "langgraph_checkpoint"]

        if not checkpoints:
            return None

        if checkpoint_id:
            target_memory = next((m for m in checkpoints if m.id == checkpoint_id), None)
        else:
            # Return the latest checkpoint if no ID is specified
            target_memory = checkpoints[0]

        if not target_memory:
            return None

        return self.serde.loads(target_memory.data.encode('latin-1'))

    def put(self, config: RunnableConfig, checkpoint: Dict[str, Any]) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        
        # The 'version' is the checkpoint_id we will return
        version = self.get_next_version(None)
        
        # Save to mem0
        created_memory = self.mem0_instance.add(
            data=self.serde.dumps(checkpoint).decode('latin-1'),
            user_id=thread_id,
            metadata={
                "type": "langgraph_checkpoint",
                "version_ts": version
            }
        )
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": created_memory.id,
            }
        }

    def list(self, config: RunnableConfig) -> List[RunnableConfig]:
        thread_id = config["configurable"]["thread_id"]
        memories = self.mem0_instance.get_all(user_id=thread_id)
        checkpoints = [m for m in memories if m.metadata.get("type") == "langgraph_checkpoint"]

        return [
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": mem.id,
                },
                "metadata": {
                    "source": "mem0",
                    "timestamp": mem.metadata.get("version_ts"),
                },
            }
            for mem in checkpoints
        ]

    def get_next_version(self, current_version: Optional[Union[int, str]]) -> Union[int, str]:
        return str(datetime.now(timezone.utc).timestamp())

    async def aget(self, config: RunnableConfig) -> Optional[Dict[str, Any]]:
        return self.get(config)

    async def aput(self, config: RunnableConfig, checkpoint: Dict[str, Any]) -> RunnableConfig:
        return self.put(config, checkpoint)

    async def alist(self, config: RunnableConfig) -> List[RunnableConfig]:
        return self.list(config)
        logger.info(f"Mem0Memory: Attempting to add interaction (summary): {interaction_summary} for user_id: {user_id}")
        try:
            # We can add metadata to distinguish this memory, e.g., type: 'interaction'
            # The `user_id` is passed to associate the memory with the specific user.
            interaction_content_str = json.dumps(interaction_summary)
            formatted_message = [{"role": "user", "content": interaction_content_str}]
            self.mem0_instance.add(
                messages=formatted_message, 
                user_id=user_id, 
                metadata={'type': 'interaction', self.user_id_field: user_id}
            )
            logger.info(f"Mem0Memory: Successfully added interaction for user_id: {user_id}")
        except Exception as e:
            logger.error(f"Mem0Memory: ERROR during self.mem0_instance.add() for user_id: {user_id}. Exception: {e}", exc_info=True)
            raise # Re-raise the exception to allow higher-level handlers to catch it

    def update_student_profile(self, user_id: str, profile_data: Dict[str, Any]) -> None:
        """Updates or sets profile data for a student using mem0."""
        logger.info(f"Mem0Memory: Attempting to update profile for user_id: {user_id} with data: {profile_data}")
        try:
            # For updates, mem0 might require searching for an existing profile memory and then updating it,
            # or it might have a direct update/upsert mechanism if a memory ID is known.
            # A simple approach is to add it as a new memory, and retrieval logic handles the latest profile.
            # Or, search for existing profile memory and use mem0.update(id=memory_id, data=...)
            # For simplicity here, we'll add it. Retrieval logic in get_student_data would need to find the latest.
            self.mem0_instance.add(
                data=profile_data, 
                user_id=user_id, 
                metadata={'type': 'profile', self.user_id_field: user_id}
            )
            logger.info(f"Mem0Memory: Updated/added profile for {user_id}.")
        except Exception as e:
            logger.error(f"Mem0Memory: Error updating profile for {user_id}: {e}")

    def clear_user_memory(self, user_id: str) -> None:
        """Clears all memory for a specific user in mem0."""
        logger.info(f"Mem0Memory: Attempting to clear all data for user_id: {user_id}")
        try:
            # mem0.delete_user_memories(user_id=user_id) or similar is expected.
            # If not available, we might need to get all memories for the user and delete them one by one.
            # Checking mem0 docs, there should be a way to clear by user_id or by tags.
            # Assuming mem0.delete_all(user_id=user_id) or similar exists.
            # Based on mem0 docs, it seems we might need to fetch all memories for the user and delete by ID.
            memories_to_delete = self.mem0_instance.get_all(user_id=user_id)
            for mem in memories_to_delete:
                self.mem0_instance.delete(memory_id=mem.id)
            logger.info(f"Mem0Memory: Cleared all data for user_id: {user_id}")
        except Exception as e:
            logger.error(f"Mem0Memory: Error clearing data for {user_id}: {e}")

    def put(self, config: Dict[str, Any], checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Stores a LangGraph checkpoint in Mem0.

        Args:
            config: The config for the checkpoint, including 'configurable': {'thread_id': ...}.
            checkpoint: The checkpoint data (AgentGraphState dictionary) to store.

        Returns:
            A dictionary with the config for the stored checkpoint, including 'thread_ts' (Mem0 memory ID).
        """
        thread_id = config["configurable"]["thread_id"]
        logger.debug(f"Mem0Memory Checkpointer: Putting checkpoint for thread_id: {thread_id}")
        try:
            meta = {
                'type': 'graph_checkpoint',
                'thread_id': thread_id,
            }
            created_memory_entry = self.mem0_instance.add(
                data=checkpoint,  # Store the whole checkpoint dict
                user_id=thread_id, # Use thread_id as the user_id for checkpointing
                metadata=meta
            )
            logger.info(f"Mem0Memory Checkpointer: Successfully stored checkpoint for thread_id: {thread_id}, mem0_id: {created_memory_entry.id}")
            return {"configurable": {"thread_id": thread_id, "thread_ts": created_memory_entry.id}}
        except Exception as e:
            logger.error(f"Mem0Memory Checkpointer: Error storing checkpoint for thread_id: {thread_id}: {e}", exc_info=True)
            raise

    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieves the latest LangGraph checkpoint from Mem0.

        Args:
            config: The config for the checkpoint, including 'configurable': {'thread_id': ...}.

        Returns:
            The checkpoint data (AgentGraphState dictionary) if found, otherwise None.
        """
        thread_id = config["configurable"]["thread_id"]
        logger.debug(f"Mem0Memory Checkpointer: Getting checkpoint for thread_id: {thread_id}")
        try:
            all_memories = self.mem0_instance.get_all(user_id=thread_id)
            
            checkpoints = []
            for mem in all_memories:
                if isinstance(mem.metadata, dict) and mem.metadata.get('type') == 'graph_checkpoint':
                    if isinstance(mem.data, dict):
                        checkpoints.append(mem) # mem object contains data and created_at
                    else:
                        logger.warning(f"Mem0Memory Checkpointer: Found 'graph_checkpoint' for thread_id {thread_id} but data is not dict. Mem ID: {mem.id}, Data type: {type(mem.data)}")
            
            if not checkpoints:
                logger.info(f"Mem0Memory Checkpointer: No checkpoint found for thread_id: {thread_id}")
                return None

            # Sort by creation time to get the latest (MemEntry has 'created_at')
            checkpoints.sort(key=lambda m: m.created_at, reverse=True)
            
            latest_checkpoint_entry = checkpoints[0]
            logger.info(f"Mem0Memory Checkpointer: Retrieved latest checkpoint for thread_id: {thread_id}, mem0_id: {latest_checkpoint_entry.id}, created_at: {latest_checkpoint_entry.created_at}")
            return latest_checkpoint_entry.data 
        
        except Exception as e:
            logger.error(f"Mem0Memory Checkpointer: Error retrieving checkpoint for thread_id: {thread_id}: {e}", exc_info=True)
            return None

# Example usage (for testing purposes):
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    # This example assumes Qdrant is running locally on default port 6333.
    # If not, mem0 will fall back to a default in-memory setup if Qdrant is not reachable,
    # or you can configure a different vector store.
    try:
        mem0_memory_instance = Mem0Memory()
        user1 = "student_gamma_003"
        user2 = "student_delta_004"

        print(f"\nInitial data for {user1}: {mem0_memory_instance.get_student_data(user1)}")
        
        interaction1_user1 = {"transcript": "Hello from mem0", "diagnosis": "Good start with mem0", "feedback": "Keep exploring mem0!"}
        mem0_memory_instance.add_interaction(user1, interaction1_user1)
        print(f"\nData for {user1} after 1st interaction: {mem0_memory_instance.get_student_data(user1)}")

        interaction2_user1 = {"transcript": "I need help with mem0.", "diagnosis": "Struggling with mem0 concepts", "feedback": "Let's review mem0 docs."}
        mem0_memory_instance.add_interaction(user1, interaction2_user1)
        print(f"\nData for {user1} after 2nd interaction: {mem0_memory_instance.get_student_data(user1)}")

        mem0_memory_instance.update_student_profile(user1, {"level": "Intermediate (mem0)", "preferred_topic": "AI Memory"})
        print(f"\nData for {user1} after profile update: {mem0_memory_instance.get_student_data(user1)}")

        print(f"\nInitial data for {user2}: {mem0_memory_instance.get_student_data(user2)}")
        interaction1_user2 = {"transcript": "mem0 is quite interesting!", "diagnosis": "Curious about mem0", "feedback": "Excellent!"}
        mem0_memory_instance.add_interaction(user2, interaction1_user2)
        print(f"\nData for {user2} after 1st interaction: {mem0_memory_instance.get_student_data(user2)}")

        print(f"\n--- Searching user1 history for 'help' ---")
        # Example of using search (assuming mem0 supports search with user_id context)
        # search_results_user1 = mem0_memory_instance.mem0_instance.search(query="help", user_id=user1)
        # print(f"Search results for 'help' for {user1}: {search_results_user1}")
        # Note: The above search is a direct call to mem0. You might want to wrap this in a method.

        mem0_memory_instance.clear_user_memory(user1)
        print(f"\nData for {user1} after clearing: {mem0_memory_instance.get_student_data(user1)}")
        print(f"\nData for {user2} (should be unaffected): {mem0_memory_instance.get_student_data(user2)}")

    except Exception as e:
        print(f"An error occurred during the mem0_memory.py example: {e}")
        print("Please ensure any external services like Qdrant are running if configured, or check API keys.")









