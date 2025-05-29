from typing import Any, Dict, List
import logging
from mem0 import Memory
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env file
mem0_api_key = os.getenv("MEM0_API_KEY")
Google_api_key = os.getenv("GOOGLE_API_KEY")

# Note: mem0ai's Memory class has built-in configuration capabilities.
# For our initial integration, we'll use the default settings.
# When ready to customize, check the mem0ai documentation for the proper configuration format.

class Mem0Memory:
    """A memory class using mem0ai for storing and retrieving user-centric data."""
    def __init__(self, user_id_field: str = "user_id"):
        # Initialize mem0 with default settings
        # For multiple users, mem0 handles data isolation internally based on the `user_id` 
        # you pass to its methods, or you can create separate Memory instances per user if preferred.
        self.mem0_instance = Memory()
        self.user_id_field = user_id_field # Field name to identify user in data
        logger.info(f"Mem0Memory initialized. Using '{user_id_field}' as user identifier.")

    def get_student_data(self, user_id: str) -> Dict[str, Any]:
        """Retrieves all data for a given user_id using mem0."""
        logger.info(f"Mem0Memory: Attempting to get all data for user_id: {user_id}")
        try:
            # mem0's get_all retrieves all memories. We might need to filter by user_id if not implicitly handled
            # or structure data with user_id as a key part of the memory entries.
            # For now, let's assume we store data with a user_id tag or in a user-specific collection.
            # This part will need refinement based on how mem0 structures multi-user data.
            # A common pattern is to pass user_id to methods like add, search, etc.
            all_memories = self.mem0_instance.get_all(user_id=user_id)
            
            # We need to reconstruct the student_data structure (profile, interaction_history)
            # from the retrieved memories.
            # This is a placeholder and will depend on how we store data in mem0.
            student_data = {
                "profile": {}, 
                "interaction_history": []
            }
            
            for mem in all_memories:
                # Example: if memories have a 'type' field (e.g., 'profile' or 'interaction')
                if mem.data.get('type') == 'profile':
                    student_data["profile"].update(mem.data.get('content', {}))
                elif mem.data.get('type') == 'interaction':
                    student_data["interaction_history"].append(mem.data.get('content', {}))
                # If no type, maybe just append to a general list or a default category
                # This logic needs to be robust based on actual storage strategy.

            if not student_data["profile"] and not student_data["interaction_history"]:
                 # Initialize with a default structure if user is new or no data found
                student_data["profile"] = {"name": "New Student (mem0)", "level": "Beginner"}
                logger.info(f"Mem0Memory: Initialized new student data for user_id: {user_id}")

            logger.info(f"Mem0Memory: Retrieved data for {user_id}: {student_data}")
            return student_data
        except Exception as e:
            logger.error(f"Mem0Memory: Error getting data for {user_id}: {e}")
            # Return a default structure on error or re-raise
            return {
                "profile": {"name": "Error Student (mem0)", "level": "Unknown"},
                "interaction_history": []
            }

    def add_interaction_to_history(self, user_id: str, interaction_summary: Dict[str, Any]) -> None:
        """Adds an interaction summary to the user's history using mem0."""
        logger.info(f"Mem0Memory: Attempting to add interaction for user_id: {user_id} with summary: {interaction_summary}")
        try:
            # We can add metadata to distinguish this memory, e.g., type: 'interaction'
            # The `user_id` is passed to associate the memory with the specific user.
            self.mem0_instance.add(
                data=interaction_summary, 
                user_id=user_id, 
                metadata={'type': 'interaction', self.user_id_field: user_id}
            )
            logger.info(f"Mem0Memory: Added interaction for {user_id}.")
        except Exception as e:
            logger.error(f"Mem0Memory: Error adding interaction for {user_id}: {e}")

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
        mem0_memory_instance.add_interaction_to_history(user1, interaction1_user1)
        print(f"\nData for {user1} after 1st interaction: {mem0_memory_instance.get_student_data(user1)}")

        interaction2_user1 = {"transcript": "I need help with mem0.", "diagnosis": "Struggling with mem0 concepts", "feedback": "Let's review mem0 docs."}
        mem0_memory_instance.add_interaction_to_history(user1, interaction2_user1)
        print(f"\nData for {user1} after 2nd interaction: {mem0_memory_instance.get_student_data(user1)}")

        mem0_memory_instance.update_student_profile(user1, {"level": "Intermediate (mem0)", "preferred_topic": "AI Memory"})
        print(f"\nData for {user1} after profile update: {mem0_memory_instance.get_student_data(user1)}")

        print(f"\nInitial data for {user2}: {mem0_memory_instance.get_student_data(user2)}")
        interaction1_user2 = {"transcript": "mem0 is quite interesting!", "diagnosis": "Curious about mem0", "feedback": "Excellent!"}
        mem0_memory_instance.add_interaction_to_history(user2, interaction1_user2)
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









