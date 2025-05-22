from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)

class SimpleMemoryStub:
    """A simple in-memory stub for user-based student data and interaction history."""
    def __init__(self):
        # Storage key is user_id
        self._storage: Dict[str, Dict[str, Any]] = {}
        logger.info("SimpleMemoryStub (user-centric) initialized.")

    def get_student_data(self, user_id: str) -> Dict[str, Any]:
        """Retrieves all data for a given user_id."""
        logger.info(f"Memory: Attempting to get student data for user_id: {user_id}")
        if user_id not in self._storage:
            # Initialize with a default structure if user is new
            self._storage[user_id] = {
                "profile": {"name": "New Student", "level": "Beginner"}, # Example profile data
                "interaction_history": []
            }
            logger.info(f"Memory: Initialized new student data for user_id: {user_id}")
        
        student_data = self._storage[user_id]
        logger.info(f"Memory: Retrieved data for {user_id}: {student_data}")
        return student_data

    def add_interaction_to_history(self, user_id: str, interaction_summary: Dict[str, Any]) -> None:
        """Adds an interaction summary to the user's history."""
        logger.info(f"Memory: Attempting to add interaction for user_id: {user_id} with summary: {interaction_summary}")
        if user_id not in self._storage:
            # Ensure user data is initialized before adding history
            self.get_student_data(user_id) # This will initialize if not present
        
        self._storage[user_id]["interaction_history"].append(interaction_summary)
        logger.info(f"Memory: Added interaction for {user_id}. History length: {len(self._storage[user_id]['interaction_history'])}")
        logger.debug(f"Memory: Full data for {user_id} after update: {self._storage[user_id]}")

    def update_student_profile(self, user_id: str, profile_data: Dict[str, Any]) -> None:
        """Updates or sets profile data for a student."""
        logger.info(f"Memory: Attempting to update profile for user_id: {user_id} with data: {profile_data}")
        if user_id not in self._storage:
            self.get_student_data(user_id) # Initialize if new
        self._storage[user_id]["profile"].update(profile_data)
        logger.info(f"Memory: Updated profile for {user_id}. Current profile: {self._storage[user_id]['profile']}")

    def clear_user_memory(self, user_id: str) -> None:
        """Clears all memory for a specific user."""
        if user_id in self._storage:
            del self._storage[user_id]
            logger.info(f"Memory: Cleared all data for user_id: {user_id}")
        else:
            logger.info(f"Memory: No data found to clear for user_id: {user_id}")

# Example usage (can be removed or kept for testing):
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    memory_stub_instance = SimpleMemoryStub()
    
    user1 = "student_alpha_001"
    user2 = "student_beta_002"

    print(f"\nInitial data for {user1}: {memory_stub_instance.get_student_data(user1)}")
    
    interaction1_user1 = {"transcript": "Hello", "diagnosis": "Good start", "feedback": "Keep going!"}
    memory_stub_instance.add_interaction_to_history(user1, interaction1_user1)
    print(f"\nData for {user1} after 1st interaction: {memory_stub_instance.get_student_data(user1)}")

    interaction2_user1 = {"transcript": "I need help.", "diagnosis": "Struggling", "feedback": "Let's review basics."}
    memory_stub_instance.add_interaction_to_history(user1, interaction2_user1)
    print(f"\nData for {user1} after 2nd interaction: {memory_stub_instance.get_student_data(user1)}")

    memory_stub_instance.update_student_profile(user1, {"level": "Intermediate", "preferred_topic": "Essays"})
    print(f"\nData for {user1} after profile update: {memory_stub_instance.get_student_data(user1)}")

    print(f"\nInitial data for {user2}: {memory_stub_instance.get_student_data(user2)}")
    interaction1_user2 = {"transcript": "This is easy!", "diagnosis": "Confident", "feedback": "Great job!"}
    memory_stub_instance.add_interaction_to_history(user2, interaction1_user2)
    print(f"\nData for {user2} after 1st interaction: {memory_stub_instance.get_student_data(user2)}")

    memory_stub_instance.clear_user_memory(user1)
    print(f"\nData for {user1} after clearing: {memory_stub_instance.get_student_data(user1)}") # Will re-initialize
    print(f"\nData for {user2} (should be unaffected): {memory_stub_instance.get_student_data(user2)}")
