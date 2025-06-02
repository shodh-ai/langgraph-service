from typing import Any, Dict, List
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class SimpleMemory:
    """A simple memory class that mimics the interface of Mem0Memory but uses a dictionary for storage."""
    def __init__(self):
        self.data_store = {}
        self.storage_file = os.path.join(os.path.dirname(__file__), "simple_memory_store.json")
        self._load_data()
        logger.info("SimpleMemory initialized.")

    def _load_data(self):
        """Load data from file if it exists."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    self.data_store = json.load(f)
                logger.info(f"Loaded data from {self.storage_file}")
            except Exception as e:
                logger.error(f"Error loading data from {self.storage_file}: {e}")
                self.data_store = {}
        else:
            self.data_store = {}

    def _save_data(self):
        """Save data to file."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.data_store, f, indent=2)
            logger.info(f"Saved data to {self.storage_file}")
        except Exception as e:
            logger.error(f"Error saving data to {self.storage_file}: {e}")

    def get_student_data(self, user_id: str) -> Dict[str, Any]:
        """Retrieves all data for a given user_id."""
        logger.info(f"SimpleMemory: Getting data for user_id: {user_id}")
        if user_id not in self.data_store:
            # Initialize with a default structure if user is new
            self.data_store[user_id] = {
                "profile": {"name": "New Student", "level": "Beginner", "skills": {}},
                "interaction_history": []
            }
            self._save_data()
        
        return self.data_store[user_id]

    def add_interaction_to_history(self, user_id: str, interaction_summary: Dict[str, Any]) -> None:
        """Adds an interaction summary to the user's history."""
        logger.info(f"SimpleMemory: Adding interaction for user_id: {user_id}")
        if user_id not in self.data_store:
            self.get_student_data(user_id)  # This will initialize the user data
        
        # Add timestamp if not provided
        if interaction_summary.get("timestamp") == "AUTO_TIMESTAMP":
            interaction_summary["timestamp"] = datetime.now().isoformat()
        
        self.data_store[user_id]["interaction_history"].append(interaction_summary)
        self._save_data()

    def update_student_profile(self, user_id: str, profile_data: Dict[str, Any]) -> None:
        """Updates or sets profile data for a student."""
        logger.info(f"SimpleMemory: Updating profile for user_id: {user_id}")
        if user_id not in self.data_store:
            self.get_student_data(user_id)  # This will initialize the user data
        
        self.data_store[user_id]["profile"].update(profile_data)
        self._save_data()

    def clear_user_memory(self, user_id: str) -> None:
        """Clears all memory for a specific user."""
        logger.info(f"SimpleMemory: Clearing data for user_id: {user_id}")
        if user_id in self.data_store:
            del self.data_store[user_id]
            self._save_data()

# Create a singleton instance
simple_memory = SimpleMemory()

# Example usage (for testing purposes):
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    try:
        user1 = "student_test_001"
        user2 = "student_test_002"

        print(f"\nInitial data for {user1}: {simple_memory.get_student_data(user1)}")
        
        interaction1_user1 = {"transcript": "Hello", "diagnosis": "Good start", "feedback": "Keep going!"}
        simple_memory.add_interaction_to_history(user1, interaction1_user1)
        print(f"\nData for {user1} after 1st interaction: {simple_memory.get_student_data(user1)}")

        interaction2_user1 = {"transcript": "I need help.", "diagnosis": "Struggling with concepts", "feedback": "Let's review."}
        simple_memory.add_interaction_to_history(user1, interaction2_user1)
        print(f"\nData for {user1} after 2nd interaction: {simple_memory.get_student_data(user1)}")

        simple_memory.update_student_profile(user1, {"level": "Intermediate", "skills": {"speaking_fluency": 0.7}})
        print(f"\nData for {user1} after profile update: {simple_memory.get_student_data(user1)}")

        print(f"\nInitial data for {user2}: {simple_memory.get_student_data(user2)}")
        interaction1_user2 = {"transcript": "This is interesting!", "diagnosis": "Curious", "feedback": "Excellent!"}
        simple_memory.add_interaction_to_history(user2, interaction1_user2)
        print(f"\nData for {user2} after 1st interaction: {simple_memory.get_student_data(user2)}")

        simple_memory.clear_user_memory(user1)
        print(f"\nData for {user1} after clearing: {simple_memory.get_student_data(user1)}")
        print(f"\nData for {user2} (should be unaffected): {simple_memory.get_student_data(user2)}")

    except Exception as e:
        print(f"An error occurred during the simple_memory.py example: {e}")
