from typing import Any, Dict, List
import logging
from mem0 import Memory
import os
from dotenv import load_dotenv
import json
# Try to import vertexai, but don't fail if it's not available
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    vertexai_available = True
except ImportError:
    vertexai_available = False
    
# Import our fallback utilities
from utils.fallback_utils import get_model_with_fallback

load_dotenv()

logger = logging.getLogger(__name__)

class Mem0Memory:
    """A memory class using mem0 for storing and retrieving user-centric data."""
    def __init__(self, user_id_field: str = "user_id"):
        self.mem0_instance = Memory()
        self.user_id_field = user_id_field # Field name to identify user in data
        logger.info(f"Mem0Memory initialized. Using '{user_id_field}' as user identifier.")

    def get_student_data(self, user_id: str) -> Dict[str, Any]:
        logger.info(f"Mem0Memory: Attempting to get all data for user_id: {user_id}")
        try:
            # mem0's get_all retrieves all memories. We might need to filter by user_id if not implicitly handled
            # or structure data with user_id as a key part of the memory entries.
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
            
            # Based on error messages, it seems all_memories contains strings
            for mem in all_memories:
                try:
                    # Try to parse the memory as JSON
                    memory_data = json.loads(mem)
                    if memory_data.get('type') == 'profile':
                        student_data["profile"].update(memory_data.get('content', {}))
                    elif memory_data.get('type') == 'interaction':
                        student_data["interaction_history"].append(memory_data.get('content', {}))
                except (json.JSONDecodeError, AttributeError):
                    # If it's not JSON or doesn't have the expected structure, skip it
                    logger.warning(f"Skipping memory that couldn't be parsed: {mem}")

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
        logger.info(f"Mem0Memory: Attempting to add interaction for user_id: {user_id} with summary: {interaction_summary}")
        try:
            memory_content = {
                'type': 'interaction',
                'content': interaction_summary,
                self.user_id_field: user_id
            }
            
            memory_str = json.dumps(memory_content)
            
            self.mem0_instance.add(memory_str, user_id=user_id)
            
            logger.info(f"Mem0Memory: Added interaction for {user_id}.")
        except Exception as e:
            logger.error(f"Mem0Memory: Error adding interaction for {user_id}: {e}")

    def update_student_profile(self, user_id: str, profile_data: Dict[str, Any]) -> None:
        """Updates or sets profile data for a student using mem0."""
        logger.info(f"Mem0Memory: Attempting to update profile for user_id: {user_id} with data: {profile_data}")
        try:
            memory_content = {
                'type': 'profile',
                'content': profile_data,
                self.user_id_field: user_id
            }
            
            memory_str = json.dumps(memory_content)
            
            self.mem0_instance.add(memory_str, user_id=user_id)
            
            logger.info(f"Mem0Memory: Updated/added profile for {user_id}.")
        except Exception as e:
            logger.error(f"Mem0Memory: Error updating profile for {user_id}: {e}")

    def clear_user_memory(self, user_id: str) -> None:
        """Clears all memory for a specific user in mem0."""
        logger.info(f"Mem0Memory: Attempting to clear all data for user_id: {user_id}")
        try:
            memories_to_delete = self.mem0_instance.get_all(user_id=user_id)
            
            for mem in memories_to_delete:
                try:
                    self.mem0_instance.delete(mem)
                except Exception as inner_e:
                    logger.warning(f"Failed to delete memory {mem}: {inner_e}")
            
            logger.info(f"Mem0Memory: Cleared all data for user_id: {user_id}")
        except Exception as e:
            logger.error(f"Mem0Memory: Error clearing data for {user_id}: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    # Test code commented out due to Google credentials issues
    print("mem0_memory module loaded successfully")
    
    '''
    try:
        print("Environment variables check:")
        for key in ['GOOGLE_APPLICATION_CREDENTIALS', 'GOOGLE_CLOUD_PROJECT', 'QDRANT_URL']:
            print(f"  {key}: {'✓ Set' if os.getenv(key) else '✗ Not set'}")
        
        # Print mem0 version info if available
        try:
            import importlib.metadata
            print(f"mem0 version: {importlib.metadata.version('mem0')}")
        except (ImportError, importlib.metadata.PackageNotFoundError):
            print("Could not determine mem0 version")
        
        # Check Vertex AI access
        print("\nChecking Vertex AI access:")
        try:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "windy-orb-460108-t0")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            vertexai.init(project=project_id, location=location)
            print(f"  Successfully initialized Vertex AI with project: {project_id}")
            
            try:
                # model = GenerativeModel("gemini-1.5-pro") - commented out to avoid import error
                print("  Successfully loaded Gemini model")
            except Exception as model_error:
                print(f"  Failed to load Gemini model: {model_error}")
        except Exception as vertex_error:
            print(f"  Failed to initialize Vertex AI: {vertex_error}")
        
        print("\n--- Starting mem0 memory test ---")
        mem0_memory_instance = Mem0Memory()
        
        print("\nTesting direct mem0 API:")
        try:
            test_memory = json.dumps({"test": "value", "user_id": "test_user"})
            print(f"  Adding test memory: {test_memory}")
            mem0_memory_instance.mem0_instance.add(test_memory, user_id="test_user")
            
            print("  Retrieving memories for test_user")
            test_memories = mem0_memory_instance.mem0_instance.get_all(user_id="test_user")
            print(f"  Retrieved {len(test_memories)} memories")
            
            if test_memories:
                print("  Cleaning up test memory")
                for mem in test_memories:
                    mem0_memory_instance.mem0_instance.delete(mem)
        except Exception as api_test_error:
            print(f"  Direct API test failed: {api_test_error}")
        
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
    '''
