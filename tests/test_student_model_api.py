import sys
import os
import logging
import asyncio
from dotenv import load_dotenv

# Add parent directory to path so we can import from the main package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from state import AgentGraphState
from utils.db_utils import fetch_user_by_id, fetch_user_skills
from agents.student_model_node import load_or_initialize_student_profile

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from test file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env.test'))

# Mock mem0_memory for testing without actual memory system
class MockMem0Memory:
    def __init__(self):
        self.student_data = {}
    
    def get_student_data(self, user_id):
        if user_id not in self.student_data:
            self.student_data[user_id] = {"profile": {}, "history": []}
        return self.student_data[user_id]
    
    def update_student_profile(self, user_id, profile):
        if user_id not in self.student_data:
            self.student_data[user_id] = {"profile": {}, "history": []}
        self.student_data[user_id]["profile"] = profile
        return self.student_data[user_id]

# Patch the mem0_memory import in student_model_node
import agents.student_model_node
agents.student_model_node.mem0_memory = MockMem0Memory()

async def test_student_model_api_integration():
    """Test the student model node's API integration."""
    # Test user ID - valid UUID from your database
    test_user_id = "8f3b8a91-879b-4f72-be2c-b17cf1040de7"
    
    logger.info(f"Testing student model API integration for user ID: {test_user_id}")
    
    # Create a state object with the user ID
    state = AgentGraphState({"user_id": test_user_id})
    
    # Call the student model node function
    result = await load_or_initialize_student_profile(state)
    
    # Check the result
    if result and "student_memory_context" in result:
        profile = result["student_memory_context"].get("profile", {})
        logger.info(f"Student profile created: {profile}")
        
        # Verify that we got real data from the API
        if profile.get("name") == "test User":
            logger.info("Successfully fetched and mapped real user data from API")
        else:
            logger.warning("Profile name doesn't match expected value")
            
        # Check skills
        skills = profile.get("skills", {})
        logger.info(f"Student skills: {skills}")
        
        return True, profile
    else:
        logger.error("Failed to create student profile")
        return False, None

if __name__ == "__main__":
    logger.info("Starting student model API integration test")
    success, profile = asyncio.run(test_student_model_api_integration())
    
    if success:
        logger.info("Student model API integration test successful!")
        logger.info(f"Final profile: {profile}")
    else:
        logger.error("Student model API integration test failed!")
