import sys
import os
import logging
import json
from dotenv import load_dotenv

# Add parent directory to path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db_utils import fetch_user_by_id, fetch_user_skills

# Load environment variables from test file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env.test'))

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_student_model_functions():
    """Test the student model functions without relying on the memory system."""
    # Test user ID - valid UUID from your database
    test_user_id = "8f3b8a91-879b-4f72-be2c-b17cf1040de7"
    
    logger.info(f"Testing student model functions for user ID: {test_user_id}")
    
    # Fetch user data
    user_data = fetch_user_by_id(test_user_id)
    logger.info(f"User data: {json.dumps(user_data, indent=2)}")
    
    # Fetch user skills
    skills_data = fetch_user_skills(test_user_id)
    logger.info(f"Skills data: {json.dumps(skills_data, indent=2)}")
    
    # Simulate profile creation
    profile = {
        "name": f"{user_data.get('firstName', '')} {user_data.get('lastName', '')}".strip(),
        "occupation": user_data.get('occupation', ''),
        "major": user_data.get('major', ''),
        "native_language": user_data.get('nativeLanguage', ''),
        "created_at": user_data.get('createdAt', ''),
        "skills": skills_data
    }
    
    logger.info(f"Created profile: {json.dumps(profile, indent=2)}")
    
    return {
        "user_data": user_data,
        "skills_data": skills_data,
        "profile": profile
    }

if __name__ == "__main__":
    test_student_model_functions()
