import asyncio
import logging
import os
from dotenv import load_dotenv
from agents.student_model_node import load_or_initialize_student_profile
from utils.db_utils import fetch_user_by_id, fetch_user_skills
from memory.mem0_memory import Mem0Memory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

async def test_db_connection():
    logger.info("Testing direct database connection...")
    
    test_user_id = "8f3b8a91-879b-4f72-be2c-b17cf1040de7"  # test user
    
    try:
        logger.info(f"Attempting to connect to PostgreSQL database:")
        logger.info(f"  Host: {os.getenv('DB_HOST', 'localhost')}")
        logger.info(f"  Port: {os.getenv('DB_PORT', '5432')}")
        logger.info(f"  Database: {os.getenv('DB_NAME', 'pronity')}")
        logger.info(f"  User: {os.getenv('DB_USER', 'postgres')}")
        
        user_data = fetch_user_by_id(test_user_id)
        if user_data:
            logger.info(f"Successfully fetched user data from PostgreSQL: {user_data}")
        else:
            logger.warning(f"No user found with ID: {test_user_id}")
            logger.info("Check if the user ID exists in the database")
        
        skills = fetch_user_skills(test_user_id)
        logger.info(f"User skills: {skills}")
        
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False
    
    return True

async def test_student_model_node():
    logger.info("Testing student model node with database integration...")
    
    test_user_id = "8f3b8a91-879b-4f72-be2c-b17cf1040de7"  # test user
    
    # Create a mock state object
    mock_state = {"user_id": test_user_id}
    
    try:
        # First, clear any existing data for this test user in mem0
        mem0 = Mem0Memory()
        mem0.clear_user_memory(test_user_id)
        logger.info(f"Cleared existing mem0 data for user: {test_user_id}")
        
        # Call the student model node function
        result = await load_or_initialize_student_profile(mock_state)
        
        # Check the result
        student_data = result.get("student_memory_context", {})
        logger.info(f"Student data from student model node: {student_data}")
        
        if student_data and student_data.get("profile"):
            logger.info("Successfully loaded student profile")
            logger.info(f"Profile: {student_data['profile']}")
            
            # Verify if data was fetched from PostgreSQL
            if student_data["profile"].get("name") and not student_data["profile"]["name"].startswith("New Student"):
                logger.info("Profile appears to have been loaded from PostgreSQL")
                logger.info(f"Name: {student_data['profile'].get('name')}")
                logger.info(f"Occupation: {student_data['profile'].get('occupation')}")
                logger.info(f"Major: {student_data['profile'].get('major')}")
                logger.info(f"Native Language: {student_data['profile'].get('native_language')}")
                if student_data["profile"].get("account_created"):
                    logger.info(f"Account Created: {student_data['profile'].get('account_created')}")
            else:
                logger.info("Profile may not have been loaded from PostgreSQL")
        else:
            logger.warning("Failed to load student profile")
    
    except Exception as e:
        logger.error(f"Student model node test failed: {e}")
        return False
    
    return True

async def main():
    logger.info("Starting database integration tests...")
    
    db_connection_success = await test_db_connection()
    
    if db_connection_success:
        await test_student_model_node()
    else:
        logger.error("Skipping student model node test due to database connection failure")
    
    logger.info("Database integration tests completed")

if __name__ == "__main__":
    asyncio.run(main())
