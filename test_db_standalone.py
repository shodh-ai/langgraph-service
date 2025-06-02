import asyncio
import logging
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "pronity")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

def get_db_connection():
    """
    Create and return a connection to the PostgreSQL database.
    """
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

def fetch_user_by_id(user_id):
    """
    Fetch user information from the database by user ID.
    
    Args:
        user_id (str): The user's unique identifier
        
    Returns:
        dict: User information including profile data
    """
    try:
        connection = get_db_connection()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # Query to fetch user data - adjusted to match actual table name and column names
        cursor.execute(
            """
            SELECT id, "firstName", "lastName", occupation, major, "nativeLanguage", "flowId", "createdAt"
            FROM "User" 
            WHERE id = %s
            """, 
            (user_id,)
        )
        
        user_data = cursor.fetchone()
        
        # Close cursor and connection
        cursor.close()
        connection.close()
        
        if user_data:
            return dict(user_data)
        else:
            logger.warning(f"No user found with ID: {user_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching user data: {e}")
        return None

def fetch_user_skills(user_id):
    """
    Fetch user skills from the database by user ID.
    
    Args:
        user_id (str): The user's unique identifier
        
    Returns:
        dict: User skills data
    """
    try:
        connection = get_db_connection()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # Check if UserSkill table exists (using proper case)
        cursor.execute(
            """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = 'UserSkill'
            """
        )
        
        skills_table_exists = cursor.fetchone()
        
        if skills_table_exists:
            # If the skills table exists, query it
            cursor.execute(
                """
                SELECT * FROM "UserSkill" 
                WHERE "userId" = %s
                """, 
                (user_id,)
            )
            
            skills_data = cursor.fetchall()
            
            # Transform list of skill records into a dictionary
            skills = {}
            for skill in skills_data:
                skills[skill['skill_name']] = skill['skill_score']
        else:
            # If no skills table exists, return empty skills
            logger.info("No UserSkill table found in the database")
            skills = {}
        
        # Close cursor and connection
        cursor.close()
        connection.close()
        
        return skills
            
    except Exception as e:
        logger.error(f"Error fetching user skills: {e}")
        return {}

async def test_db_connection():
    """Test direct database connection"""
    logger.info("Testing direct database connection...")
    
    # Use an existing user ID from the database
    test_user_id = "8f3b8a91-879b-4f72-be2c-b17cf1040de7"  # test User
    # Alternative user: "cea9bd18-1939-4ba4-a233-6219494b1ba5"  # Anushk User
    
    try:
        # Test database connection by fetching user data
        user_data = fetch_user_by_id(test_user_id)
        if user_data:
            logger.info(f"Successfully fetched user data from PostgreSQL: {user_data}")
        else:
            logger.warning(f"No user found with ID: {test_user_id}")
            logger.info("Check if the user ID exists in the database and if the column names match what's expected in fetch_user_by_id()")
        
        # Test fetching user skills
        skills = fetch_user_skills(test_user_id)
        logger.info(f"User skills: {skills}")
        
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        logger.info("Check your database connection parameters in .env file")
        logger.info(f"Current DB_HOST: {os.getenv('DB_HOST')}")
        logger.info(f"Current DB_PORT: {os.getenv('DB_PORT')}")
        logger.info(f"Current DB_NAME: {os.getenv('DB_NAME')}")
        return False
    
    return True

async def main():
    """Main test function"""
    logger.info("Starting database connection test...")
    
    # Test database connection
    db_connection_success = await test_db_connection()
    
    if db_connection_success:
        logger.info("Database connection test successful!")
    else:
        logger.error("Database connection test failed!")
    
    logger.info("Database test completed")

if __name__ == "__main__":
    asyncio.run(main())
