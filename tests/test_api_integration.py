import sys
import os
import logging
import json
import requests
from dotenv import load_dotenv

# Add parent directory to path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API configuration
API_BASE_URL = "http://localhost:8000/api"
VALID_USER_ID = "8f3b8a91-879b-4f72-be2c-b17cf1040de7"

def test_api_endpoints():
    """Test direct API connections to the backend endpoints."""
    logger.info(f"Testing API endpoints at {API_BASE_URL}")
    
    # Test health endpoint
    try:
        health_url = f"{API_BASE_URL}/health"
        logger.info(f"Testing health endpoint: {health_url}")
        health_response = requests.get(health_url, timeout=5)
        logger.info(f"Health check status: {health_response.status_code}")
    except Exception as e:
        logger.error(f"Error connecting to health endpoint: {e}")
    
    # Test user endpoint
    try:
        user_url = f"{API_BASE_URL}/users/{VALID_USER_ID}"
        logger.info(f"Testing user endpoint: {user_url}")
        user_response = requests.get(user_url, timeout=5)
        logger.info(f"User endpoint status: {user_response.status_code}")
        
        if user_response.status_code == 200:
            user_data = user_response.json()
            logger.info(f"User data: {json.dumps(user_data, indent=2)}")
            
            # Check if we got the expected structure
            if "success" in user_data and user_data["success"] and "data" in user_data:
                logger.info("User endpoint returned the expected structure")
                logger.info(f"User name: {user_data['data'].get('firstName', '')} {user_data['data'].get('lastName', '')}")
            else:
                logger.warning("User endpoint did not return the expected structure")
        else:
            logger.error(f"Failed to get user data: {user_response.status_code}")
    except Exception as e:
        logger.error(f"Error connecting to user endpoint: {e}")
    
    # Test skills endpoint
    try:
        skills_url = f"{API_BASE_URL}/users/{VALID_USER_ID}/skills"
        logger.info(f"Testing skills endpoint: {skills_url}")
        skills_response = requests.get(skills_url, timeout=5)
        logger.info(f"Skills endpoint status: {skills_response.status_code}")
        
        if skills_response.status_code == 200:
            skills_data = skills_response.json()
            logger.info(f"Skills data: {json.dumps(skills_data, indent=2)}")
            
            # Check if we got the expected structure
            if "success" in skills_data and skills_data["success"] and "data" in skills_data:
                logger.info("Skills endpoint returned the expected structure")
                if "fallback" in skills_data and skills_data["fallback"]:
                    logger.info("Skills data is using fallback values")
            else:
                logger.warning("Skills endpoint did not return the expected structure")
        else:
            logger.error(f"Failed to get skills data: {skills_response.status_code}")
    except Exception as e:
        logger.error(f"Error connecting to skills endpoint: {e}")
    
    return {
        "api_base_url": API_BASE_URL,
        "user_id": VALID_USER_ID,
        "endpoints_tested": ["health", "users", "skills"]
    }

if __name__ == "__main__":
    test_api_endpoints()
