import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:3000/api")
API_KEY = os.getenv("API_KEY")  # For service-to-service authentication

def fetch_user_by_id(user_id):
    """Fetch user information from the backend API by user ID."""
    try:
        logger.info(f"Fetching user with ID: {user_id} from API: {API_BASE_URL}")
        headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
        response = requests.get(f"{API_BASE_URL}/users/{user_id}", headers=headers)
        
        # Log response details for debugging
        logger.info(f"API response status: {response.status_code}")
        
        response.raise_for_status()
        response_json = response.json()
        logger.info(f"Successfully fetched user data: {response_json}")
        
        # Extract data from the API response structure
        if response_json.get("success") and "data" in response_json:
            return response_json["data"]
        else:
            logger.warning("API response did not contain expected structure")
            return response_json
    except Exception as e:
        logger.error(f"API request error fetching user: {e}")
        # Provide a fallback user data structure for testing
        logger.warning("Using fallback mock user data")
        return {
            "id": user_id,
            "firstName": "Test",
            "lastName": "User",
            "occupation": "Student",
            "major": "Computer Science",
            "nativeLanguage": "English",
            "createdAt": "2025-06-01T00:00:00Z"
        }

def fetch_user_skills(user_id):
    """Fetch user skills from the backend API by user ID."""
    try:
        logger.info(f"Fetching skills for user with ID: {user_id} from API: {API_BASE_URL}")
        headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
        response = requests.get(f"{API_BASE_URL}/users/{user_id}/skills", headers=headers)
        
        # Log response details for debugging
        logger.info(f"API response status: {response.status_code}")
        
        response.raise_for_status()
        response_json = response.json()
        logger.info(f"Successfully fetched skills data: {response_json}")
        
        # Extract data from the API response structure
        if response_json.get("success") and "data" in response_json:
            return response_json["data"]
        else:
            logger.warning("API response did not contain expected structure")
            return response_json
    except Exception as e:
        logger.error(f"API request error fetching skills: {e}")
        # Provide fallback skills data for testing
        logger.warning("Using fallback mock skills data")
        return {
            "speaking_fluency": 3,
            "speaking_coherence": 3,
            "speaking_vocabulary": 3,
            "speaking_grammar": 3,
            "speaking_pronunciation": 3
        }