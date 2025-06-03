import requests
import json
import logging
import uuid
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
API_URL = "http://localhost:5005/process_interaction"  # Your FastAPI endpoint
TEST_USER_ID = "8f3b8a91-879b-4f72-be2c-b17cf1040de7"  # Replace with a known user ID from your system

def test_process_interaction():
    """
    Test the /process_interaction endpoint with a known user ID.
    This simulates a request from the frontend to the backend AI service.
    """
    # Create a unique session ID for this test
    session_id = str(uuid.uuid4())
    
    # Prepare the request payload according to your InteractionRequest model
    payload = {
        "transcript": "Hello, I'd like to practice for the TOEFL speaking section.",
        "current_context": {
            "user_id": TEST_USER_ID,
            "task_stage": "ROX_WELCOME_INIT",
            "toefl_section": "Speaking",
            "question_type": "Q1_Independent"
        },
        "session_id": session_id,
        "chat_history": [
            {"role": "user", "content": "I want to practice TOEFL speaking."},
            {"role": "ai", "content": "I'd be happy to help you practice for the TOEFL speaking section."}
        ]
    }
    
    logger.info(f"Sending test request to {API_URL} with user_id: {TEST_USER_ID}")
    logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Send the POST request to the endpoint
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse and log the response
        response_data = response.json()
        logger.info("Request successful!")
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response content: {json.dumps(response_data, indent=2)}")
        
        # Check if the student model node was able to fetch user data
        # This would be reflected in the response content in some way
        if "response" in response_data:
            logger.info(f"AI response: {response_data['response']}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response content: {e.response.text}")
        return False

if __name__ == "__main__":
    # Make sure your FastAPI server is running before executing this test
    logger.info("Starting test for /process_interaction endpoint")
    
    if test_process_interaction():
        logger.info("Test completed successfully!")
    else:
        logger.error("Test failed!")
