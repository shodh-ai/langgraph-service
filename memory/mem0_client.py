import os
import threading
from typing import Optional, List, Dict, Any
from mem0 import MemoryClient
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Singleton class for Mem0 client to avoid file lock issues on Windows
class Mem0Client:
    _instance: Optional['Mem0Client'] = None
    _lock = threading.Lock()
    is_initialized = False

    def __new__(cls, *args, **kwargs) -> 'Mem0Client':
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self.is_initialized:
            return
        with self._lock:
            if self.is_initialized:
                return
            
            logger.info("Initializing Mem0Client singleton...")
            
            self.api_key = os.getenv("MEM0_API_KEY")
            self.org_id = os.getenv("MEM0_ORG_ID")
            self.project_id = os.getenv("MEM0_PROJECT_ID")

            if not self.api_key:
                logger.error("MEM0_API_KEY not found in environment variables.")
                raise ValueError("MEM0_API_KEY not found in environment variables.")
            if not self.org_id:
                logger.error("MEM0_ORG_ID not found in environment variables.")
                raise ValueError("MEM0_ORG_ID not found in environment variables.")
            if not self.project_id:
                logger.error("MEM0_PROJECT_ID not found in environment variables.")
                raise ValueError("MEM0_PROJECT_ID not found in environment variables.")

            try:
                self.mem0_instance = MemoryClient(
                    api_key=self.api_key,
                    org_id=self.org_id,
                    project_id=self.project_id
                )
                logger.info("Mem0Client initialized successfully using MemoryClient with org_id and project_id.")
            except Exception as e:
                logger.error(f"Failed to initialize Mem0 client with MemoryClient: {e}", exc_info=True)
                raise
            
            self.is_initialized = True

    def add(self, messages: List[Dict[str, str]], user_id: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        logger.debug(f"Mem0Client: Calling add for user_id: {user_id} with messages: {messages} and metadata: {metadata}")
        try:
            return self.mem0_instance.add(messages=messages, user_id=user_id, metadata=metadata)
        except Exception as e:
            logger.error(f"Mem0Client: Error in add method: {e}", exc_info=True)
            raise

    def get_all(self, user_id: str) -> List[Dict[str, Any]]:
        logger.debug(f"Mem0Client: Calling get_all for user_id: {user_id}")
        try:
            # The MemoryClient().get_all() might return a different structure or need different handling.
            # Assuming it's similar to mem0.Memory().get_all() for now.
            # Refer to Mem0Client documentation if issues arise.
            raw_memories = self.mem0_instance.get_all(user_id=user_id)
            # Ensure the output format is consistent if it differs, e.g. by wrapping in {'results': raw_memories}
            # For now, assuming it returns a list directly, or an object with a 'results' key like before.
            if isinstance(raw_memories, dict) and 'results' in raw_memories:
                 return raw_memories # Matches previous structure
            elif isinstance(raw_memories, list):
                 return {'results': raw_memories} # Adapt if it returns a list directly
            logger.warning(f"Mem0Client: get_all returned unexpected format: {type(raw_memories)}. Adjust if needed.")
            return raw_memories # Fallback, might need adjustment

        except Exception as e:
            logger.error(f"Mem0Client: Error in get_all method: {e}", exc_info=True)
            raise

    def search(self, query: str, user_id: str, limit: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        logger.debug(f"Mem0Client: Calling search for user_id: {user_id} with query: {query}, limit: {limit}, metadata: {metadata}")
        try:
            return self.mem0_instance.search(query=query, user_id=user_id, limit=limit, metadata=metadata)
        except Exception as e:
            logger.error(f"Mem0Client: Error in search method: {e}", exc_info=True)
            raise

# Create a single, shared instance of the client
logger.info("Creating shared_mem0_client instance.")
shared_mem0_client = Mem0Client()
logger.info("shared_mem0_client instance created.")
