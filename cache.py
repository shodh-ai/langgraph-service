# cache.py
from typing import Dict, Any

# A simple in-memory cache for the service.
# This can be replaced with a more robust solution like Redis in the future.
SESSION_DATA_CACHE: Dict[str, Dict[str, Any]] = {}
