# TOEFL Tutor Backend Environment Setup Guide

## Google Cloud Authentication

The error `File AIzaSyDhHE9MFgtBr-BX-Z0-S3umEXEIWSZOurQ was not found` indicates you're using an API key directly as the credentials file path. Instead, you need to:

1. Create a service account key file (JSON) in Google Cloud Console
2. Download the JSON key file
3. Set the environment variable to point to this file path

```bash
# In your .env file:
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account-key.json
GOOGLE_CLOUD_PROJECT=windy-orb-460108-t0
```

## PostgreSQL Connection

The error `role "postgres" does not exist` indicates a PostgreSQL authentication issue. Since you're now using API-based access instead of direct PostgreSQL connections, you should:

1. Make sure your backend API is running on port 8000
2. Update your .env file with the correct API settings:

```bash
# API connection (new approach)
API_BASE_URL=http://localhost:8000/api
API_KEY=your_api_key_if_needed

# Database connection (legacy approach)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pronity
DB_USER=your_mac_username  # Instead of "postgres"
DB_PASSWORD=postgres
```

## Running the Application

To run the application with the correct environment:

1. Make sure your backend API server is running on port 8000
2. Start the AI service with:

```bash
python -m uvicorn app:app --reload --host 0.0.0.0 --port 5005
```

## Testing API Connection

You can test your API connection with:

```bash
python tests/test_api_integration.py
```

This will verify that your backend API is accessible and returning the expected data.
