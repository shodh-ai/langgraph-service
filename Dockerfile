# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# It's good practice to set a default port, can be overridden at runtime
ENV PORT 8000

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by some Python packages
# For example, if you had a package that needed gcc or other build tools:
# RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt requirements.txt

# Install Python dependencies
# Using --no-cache-dir can reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# This includes app.py, models.py, graph_builder.py, state.py, and the agents/ & memory/ directories.
COPY . .

# Expose the port the app runs on
EXPOSE ${PORT}

# Command to run the Uvicorn server
# This will run the FastAPI application defined in app.py
# Using 0.0.0.0 to bind to all network interfaces
# The application module is specified as backend_ai_service_langgraph.app:app because WORKDIR is /app
# and the 'app' directory (which is the root of our project locally) is copied into /app.
# So, inside the container, the structure is /app/app.py, /app/models.py etc.
# If your main FastAPI 'app' instance is in 'app.py' at the root of what's copied to /app, 
# then 'app:app' is correct if uvicorn is run from /app.
# Check the uvicorn command in app.py's __main__ block for local running for reference.
# For Gunicorn (more production-ready):
# CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:${PORT}"]
# For simple Uvicorn as per current setup:
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT}"]
