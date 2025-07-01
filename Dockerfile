# Date: 2025-04-05
# Created by: Addisu Taye
# Purpose: Containerize the FastAPI application for deployment consistency and scalability.
# Key features:
# - Sets up Python environment
# - Installs dependencies from requirements.txt
# - Copies source code into container
# - Exposes port 8000
# - Runs the FastAPI service using Uvicorn
# Task Number: 6
# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src ./src
COPY tests ./tests

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]