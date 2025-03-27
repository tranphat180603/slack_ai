# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install httpx first with specific version, then other dependencies
RUN pip install httpx==0.27.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and wait-for-it script
COPY . .
COPY wait-for-it.sh .

# Make wait-for-it.sh executable
RUN chmod +x wait-for-it.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose the port the app runs on
EXPOSE ${PORT}

# Command will be specified in docker-compose.yml 