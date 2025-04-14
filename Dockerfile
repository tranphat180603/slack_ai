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

# Copy all directories by name to ensure correct structure
COPY ops_linear_db/ /app/ops_linear_db/
COPY ops_conversation_db/ /app/ops_conversation_db/
COPY ops_slack/ /app/ops_slack/
COPY app/ /app/app/
COPY tools/ /app/tools/
COPY llm/ /app/llm/
COPY prompts/ /app/prompts/
COPY *.py /app/
COPY wait-for-it.sh .

# Make wait-for-it.sh executable
RUN chmod +x wait-for-it.sh

# Verify directory structure
RUN ls -la /app && \
    ls -la /app/ops_linear_db && \
    ls -la /app/ops_conversation_db

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose the port the app runs on
EXPOSE ${PORT}

# Command will be specified in docker-compose.yml 