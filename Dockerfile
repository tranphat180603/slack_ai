# Use an Alpine base to avoid apt-based package installs
FROM python:3.11-alpine

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies using apk (no apt involved)
RUN apk add --no-cache \
    build-base \
    postgresql-dev \
    postgresql-client \
    ca-certificates \
    libffi-dev \
    openssl-dev

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python deps (pin httpx first as you wanted)
RUN pip install --upgrade pip && \
    pip install httpx==0.27.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY ops_linear_db/        /app/ops_linear_db/
COPY ops_conversation_db/  /app/ops_conversation_db/
COPY ops_slack/            /app/ops_slack/
COPY ops_website_db/       /app/ops_website_db/
COPY ops_gdrive/           /app/ops_gdrive/
COPY ops_posthog/          /app/ops_posthog/
COPY app/                  /app/app/
COPY tools/                /app/tools/
COPY llm/                  /app/llm/
COPY prompts/              /app/prompts/
COPY *.py                  /app/
COPY wait-for-it.sh        .

# Make wait-for-it.sh executable
RUN chmod +x wait-for-it.sh

# (Optional) Verify structure
RUN ls -la /app && \
    ls -la /app/ops_linear_db && \
    ls -la /app/ops_conversation_db && \
    ls -la /app/ops_website_db && \
    ls -la /app/ops_gdrive && \
    ls -la /app/ops_posthog

# Expose the app port
EXPOSE ${PORT}

# CMD/ENTRYPOINT defined in docker-compose.yml
