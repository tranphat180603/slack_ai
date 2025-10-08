# Use a stable Debian base and avoid deb.debian.org timeouts on DO
FROM python:3.11-slim-bookworm

# Reduce noisy prompts in apt
ENV DEBIAN_FRONTEND=noninteractive
# Optional but helpful
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set working directory
WORKDIR /app

# Configure APT: force IPv4 + retries + short timeouts,
# remove default deb.debian.org deb822 source, and use HTTPS mirrors.
RUN set -eux; \
  # Force IPv4 and add retries/timeouts so failures happen fast
  printf 'Acquire::Retries "5";\nAcquire::ForceIPv4 "true";\nAcquire::http::Timeout "10";\nAcquire::https::Timeout "10";\n' \
    > /etc/apt/apt.conf.d/99net; \
  # The base image ships an extra deb822 source pointing at deb.debian.org — remove it
  rm -f /etc/apt/sources.list.d/debian.sources; \
  # Use official Debian mirrors (IPv4 forced above)
  printf '%s\n' \
    'deb http://deb.debian.org/debian bookworm main' \
    'deb http://deb.debian.org/debian bookworm-updates main' \
    'deb http://security.debian.org/debian-security bookworm-security main' \
    > /etc/apt/sources.list; \
  # Update + install system deps
  apt-get update; \
  apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    postgresql-client \
    ca-certificates \
  ; rm -rf /var/lib/apt/lists/*

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
