#!/bin/bash

echo "Testing using the init_conversation_db.py file directly with absolute imports..."

# Create a Docker container based on python:3.11-slim
docker run --rm -it --name test_init_file \
  --network=slack_ai_slack_ai_network \
  -e DATABASE_URL=postgresql://phattran:phatdeptrai123@db:5432/tmai_db \
  -e POSTGRES_HOST=db \
  -v $(pwd):/app \
  -w /app \
  python:3.11-slim \
  /bin/bash -c "pip install sqlalchemy psycopg2-binary python-dotenv && PYTHONPATH=/app python ops_conversation_db/init_conversation_db.py" 