#!/bin/bash

echo "Testing database initialization approach for Docker..."

# Create a Docker container based on python:3.11-slim
docker run --rm -it --name test_db_init \
  --network=slack_ai_slack_ai_network \
  -e DATABASE_URL=postgresql://phattran:phatdeptrai123@db:5432/tmai_db \
  -v $(pwd)/ops_conversation_db:/app/ops_conversation_db \
  -v $(pwd)/test_init_db.py:/app/test_init_db.py \
  -w /app \
  python:3.11-slim \
  /bin/bash -c "pip install sqlalchemy psycopg2-binary && python test_init_db.py db" 