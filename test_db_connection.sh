#!/bin/bash

echo "Testing direct database connection..."

# Run psql command inside the database container
echo "1. Testing connection using psql inside the db container:"
docker-compose exec db psql -U phattran -d tmai_db -c "SELECT version();"

echo "\n2. Testing connection from another container:"
docker-compose exec website-db-sync psql -h db -U phattran -d tmai_db -c "SELECT version();"

echo "\n3. Checking PostgreSQL authentication configuration:"
docker-compose exec db cat /var/lib/postgresql/data/pg_hba.conf

echo "\n4. Checking database users:"
docker-compose exec db psql -U postgres -c "SELECT usename, usesuper FROM pg_user;" 