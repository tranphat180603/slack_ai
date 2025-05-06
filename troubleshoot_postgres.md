# PostgreSQL Authentication Troubleshooting Guide

This guide will help you troubleshoot PostgreSQL authentication issues in your Docker environment.

## Steps to Debug on VM

1. **Check Docker Compose Configuration**

   ```bash
   # Copy the fixed docker-compose.yml to your VM
   # Make sure the command lines in your docker-compose.yml file are not truncated
   ```

2. **Check Database Logs**

   ```bash
   # View PostgreSQL logs
   docker-compose logs db
   
   # Check for auth failures
   docker-compose logs db | grep -i "auth"
   docker-compose logs db | grep -i "fail"
   ```

3. **Test Direct Database Connection**

   ```bash
   # Run the test script
   ./test_db_connection.sh
   
   # Or directly use psql inside the container
   docker-compose exec db psql -U phattran -d tmai_db -c "SELECT version();"
   ```

4. **Check pg_hba.conf Configuration**

   ```bash
   # View PostgreSQL authentication config
   docker-compose exec db cat /var/lib/postgresql/data/pg_hba.conf
   
   # If needed, you can modify it with:
   docker-compose exec db bash -c "echo 'host all all all trust' >> /var/lib/postgresql/data/pg_hba.conf"
   docker-compose exec db pg_ctl reload
   ```

5. **Check Database Users**

   ```bash
   # List database users
   docker-compose exec db psql -U postgres -c "SELECT usename, usesuper FROM pg_user;"
   
   # Create missing user manually if needed
   docker-compose exec db psql -U postgres -c "CREATE USER phattran WITH SUPERUSER PASSWORD 'phatdeptrai123';"
   ```

6. **Reset Database if Needed**

   ```bash
   # Stop and remove containers, volumes
   docker-compose down -v
   
   # Restart with fixed configuration
   docker-compose up -d
   ```

## Common Issues and Solutions

1. **Password Authentication Failed**
   
   This typically happens when:
   - Password in environment variables doesn't match PostgreSQL user's password
   - User doesn't exist in database
   - PostgreSQL auth method is not set to 'trust'
   
   **Solution**: Create or modify user with correct password:
   ```sql
   CREATE USER phattran WITH PASSWORD 'phatdeptrai123' SUPERUSER;
   ```

2. **Trust Authentication Failed**
   
   If you see errors despite setting `POSTGRES_HOST_AUTH_METHOD=trust`, check:
   - The pg_hba.conf file may have been modified
   - The container might be using a persistent volume with old configuration
   
   **Solution**: Check and update the pg_hba.conf:
   ```bash
   docker-compose exec db bash -c "echo 'host all all all trust' >> /var/lib/postgresql/data/pg_hba.conf"
   docker-compose exec db pg_ctl reload
   ```

3. **Database Doesn't Exist**
   
   If database `tmai_db` doesn't exist:
   
   **Solution**: Create database manually:
   ```bash
   docker-compose exec db psql -U postgres -c "CREATE DATABASE tmai_db OWNER phattran;"
   ```

4. **Container Connection Issues**
   
   If services can't connect to the database container:
   
   **Solution**: Check networking and make sure all services are in the same network:
   ```bash
   docker network inspect slack_ai_network
   ```

## Recommended Fix for Current Issue

Based on error logs showing authentication failure for user `phattran`, follow these steps:

1. Ensure `create_postgres_user.sql` contains proper SQL to create phattran user
2. Down and up the containers with:
   ```bash
   docker-compose down
   docker-compose up -d
   ```
3. If issue persists, manually create the user:
   ```bash
   docker-compose exec db psql -U postgres -c "CREATE USER phattran WITH SUPERUSER PASSWORD 'phatdeptrai123';"
   docker-compose exec db psql -U postgres -c "ALTER DATABASE tmai_db OWNER TO phattran;"
   ``` 