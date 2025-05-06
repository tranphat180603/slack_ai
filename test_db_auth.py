#!/usr/bin/env python3
"""
Enhanced database connection test with authentication debugging
"""

import os
import sys
import time
import socket
import logging
import psycopg2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_auth_test")

def test_network_connection(host, port):
    """Test basic TCP connection to the database host and port"""
    logger.info(f"Testing network connection to {host}:{port}")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((host, port))
        s.close()
        logger.info(f"✅ Network connection to {host}:{port} successful")
        return True
    except Exception as e:
        logger.error(f"❌ Network connection to {host}:{port} failed: {str(e)}")
        return False

def test_db_connection_with_retries(max_retries=3, retry_delay=5):
    """Test connection to the database with retries"""
    # Get database connection details from environment
    db_host = os.environ.get("POSTGRES_HOST", "db")
    db_port = int(os.environ.get("POSTGRES_PORT", "5432"))
    db_name = os.environ.get("POSTGRES_DB", "tmai_db")
    db_user = os.environ.get("POSTGRES_USER", "phattran")
    db_password = os.environ.get("POSTGRES_PASSWORD", "phatdeptrai123")
    
    logger.info("=== Database Connection Test ===")
    logger.info(f"Host: {db_host}, Port: {db_port}, Database: {db_name}")
    logger.info(f"User: {db_user}, Password: {'*' * len(db_password)}")
    
    # Test network connection first
    if not test_network_connection(db_host, db_port):
        logger.error("Network connectivity test failed - database server may be unreachable")
        return False
    
    # Build connection string
    conn_string = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password}"
    
    for attempt in range(1, max_retries + 1):
        logger.info(f"Connection attempt {attempt} of {max_retries}")
        try:
            # Connect to the database
            logger.info("Connecting to PostgreSQL...")
            conn = psycopg2.connect(conn_string)
            
            # Test authentication
            logger.info("Testing authentication and connection...")
            with conn.cursor() as cur:
                # Get database version
                cur.execute("SELECT version();")
                db_version = cur.fetchone()
                logger.info(f"✅ Connected successfully to PostgreSQL version: {db_version[0]}")
                
                # Get current user
                cur.execute("SELECT current_user, session_user;")
                user_info = cur.fetchone()
                logger.info(f"✅ Authenticated as: current_user={user_info[0]}, session_user={user_info[1]}")
                
                # Check user permissions
                cur.execute("""
                    SELECT r.rolname, r.rolsuper, r.rolinherit, r.rolcreaterole, 
                           r.rolcreatedb, r.rolcanlogin, r.rolreplication
                    FROM pg_roles r
                    WHERE r.rolname = current_user;
                """)
                role_info = cur.fetchone()
                logger.info(f"User permissions: superuser={role_info[1]}, createdb={role_info[4]}, login={role_info[5]}")
                
                # List all tables
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema='public'
                """)
                tables = cur.fetchall()
                if tables:
                    logger.info(f"Found {len(tables)} tables:")
                    for table in tables:
                        logger.info(f"  - {table[0]}")
                else:
                    logger.warning("No tables found in database")
                
            conn.close()
            logger.info("✅ Database connection test completed successfully")
            return True
            
        except psycopg2.OperationalError as e:
            if "authentication failed" in str(e):
                logger.error(f"❌ Authentication error: {str(e)}")
                logger.error("This suggests an incorrect username or password")
            elif "does not exist" in str(e):
                logger.error(f"❌ Database error: {str(e)}")
                logger.error("The database or user may not exist")
            else:
                logger.error(f"❌ Database connection error: {str(e)}")
            
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed after {max_retries} attempts")
                return False
                
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed after {max_retries} attempts")
                return False

if __name__ == "__main__":
    print("Starting enhanced database authentication test...")
    try:
        success = test_db_connection_with_retries()
        if success:
            print("✅ Database connection and authentication test passed")
            sys.exit(0)
        else:
            print("❌ Database connection and authentication test failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        sys.exit(1) 