-- Create postgres role to prevent TimescaleDB errors
CREATE ROLE postgres WITH LOGIN PASSWORD 'postgres';

-- Create phattran user with superuser privileges
DO
$$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'phattran') THEN
    CREATE ROLE phattran WITH LOGIN SUPERUSER PASSWORD 'phatdeptrai123';
  END IF;
END
$$;