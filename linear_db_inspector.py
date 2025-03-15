"""
Database inspector for the Linear RAG system.
Displays the database schema and sample data from each table.
"""
import os
import psycopg2
import pandas as pd
from tabulate import tabulate
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("linear_db_inspector")

def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    database = os.environ.get("POSTGRES_DB", "linear_rag")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "Phatdeptrai@123")
    
    try:
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise

def get_all_tables(conn) -> List[str]:
    """Get a list of all tables in the database."""
    cur = conn.cursor()
    try:
        cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
        """)
        tables = [row[0] for row in cur.fetchall()]
        return tables
    finally:
        cur.close()

def get_table_count(conn, table_name: str) -> int:
    """Get the number of rows in a table."""
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cur.fetchone()[0]
        return count
    except Exception as e:
        logger.error(f"Error getting count for table {table_name}: {str(e)}")
        return -1
    finally:
        cur.close()

def get_table_columns(conn, table_name: str) -> List[str]:
    """Get the column names for a table."""
    cur = conn.cursor()
    try:
        cur.execute(f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """, (table_name,))
        columns = [row[0] for row in cur.fetchall()]
        return columns
    finally:
        cur.close()

def get_table_sample(conn, table_name: str, limit: int = 10) -> List[Dict]:
    """Get a sample of rows from a table."""
    cur = conn.cursor()
    try:
        # Get column names
        columns = get_table_columns(conn, table_name)
        
        # Fetch sample data
        cur.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        rows = cur.fetchall()
        
        # Format as list of dictionaries
        results = []
        for row in rows:
            result = {}
            for i, col in enumerate(columns):
                # Handle data types that need special formatting
                if isinstance(row[i], bytes):
                    result[col] = "<binary data>"
                else:
                    result[col] = row[i]
            results.append(result)
        
        return results, columns
    except Exception as e:
        logger.error(f"Error getting sample for table {table_name}: {str(e)}")
        return [], []
    finally:
        cur.close()

def truncate_long_value(value: Any, max_length: int = 100) -> str:
    """Truncate long string values for better display."""
    if value is None:
        return "NULL"
    
    string_value = str(value)
    if len(string_value) > max_length:
        return string_value[:max_length] + "..."
    return string_value

def inspect_database():
    """Inspect the database and display information."""
    conn = get_db_connection()
    
    try:
        # Get all tables
        tables = get_all_tables(conn)
        print(f"Found {len(tables)} tables in the database:")
        
        # Display table counts
        table_counts = []
        for table in tables:
            count = get_table_count(conn, table)
            table_counts.append((table, count))
        
        # Sort by row count (descending)
        table_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Display table counts
        count_df = pd.DataFrame(table_counts, columns=["Table", "Row Count"])
        print("\n=== Table Row Counts ===")
        print(tabulate(count_df, headers="keys", tablefmt="grid", showindex=False))
        
        # Process each table
        for table, count in table_counts:
            print(f"\n\n=== Table: {table} ({count} rows) ===")
            
            # Skip vector embeddings table for display (it's usually very large)
            if table == "embeddings" and count > 100:
                print("Skipping full display of embeddings table (too large).")
                # Just show the schema
                columns = get_table_columns(conn, table)
                print("\nColumns:")
                for col in columns:
                    print(f"- {col}")
                continue
            
            # Get sample data
            sample_data, columns = get_table_sample(conn, table)
            
            if not sample_data:
                print("No data available or error occurred.")
                continue
            
            # Format the data for display
            formatted_data = []
            for row in sample_data:
                formatted_row = {}
                for col, val in row.items():
                    # Skip embedding vector display
                    if col == "embedding":
                        formatted_row[col] = "<vector data>"
                    else:
                        formatted_row[col] = truncate_long_value(val)
                formatted_data.append(formatted_row)
            
            # Display sample data
            if formatted_data:
                # Create DataFrame for nice display
                df = pd.DataFrame(formatted_data)
                
                # Select a subset of columns if there are too many
                if len(df.columns) > 8:
                    # Prioritize ID columns and exclude large text fields
                    priority_cols = [col for col in df.columns if 'id' in col.lower()]
                    other_cols = [col for col in df.columns if 'id' not in col.lower() 
                                 and col not in ['description', 'content', 'embedding', 'metadata']]
                    display_cols = priority_cols + other_cols
                    display_cols = display_cols[:8]  # Take at most 8 columns
                    print(f"Table has {len(df.columns)} columns. Showing {len(display_cols)} key columns.")
                    print(tabulate(df[display_cols], headers="keys", tablefmt="grid", showindex=True))
                    print(f"Hidden columns: {', '.join(col for col in df.columns if col not in display_cols)}")
                else:
                    print(tabulate(df, headers="keys", tablefmt="grid", showindex=True))
    
    except Exception as e:
        logger.error(f"Error inspecting database: {str(e)}")
        print(f"Error inspecting database: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    inspect_database() 