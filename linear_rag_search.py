"""
Search functionality for the Linear RAG system.
"""
import os
import logging
import psycopg2
import json
import openai
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd
from tabulate import tabulate
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("linear_rag_search")

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=60))
def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Get embedding vector for a text using OpenAI API.
    
    Args:
        text: The text to embed
        model: The embedding model to use
        
    Returns:
        Embedding vector as a list of floats
    """
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise

def semantic_search(
    query: str, 
    top_k: int = 5, 
    team_filter: Optional[str] = None,
    cycle_filter: Optional[str] = None,
    employee_filter: Optional[str] = None,
    source_type_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Perform semantic search using vector similarity.
    
    Args:
        query: The search query
        top_k: Number of results to return
        team_filter: Filter results by team name
        cycle_filter: Filter results by cycle name
        employee_filter: Filter results by employee name
        source_type_filter: Filter by source type
        
    Returns:
        List of search results
    """
    # Get embedding for the query
    logger.info(f"Generating embedding for query: {query}")
    query_embedding = get_embedding(query)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Build the filter conditions
        filters = []
        filter_params = [query_embedding, query_embedding]
        
        if team_filter:
            filters.append("(metadata->>'team_name' = %s OR metadata->>'team_key' = %s)")
            filter_params.extend([team_filter, team_filter])
        
        if cycle_filter:
            filters.append("(metadata->>'cycle_name' = %s OR metadata->>'cycle_number' = %s)")
            filter_params.extend([cycle_filter, cycle_filter])
        
        if employee_filter:
            filters.append("metadata->>'assignee_name' = %s")
            filter_params.append(employee_filter)
        
        if source_type_filter:
            filters.append("source_type = %s")
            filter_params.append(source_type_filter)
        
        filter_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        
        # Add limit to the query
        filter_params.append(top_k)
        
        # Execute the query with explicit vector casting
        query_text = f"""
        SELECT 
            source_type,
            source_id,
            content,
            metadata,
            1 - (embedding <=> %s::vector) as similarity
        FROM 
            embeddings
        {filter_clause}
        ORDER BY 
            embedding <=> %s::vector
        LIMIT %s
        """
        
        logger.info(f"Executing search query with filters: {filter_clause}")
        cur.execute(query_text, filter_params)
        
        results = []
        for row in cur.fetchall():
            source_type, source_id, content, metadata_json, similarity = row
            
            # Handle metadata that could be a JSON string or already a dictionary
            if isinstance(metadata_json, str):
                metadata = json.loads(metadata_json) if metadata_json else {}
            else:
                metadata = metadata_json if metadata_json else {}
            
            result = {
                "source_type": source_type,
                "source_id": source_id,
                "content": content,
                "similarity": similarity,
                "metadata": metadata
            }
            
            # Enhance result with metadata for display
            result.update({
                "title": metadata.get("title", ""),
                "team": metadata.get("team_name", ""),
                "assignee": metadata.get("assignee_name", ""),
                "state": metadata.get("state", ""),
                "cycle": metadata.get("cycle_name", "")
            })
            
            results.append(result)
        
        logger.info(f"Search returned {len(results)} results")
        return results
    
    finally:
        cur.close()
        conn.close()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=60))
def generate_answer(query: str, context: str) -> str:
    """
    Generate an answer using GPT-4 based on the retrieved context.
    
    Args:
        query: The user's query
        context: The retrieved context to base the answer on
        
    Returns:
        Generated answer
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that answers questions about Linear issues and team activities. "
                               "Base your answers only on the provided context. If the information isn't in the context, "
                               "be honest about not knowing."
                },
                {
                    "role": "user", 
                    "content": f"Answer this question based on the following context:\n\n"
                               f"Question: {query}\n\n"
                               f"Context: {context}"
                }
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"I'm sorry, I encountered an error generating an answer: {str(e)}"

def display_results(results: List[Dict[str, Any]]) -> None:
    """Display search results in a tabular format."""
    if not results:
        print("No results found.")
        return
    
    # Convert to DataFrame for easy display
    df = pd.DataFrame(results)
    
    # Format the display
    display_df = df[["similarity", "source_type", "title", "team", "assignee", "state"]].copy()
    
    # Format similarity score
    display_df["similarity"] = display_df["similarity"].apply(lambda x: f"{x:.4f}")
    
    # Rename columns
    display_df.columns = ["Similarity", "Type", "Title", "Team", "Assignee", "State"]
    
    # Print table
    print(tabulate(display_df, headers="keys", tablefmt="grid", showindex=True))
    
    # Print detailed view of top result
    print("\n=== Top Result Details ===")
    top_result = results[0]
    print(f"Type: {top_result['source_type']}")
    
    if "title" in top_result and top_result["title"]:
        print(f"Title: {top_result['title']}")
    
    if "team" in top_result and top_result["team"]:
        print(f"Team: {top_result['team']}")
    
    if "assignee" in top_result and top_result["assignee"]:
        print(f"Assignee: {top_result['assignee']}")
    
    if "state" in top_result and top_result["state"]:
        print(f"State: {top_result['state']}")
    
    if "cycle" in top_result and top_result["cycle"]:
        print(f"Cycle: {top_result['cycle']}")
    
    print("\nContent:")
    print("---")
    print(top_result["content"])
    print("---")

def main():
    """Run the search interface."""
    parser = argparse.ArgumentParser(description="Search Linear data")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--team", help="Filter by team name or key")
    parser.add_argument("--cycle", help="Filter by cycle name or number")
    parser.add_argument("--employee", help="Filter by employee name")
    parser.add_argument("--type", choices=["issue_title", "issue_description", "comment", "employee_cycle_summary"],
                        help="Filter by content type")
    parser.add_argument("--generate", action="store_true", help="Generate an answer based on search results")
    
    args = parser.parse_args()
    
    # Perform search
    results = semantic_search(
        query=args.query,
        top_k=args.top_k,
        team_filter=args.team,
        cycle_filter=args.cycle,
        employee_filter=args.employee,
        source_type_filter=args.type
    )
    
    # Display results
    display_results(results)
    
    # Generate answer if requested
    if args.generate and results:
        print("\n=== Generated Answer ===")
        
        # Create context from results
        context = ""
        for i, result in enumerate(results):
            context += f"--- Document {i+1} ---\n"
            
            if result.get("title"):
                context += f"Title: {result['title']}\n"
            
            if result.get("team"):
                context += f"Team: {result['team']}\n"
            
            if result.get("assignee"):
                context += f"Assignee: {result['assignee']}\n"
            
            if result.get("state"):
                context += f"State: {result['state']}\n"
            
            if result.get("cycle"):
                context += f"Cycle: {result['cycle']}\n"
            
            context += f"\n{result['content']}\n\n"
        
        answer = generate_answer(args.query, context)
        print(answer)

if __name__ == "__main__":
    main() 