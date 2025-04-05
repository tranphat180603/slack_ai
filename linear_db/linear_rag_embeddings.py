"""
Generate embeddings for Linear objects (issues, projects, comments) and store them in the PostgreSQL database.
Uses a flexible single-table approach with metadata for different object types.
"""
import os
import sys
import logging
import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values
import json
import tiktoken
import openai
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import argparse
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from datetime import datetime

# Fix imports when running as a script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from linear_db.db_pool import get_db_connection
else:
    from .db_pool import get_db_connection

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("linear_rag_embeddings")

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
    if not text or text.isspace():
        # Return zero vector for empty text (1536 dimensions for text-embedding-3-small)
        return [0.0] * 1536
    
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise

def store_issue_embedding(issue: Dict[str, Any]) -> bool:
    """
    Generate and store embedding for a Linear issue.
    
    Args:
        issue: Issue data from Linear API
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Skip if issue is None or not a dictionary
        if not issue or not isinstance(issue, dict):
            logger.warning(f"Invalid issue object: {type(issue)}")
            return False
            
        # Extract key fields for the embedding context
        issue_id = issue.get('id')
        issue_number = issue.get('number')
        title = issue.get('title', '')
        description = issue.get('description', '')
        
        # Check for required fields
        if not issue_id or not title:
            logger.warning(f"Missing required fields for issue {issue_id}")
            return False
        
        # Create embedding text - only title and description
        text = f"{title}\n\n{description}"
        
        # Generate embedding
        embedding = get_embedding(text)
        
        # Prepare metadata - safely access nested fields
        team = issue.get('team', {}) if isinstance(issue.get('team'), dict) else {}
        assignee = issue.get('assignee', {}) if isinstance(issue.get('assignee'), dict) else {}
        cycle = issue.get('cycle', {}) if isinstance(issue.get('cycle'), dict) else {}
        
        metadata = {
            'id': issue_id,
            'number': issue_number,
            'team_key': team.get('key'),
            'assignee_name': assignee.get('displayName'),
            'cycle_id': cycle.get('id'),
            'cycle_number': cycle.get('number')
        }
        
        # Store in database
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO linear_embeddings
                (object_type, object_id, text, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (object_type, object_id) 
                DO UPDATE SET 
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
                """, ('Issue', issue_id, text, embedding, json.dumps(metadata)))
                conn.commit()
        
        return True
    
    except Exception as e:
        logger.error(f"Error storing issue embedding for {issue.get('id') if isinstance(issue, dict) else 'unknown'}: {str(e)}")
        return False

def store_project_embedding(project: Dict[str, Any]) -> bool:
    """
    Generate and store embedding for a Linear project.
    
    Args:
        project: Project data from Linear API
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Skip if project is None or not a dictionary
        if not project or not isinstance(project, dict):
            logger.warning(f"Invalid project object: {type(project)}")
            return False
            
        # Extract key fields for the embedding context
        project_id = project.get('id')
        name = project.get('name', '')
        description = project.get('description', '')
        
        # Check for required fields
        if not project_id or not name:
            logger.warning(f"Missing required fields for project {project_id}")
            return False
        
        # Create embedding text - only name and description
        text = f"{name}\n\n{description}"
        
        # Generate embedding
        embedding = get_embedding(text)
        
        # Prepare metadata - safely access nested fields
        lead = project.get('lead', {}) if isinstance(project.get('lead'), dict) else {}
        team = project.get('team', {}) if isinstance(project.get('team'), dict) else {}
        
        metadata = {
            'id': project_id,
            'lead_name': lead.get('displayName'),
            'state': project.get('state'),
            'team_key': team.get('key')
        }
        
        # Store in database
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO linear_embeddings
                (object_type, object_id, text, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (object_type, object_id) 
                DO UPDATE SET 
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
                """, ('Project', project_id, text, embedding, json.dumps(metadata)))
                conn.commit()
        
        return True
    
    except Exception as e:
        logger.error(f"Error storing project embedding for {project.get('id') if isinstance(project, dict) else 'unknown'}: {str(e)}")
        return False

def store_comment_embedding(comment: Dict[str, Any], issue: Dict[str, Any] = None) -> bool:
    """
    Generate and store embedding for a Linear comment.
    
    Args:
        comment: Comment data from Linear API
        issue: Associated issue data (optional)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract key fields for the embedding context
        comment_id = None
        body = None

        if comment.get('id'):
            comment_id = comment.get('id')
        if comment.get('body') and comment.get('body') != '':
            body = comment.get('body')
        
        # Check for required fields
        if not comment_id or not body:
            logger.warning(f"Missing required fields for comment {comment_id}")
            return False
        
        # Get issue number if possible
        issue_number = None
        
        # If issue is provided, use it
        if issue and isinstance(issue, dict):
            issue_number = issue.get('number')
        # If comment has issue reference, use it
        elif comment.get('issue') and isinstance(comment.get('issue'), dict):
            issue_ref = comment.get('issue')
            issue_number = issue_ref.get('number')
        
        # Create embedding text - just the body
        text = body
        
        # Generate embedding
        embedding = get_embedding(text)
        
        # Prepare metadata - simplified as per requirements
        user = comment.get('user', {}) if isinstance(comment.get('user'), dict) else {}
        
        metadata = {
            'issue_number': issue_number,
            'creator_name': user.get('displayName')
        }
        
        # Store in database
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO linear_embeddings
                (object_type, object_id, text, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (object_type, object_id) 
                DO UPDATE SET 
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
                """, ('Comment', comment_id, text, embedding, json.dumps(metadata)))
                conn.commit()
        
        return True
    
    except Exception as e:
        logger.error(f"Error storing comment embedding for {comment.get('id')}: {str(e)}")
        return False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=60))
def rerank_results(query: str, candidates: List[Dict], limit: int = 5) -> List[Dict]:
    """
    Rerank search candidates using LLM relevance scoring.
    
    Args:
        query: The original search query
        candidates: List of candidate results from vector search
        limit: Maximum number of results to return after reranking
        
    Returns:
        Reranked list of results with relevance scores
    """
    if not candidates:
        return []
    
    # Prepare candidate texts with their IDs for the LLM
    candidate_texts = []
    for i, item in enumerate(candidates):
        try:
            # Safely extract fields with fallbacks
            object_type = item.get('object_type', 'Unknown')
            content = item.get('content', '')
            metadata = item.get('metadata', {}) if isinstance(item.get('metadata'), dict) else {}
            
            # Truncate content safely if it exists
            content_preview = content[:500] + "..." if content and len(content) > 500 else content
            
            # Format the candidate based on object type
            if object_type == 'Issue':
                issue_number = metadata.get('number', 'unknown')
                team_key = metadata.get('team_key', 'unknown')
                formatted_text = f"ISSUE #{issue_number} ({team_key}): {content_preview}"
            elif object_type == 'Project':
                project_name = item.get('project_id', 'unknown')
                team_key = metadata.get('team_key', 'unknown')
                formatted_text = f"PROJECT {project_name} ({team_key}): {content_preview}"
            elif object_type == 'Comment':
                issue_number = metadata.get('issue_number', 'unknown')
                creator = metadata.get('creator_name', 'unknown')
                formatted_text = f"COMMENT by {creator} on issue #{issue_number}: {content_preview}"
            else:
                formatted_text = f"{object_type}: {content_preview}"
                
            candidate_texts.append({
                "id": i,  # Index in the original candidates list
                "text": formatted_text
            })
        except Exception as e:
            logger.warning(f"Error formatting candidate {i} for reranking: {str(e)}")
            # Include a simplified version of the candidate
            candidate_texts.append({
                "id": i,
                "text": f"Item {i} (type: {item.get('object_type', 'unknown')}): {item.get('content', '')[:100]}..."
            })
    
    try:
        # Create the prompt for reranking
        system_prompt = """You are an expert search engine that evaluates the relevance of documents to a query.
For each candidate document, assign a relevance score from 0 to 10 where:
- 0 means completely irrelevant
- 10 means perfectly relevant and directly answers the query
Consider both keyword matches and semantic relevance.
Return ONLY a JSON array of objects with the format: [{"id": document_id, "score": relevance_score}]
"""
        
        # Prepare candidates as a formatted string
        candidates_text = "\n\n".join([f"Document {c['id']}: {c['text']}" for c in candidate_texts])
        
        user_prompt = f"""Query: {query}

Candidate documents:
{candidates_text}

Evaluate the relevance of each document to the query and return a JSON array of scores.
"""
        
        # Call the OpenAI API for reranking
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use a smaller model for efficiency
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        
        # Parse the response
        try:
            content = response.choices[0].message.content
            if not content or content.isspace():
                logger.warning("Empty response from reranker")
                return candidates[:limit]
                
            rankings = json.loads(content)
            
            # Handle different possible response formats
            rankings_list = None
            if isinstance(rankings, list):
                rankings_list = rankings
            elif isinstance(rankings, dict):
                # Try to find an array in the response
                for key, value in rankings.items():
                    if isinstance(value, list) and len(value) > 0:
                        rankings_list = value
                        break
            
            if not rankings_list:
                logger.warning("Could not find rankings array in response")
                return candidates[:limit]
            
            # Sort the candidates by the new relevance scores
            scored_candidates = []
            for ranking in rankings_list:
                # Safely extract ID and score with proper type checking
                if isinstance(ranking, dict) and "id" in ranking and "score" in ranking:
                    try:
                        doc_id = int(ranking["id"])
                        score = float(ranking["score"])
                        
                        if 0 <= doc_id < len(candidates):
                            candidate = candidates[doc_id].copy()
                            candidate["relevance_score"] = score
                            scored_candidates.append(candidate)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid ID or score in ranking: {ranking}")
            
            # If no valid scores were found, return original candidates
            if not scored_candidates:
                logger.warning("No valid scored candidates after reranking")
                return candidates[:limit]
                
            # Sort by relevance score (descending)
            reranked_results = sorted(scored_candidates, key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # Return the top results up to the limit
            return reranked_results[:limit]
            
        except json.JSONDecodeError as e:
            logger.warning(f"Error parsing reranker JSON response: {str(e)}")
            return candidates[:limit]
        except (KeyError, IndexError, ValueError, TypeError) as e:
            logger.warning(f"Error processing reranker response: {str(e)}")
            return candidates[:limit]
            
    except Exception as e:
        logger.warning(f"Reranking failed: {str(e)}")
        # Fall back to original ordering
        return candidates[:limit]

def semantic_search(query: str, limit: int = 5, use_reranker: bool = True, candidate_pool_size: int = 20, team_key: str = None, object_type: str = None) -> List[Dict]:
    """
    Semantic search to find Linear content similar to the query.
    Optionally applies a reranker to improve results.
    
    Args:
        query: Natural language query to search for
        limit: Maximum number of results to return
        use_reranker: Whether to apply the LLM reranker
        candidate_pool_size: Size of the initial candidate pool for reranking
        team_key: Optional filter by team key (e.g., 'ENG', 'OPS')
        object_type: Optional filter by object type ('Issue', 'Project', 'Comment')
        
    Returns:
        List of matching objects from the linear_embeddings table with:
          - object_type: Type of object ('Issue', 'Project', 'Comment')
          - object_id: ID of the object in Linear
          - content: Text content that matched the query
          - metadata: Object with metadata properties appropriate for the object type
          - similarity: Similarity score (0-1) with higher being better match
          - relevance_score: (if reranked) LLM-assigned relevance score (0-10)
    """
    try:
        # Generate embedding for the query text
        query_embedding = get_embedding(query)
        
        # Determine how many initial candidates to retrieve
        retrieval_limit = candidate_pool_size if use_reranker else limit
        
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Build WHERE clause using available indexes
                conditions = ["embedding IS NOT NULL"]
                params = [query_embedding]
                
                # Add team_key filter if provided (using team_key index)
                if team_key:
                    conditions.append("metadata->>'team_key' = %s")
                    params.append(team_key)
                
                # Add object_type filter if provided
                if object_type:
                    conditions.append("object_type = %s")
                    params.append(object_type)
                
                # Construct the final WHERE clause
                where_clause = " AND ".join(conditions)
                
                # Construct the full query with similarity search
                sql = f"""
                SELECT 
                    object_type, 
                    object_id, 
                    text, 
                    metadata,
                    1 - (embedding <=> CAST(%s AS vector)) AS similarity
                FROM 
                    linear_embeddings
                WHERE 
                    {where_clause}
                ORDER BY 
                    similarity DESC
                LIMIT %s
                """
                
                # Add the limit parameter
                params.append(retrieval_limit)
                
                # Log the query for debugging
                logger.debug(f"Executing search query with params: {params}")
                
                # Execute the query
                cur.execute(sql, params)
                results = cur.fetchall()
                
                # Format the results to include key metadata fields at the top level
                formatted_results = []
                for result in results:
                    item = dict(result)
                    # Convert metadata from JSON string to dict if needed
                    if isinstance(item['metadata'], str):
                        item['metadata'] = json.loads(item['metadata'])
                    
                    # Rename text field to content for clarity
                    item['content'] = item.pop('text')
                    
                    # Format result based on object type
                    if item['object_type'] == 'Issue':
                        # For Issues: id, number, team_key, assignee_name
                        item['issue_id'] = item['metadata'].get('id')
                        item['issue_number'] = item['metadata'].get('number')
                        item['team_key'] = item['metadata'].get('team_key')
                        item['assignee'] = item['metadata'].get('assignee_name')
                        item['cycle_number'] = item['metadata'].get('cycle_number')
                    
                    elif item['object_type'] == 'Project':
                        # For Projects: id, lead_name, state
                        item['project_id'] = item['metadata'].get('id')
                        item['lead'] = item['metadata'].get('lead_name')
                        item['state'] = item['metadata'].get('state')
                        item['team_key'] = item['metadata'].get('team_key')
                    
                    elif item['object_type'] == 'Comment':
                        # For Comments: issue_number, creator_name
                        item['issue_number'] = item['metadata'].get('issue_number')
                        item['creator'] = item['metadata'].get('creator_name')
                    
                    formatted_results.append(item)
                
                # Apply reranking if enabled and we have results
                if use_reranker and formatted_results:
                    logger.info(f"Applying reranker to {len(formatted_results)} candidates for query: '{query}'")
                    reranked_results = rerank_results(query, formatted_results, limit)
                    return reranked_results
                
                # Otherwise just return the vector similarity results
                return formatted_results[:limit]
    
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return []

def clear_embeddings(object_type: str = None):
    """
    Clear embeddings from the database.
    
    Args:
        object_type: Object type to clear (None for all)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if object_type:
                    cur.execute("DELETE FROM linear_embeddings WHERE object_type = %s", (object_type,))
                    logger.info(f"Cleared embeddings for {object_type}")
                else:
                    cur.execute("DELETE FROM linear_embeddings")
                    logger.info("Cleared all embeddings")
                conn.commit()
    except Exception as e:
        logger.error(f"Error clearing embeddings: {str(e)}")
        raise

def main():
    """Run the embedding generation process with CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate embeddings for Linear objects using single-table approach")
    parser.add_argument('--object-type', choices=['Issue', 'Project', 'Comment', 'all'], default='all',
                        help='Type of object to process (default: all)')
    parser.add_argument('--clear', action='store_true', 
                        help='Clear existing embeddings before processing')
    parser.add_argument('--test-search', action='store_true',
                        help='Run a test search after processing')
    parser.add_argument('--test-reranker', action='store_true',
                        help='Test the reranker with a search query')
    parser.add_argument('--limit', type=int, default=100,
                        help='Limit the number of objects to process')
    parser.add_argument('--search-query', type=str, default="How do I fix the API rate limit error?",
                        help='Query to use for test search')
    
    args = parser.parse_args()
    
    try:
        # Clear embeddings if requested
        if args.clear:
            if args.object_type == 'all':
                clear_embeddings()
                print("✓ Cleared all embeddings")
            else:
                clear_embeddings(args.object_type)
                print(f"✓ Cleared embeddings for {args.object_type}")
        
        # Here you would connect to the Linear API and fetch data
        # This is a placeholder for the actual API call
        
        print(f"To use this script, you need to connect to the Linear API and fetch data.")
        print(f"Once you have the data, you can use the following functions:")
        print(f"- store_issue_embedding(issue_data)")
        print(f"- store_project_embedding(project_data)")
        print(f"- store_comment_embedding(comment_data)")
        
        # Example of how to use these functions with the Linear API
        print("\nExample code to integrate with Linear API:")
        print("```python")
        print("from linear_db.linear_client import LinearClient")
        print("from linear_db.linear_rag_embeddings import store_issue_embedding")
        print("")
        print("# Initialize Linear client")
        print("linear = LinearClient(os.getenv('LINEAR_API_KEY'))")
        print("")
        print("# Fetch issues from Linear")
        print("issues = linear.getAllIssues('ENG')")
        print("")
        print("# Store embeddings for each issue")
        print("for issue in issues:")
        print("    store_issue_embedding(issue)")
        print("```")
        
        # Test search if requested
        if args.test_search or args.test_reranker:
            query = args.search_query
            print("\nTesting similarity search:")
            
            if args.test_reranker:
                print("Using vector search with reranking:")
                # Compare with and without reranker
                results_with_reranker = semantic_search(query, limit=5, use_reranker=True)
                results_no_reranker = semantic_search(query, limit=5, use_reranker=False)
                
                # Display results with reranker
                print(f"\nResults WITH reranker:")
                for i, result in enumerate(results_with_reranker):
                    print(f"\n{i+1}. {result['object_type']} (Relevance score: {result.get('relevance_score', 'N/A')}, Vector similarity: {result['similarity']:.3f})")
                    print(f"   ID: {result['object_id']}")
                    
                    if result['object_type'] == 'Issue':
                        print(f"   Issue #{result['issue_number']} - Team: {result['team_key']}")
                    
                    # Show a snippet of the text
                    text = result['content']
                    if len(text) > 100:
                        text = text[:100] + "..."
                    print(f"   Text: {text}")
                
                # Display results without reranker
                print(f"\nResults WITHOUT reranker (vector similarity only):")
                for i, result in enumerate(results_no_reranker):
                    print(f"\n{i+1}. {result['object_type']} (Vector similarity: {result['similarity']:.3f})")
                    print(f"   ID: {result['object_id']}")
                    
                    if result['object_type'] == 'Issue':
                        print(f"   Issue #{result['issue_number']} - Team: {result['team_key']}")
                    
                    # Show a snippet of the text
                    text = result['content']
                    if len(text) > 100:
                        text = text[:100] + "..."
                    print(f"   Text: {text}")
            else:
                # Standard vector search only
                results = semantic_search(query, limit=3, use_reranker=False)
                
                print(f"Query: '{query}'")
                print(f"Found {len(results)} similar objects:")
                
                for i, result in enumerate(results):
                    print(f"\n{i+1}. {result['object_type']} (Similarity: {result['similarity']:.3f})")
                    print(f"   ID: {result['object_id']}")
                    
                    if result['object_type'] == 'Issue':
                        meta = result['metadata']
                        print(f"   Issue #{meta.get('number')} - Team: {meta.get('team_key')}")
                    
                    # Show a snippet of the text
                    text = result['content']
                    if len(text) > 100:
                        text = text[:100] + "..."
                    print(f"   Text: {text}")
        
    except Exception as e:
        logger.error(f"Error in embedding generation: {str(e)}")
        raise

# Run the main function when executed directly
if __name__ == "__main__":
    main() 