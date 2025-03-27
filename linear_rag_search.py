"""
Search functionality for the Linear RAG system using the simplified schema.
Supports both vector similarity search and JSON-based filtering.
"""
import os
import logging
import json
import openai
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import traceback
from db_pool import DatabasePool, get_db_connection
import psycopg2
from tenacity import retry, stop_after_attempt, wait_exponential
import linear_rag_db_import

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("linear_rag_search")

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize database pool
DatabasePool.initialize(minconn=1, maxconn=10)

def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    database = os.environ.get("POSTGRES_DB", "linear_rag")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "phatdeptrai123")
    
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

def get_embedding_for_query(query: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Get embedding vector for a text query using OpenAI API.
    
    Args:
        query: The query text to embed
        model: The embedding model to use
        
    Returns:
        Embedding vector as a list of floats
    """
    if not query or query.isspace():
        # Return zero vector for empty text (1536 dimensions for text-embedding-3-small)
        return [0.0] * 1536
    
    try:
        response = client.embeddings.create(
            model=model,
            input=query
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding for query: {str(e)}")
        raise

def search_issues(
    query: Optional[str] = None,
    team_key: Optional[str] = None,
    cycle_name: Optional[str] = None,
    assignee_name: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search issues using vector similarity and optional filters.
    If query is None, performs only SQL search without vector similarity.
    
    Args:
        query: The search query (optional)
        team_key: Optional team key filter
        cycle_name: Optional cycle name filter
        assignee_name: Optional assignee name filter
        limit: Maximum number of results to return (required)
        
    Returns:
        List of matching issues with their content and metadata
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # SQL query depends on whether we have a search query or not
                if query:
                    # Get embedding for the query
                    query_embedding = get_embedding_for_query(query)
                    
                    # Build the query with vector similarity
                    sql_query = """
                    SELECT 
                        e.issue_id,
                        e.content,
                        e.data,
                        1 - (e.embedding <=> %s::vector) as similarity_score,
                        i.full_context
                    FROM 
                        embeddings_simplified e
                    JOIN
                        issues_simplified i ON e.issue_id = i.id
                    WHERE 1=1
                    AND e.data->>'title' NOT ILIKE %s
                    """
                    params = [query_embedding, '%call%']
                    
                    # Add filters if provided
                    if team_key:
                        sql_query += " AND e.data->'team'->>'key' = %s"
                        params.append(team_key)
                    
                    if cycle_name:
                        sql_query += " AND e.data->'cycle'->>'name' = %s"
                        params.append(cycle_name)
                    
                    if assignee_name:
                        sql_query += " AND e.data->'assignee'->>'name' = %s"
                        params.append(assignee_name)
                    
                    # Order by similarity and limit results
                    sql_query += """
                    ORDER BY 
                        similarity_score DESC
                    LIMIT %s
                    """
                    params.append(limit)
                else:
                    # SQL-only search without vector similarity
                    sql_query = """
                    SELECT 
                        i.id as issue_id,
                        '' as content,
                        i.data,
                        1.0 as similarity_score,
                        i.full_context
                    FROM 
                        issues_simplified i
                    WHERE 1=1
                    AND i.data->>'title' NOT ILIKE %s
                    """
                    params = ['%call%', '%meeting%']
                    
                    # Add filters if provided
                    if team_key:
                        sql_query += " AND i.data->'team'->>'key' = %s"
                        params.append(team_key)
                    
                    if cycle_name:
                        sql_query += " AND i.data->'cycle'->>'name' = %s"
                        params.append(cycle_name)
                    
                    if assignee_name:
                        sql_query += " AND i.data->'assignee'->>'name' = %s"
                        params.append(assignee_name)
                    
                    # Order by recent issues and limit results
                    sql_query += """
                    ORDER BY 
                        i.created_at DESC
                    LIMIT %s
                    """
                    params.append(limit)
                
                # Execute the query
                cur.execute(sql_query, params)
                results = cur.fetchall()
                
                # Format the results
                formatted_results = []
                for result in results:
                    issue_id, content, data_json, similarity_score, full_context = result
                    
                    # Parse JSON data
                    try:
                        issue_data = json.loads(data_json) if isinstance(data_json, str) else data_json
                    except json.JSONDecodeError:
                        logger.error(f"Error parsing JSON for issue {issue_id}")
                        issue_data = {}
                    
                    # Format the result
                    formatted_result = {
                        "issue_id": issue_id,
                        "matched_content": content,
                        "similarity_score": similarity_score,
                        "title": issue_data.get("title", ""),
                        "team": issue_data.get("team", {}).get("key", ""),
                        "cycle": issue_data.get("cycle", {}).get("name", ""),
                        "assignee": issue_data.get("assignee", {}).get("name", ""),
                        "state": issue_data.get("state", ""),
                        "full_context": full_context,
                        "data": issue_data
                    }
                    
                    formatted_results.append(formatted_result)
                
                return formatted_results
                
    except Exception as e:
        logger.error(f"Error searching issues: {str(e)}")
        raise

def find_team_cycle_issues(
    team_key: str,
    cycle_name: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Find issues for a specific team and optionally cycle without semantic search.
    
    Args:
        team_key: Team key to filter by
        cycle_name: Optional cycle name to filter by
        limit: Maximum number of results to return
        
    Returns:
        List of matching issues
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Start building the query
                sql_query = """
                SELECT 
                    id,
                    data,
                    full_context
                FROM 
                    issues_simplified
                WHERE 
                    data->'team'->>'key' = %s
                """
                params = [team_key]
                
                # Add cycle filter if provided
                if cycle_name:
                    sql_query += " AND data->'cycle'->>'name' = %s"
                    params.append(cycle_name)
                else:
                    # If no cycle provided, find the latest cycle by looking at created_at dates
                    sql_query += " ORDER BY created_at DESC"
                
                # Add limit
                sql_query += " LIMIT %s"
                params.append(limit)
                
                # Execute the query
                cur.execute(sql_query, params)
                results = cur.fetchall()
                
                # Format the results
                formatted_results = []
                for result in results:
                    issue_id, data_json, full_context = result
                    
                    # Parse JSON data
                    try:
                        issue_data = json.loads(data_json) if isinstance(data_json, str) else data_json
                    except json.JSONDecodeError:
                        logger.error(f"Error parsing JSON for issue {issue_id}")
                        issue_data = {}
                    
                    # Format the result
                    formatted_result = {
                        "issue_id": issue_id,
                        "title": issue_data.get("title", ""),
                        "team": issue_data.get("team", {}).get("key", ""),
                        "cycle": issue_data.get("cycle", {}).get("name", ""),
                        "assignee": issue_data.get("assignee", {}).get("name", ""),
                        "state": issue_data.get("state", ""),
                        "priority": issue_data.get("priority"),
                        "full_context": full_context,
                        "data": issue_data
                    }
                    
                    formatted_results.append(formatted_result)
                
                return formatted_results
                
    except Exception as e:
        logger.error(f"Error finding team issues: {str(e)}")
        raise

def get_available_teams_and_cycles() -> Dict[str, Any]:
    """
    Get a list of available teams and cycles in the database.
    
    Returns:
        Dictionary with teams and cycles information
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get unique teams
                cur.execute("""
                SELECT DISTINCT data->'team'->>'key', data->'team'->>'name'
                FROM issues_simplified
                WHERE data->'team'->>'key' IS NOT NULL
                ORDER BY data->'team'->>'key'
                """)
                
                teams = [{"key": row[0], "name": row[1]} for row in cur.fetchall()]
                
                # Get unique cycles
                cur.execute("""
                SELECT DISTINCT data->'cycle'->>'name', COUNT(*)
                FROM issues_simplified
                WHERE data->'cycle'->>'name' IS NOT NULL
                GROUP BY data->'cycle'->>'name'
                ORDER BY COUNT(*) DESC
                """)
                
                cycles = [{"name": row[0], "issue_count": row[1]} for row in cur.fetchall()]
                
                # Get team-cycle combinations
                cur.execute("""
                SELECT 
                    data->'team'->>'key', 
                    data->'cycle'->>'name',
                    COUNT(*)
                FROM 
                    issues_simplified
                WHERE 
                    data->'team'->>'key' IS NOT NULL AND
                    data->'cycle'->>'name' IS NOT NULL
                GROUP BY 
                    data->'team'->>'key', 
                    data->'cycle'->>'name'
                ORDER BY 
                    data->'team'->>'key',
                    COUNT(*) DESC
                """)
                
                team_cycles = {}
                for row in cur.fetchall():
                    team_key, cycle_name, count = row
                    if team_key not in team_cycles:
                        team_cycles[team_key] = []
                    team_cycles[team_key].append({"name": cycle_name, "issue_count": count})
                
                return {
                    "teams": teams,
                    "cycles": cycles,
                    "team_cycles": team_cycles
                }
                
    except Exception as e:
        logger.error(f"Error getting teams and cycles: {str(e)}")
        raise

def structure_results(results: List[Dict[str, Any]], structure_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Restructure the query results based on the specified output structure.
    """
    structured_results = []
    
    # If structure_spec is empty, return original results
    if not structure_spec:
        return results
    
    # If results are empty, return empty list
    if not results:
        return []
    
    # DEBUGGING - Log sample result structure
    if results and len(results) > 0:
        logger.debug(f"Sample result structure: {results[0]}")
    
    # If structure_spec is just a list of field names
    if isinstance(structure_spec, list):
        for result in results:
            structured_result = {}
            for field in structure_spec:
                # Handle nested paths with dot notation
                if '.' in field and not isinstance(field, dict):
                    parts = field.split('.')
                    value = result
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        elif isinstance(value, dict) and 'data' in value and isinstance(value['data'], dict) and part in value['data']:
                            value = value['data'][part]
                        else:
                            value = None
                            break
                    structured_result[field] = value
                elif isinstance(field, dict) and 'field' in field:
                    field_name = field['field']
                    alias = field.get('alias', field_name.split('.')[-1])
                    format_fn = field.get('format')
                    
                    # Get the value
                    value = result
                    for part in field_name.split('.'):
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        elif isinstance(value, dict) and 'data' in value and isinstance(value['data'], dict) and part in value['data']:
                            value = value['data'][part]
                        else:
                            value = None
                            break
                    
                    # Apply formatting if specified
                    if format_fn and value is not None:
                        if format_fn == 'date':
                            try:
                                from datetime import datetime
                                if isinstance(value, str):
                                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                                    value = dt.strftime('%Y-%m-%d')
                            except:
                                pass
                        elif format_fn == 'capitalize':
                            if isinstance(value, str):
                                value = value.capitalize()
                    
                    structured_result[alias] = value
                else:
                    # Direct field access - check both original field name and simplified version
                    if field in result:
                        structured_result[field] = result[field]
                    elif field.split('->')[-1].strip("'") in result:
                        # Handle SQL column alias case
                        column_alias = field.split('->')[-1].strip("'")
                        structured_result[field] = result[column_alias]
                    elif 'data' in result and isinstance(result['data'], dict) and field in result['data']:
                        structured_result[field] = result['data'][field]
            
            structured_results.append(structured_result)
    
    # If structure_spec is a dictionary with field mappings
    elif isinstance(structure_spec, dict):
        for result in results:
            structured_result = {}
            
            for output_field, source_field in structure_spec.items():
                # Handle simple string mapping
                if isinstance(source_field, str):
                    # Handle nested paths with dot notation
                    if '.' in source_field:
                        parts = source_field.split('.')
                        value = result
                        for part in parts:
                            if isinstance(value, dict) and part in value:
                                value = value[part]
                            elif isinstance(value, dict) and 'data' in value and isinstance(value['data'], dict) and part in value['data']:
                                value = value['data'][part]
                            else:
                                value = None
                                break
                        structured_result[output_field] = value
                    else:
                        # Direct field access - try several possible keys
                        if source_field in result:
                            structured_result[output_field] = result[source_field]
                        # Extract the simple field name from the JSON path for SQL column alias cases
                        elif '->' in source_field and source_field.split('->')[-1].strip("'") in result:
                            column_alias = source_field.split('->')[-1].strip("'")
                            structured_result[output_field] = result[column_alias]
                        # Try looking for the field without the "data->" prefix
                        elif source_field.startswith('data->') and source_field[6:] in result:
                            structured_result[output_field] = result[source_field[6:]]
                        # Try looking for just the last part of the field path
                        elif source_field.split('->')[-1].strip("'") in result:
                            structured_result[output_field] = result[source_field.split('->')[-1].strip("'")]
                        elif 'data' in result and isinstance(result['data'], dict) and source_field in result['data']:
                            structured_result[output_field] = result['data'][source_field]
                        else:
                            # Field not found, set to None
                            structured_result[output_field] = None
                # Handle complex field mapping with formatting
                elif isinstance(source_field, dict) and 'field' in source_field:
                    field_name = source_field['field']
                    format_fn = source_field.get('format')
                    
                    # Get the value - try direct access first
                    value = None
                    if field_name in result:
                        value = result[field_name]
                    # Try SQL column alias
                    elif '->' in field_name and field_name.split('->')[-1].strip("'") in result:
                        column_alias = field_name.split('->')[-1].strip("'")
                        value = result[column_alias]
                    # If not found directly, try nested path
                    else:
                        value = result
                        for part in field_name.split('.'):
                            if isinstance(value, dict) and part in value:
                                value = value[part]
                            elif isinstance(value, dict) and 'data' in value and isinstance(value['data'], dict) and part in value['data']:
                                value = value['data'][part]
                            else:
                                value = None
                                break
                    
                    # Apply formatting if specified
                    if format_fn and value is not None:
                        if format_fn == 'date':
                            try:
                                from datetime import datetime
                                if isinstance(value, str):
                                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                                    value = dt.strftime('%Y-%m-%d')
                            except:
                                pass
                        elif format_fn == 'capitalize':
                            if isinstance(value, str):
                                value = value.capitalize()
                    
                    structured_result[output_field] = value
            
            structured_results.append(structured_result)
    
    return structured_results

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=60))
def advanced_search(query_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute an advanced search query using the structured query specification.
    
    Args:
        query_spec: Dictionary containing:
            - fields: List of fields to select
            - returned_fields: Structure of output data
            - filters (optional): List of filter conditions
            - grouping (optional): Field to group by
            - aggregations (optional): List of aggregation specs
            - sorting (optional): Sort specification
            - limit: Maximum number of results
            - distinct_on (optional): Field to get distinct values on
            - per_group_limit (optional): Max number of results per group
            - semantic_search (optional): A text query to compute vector similarity
            
    Returns:
        Dictionary containing:
            "count": number of records in the results
            "results": list of dictionaries, each representing a row
    """
    # Add default filter to exclude titles containing 'call'
    filters = query_spec.get("filters", [])
    filters.append({
        "field": "title",
        "operator": "NOT ILIKE",
        "value": "%call%"
    })
    query_spec["filters"] = filters
    
    # Validate required parameters
    if 'fields' not in query_spec:
        raise ValueError("'fields' parameter is required - specify which fields to select from the database")
    if 'returned_fields' not in query_spec:
        raise ValueError("'returned_fields' parameter is required - specify the structure of the output data")
    
    # Extract top-level options
    grouping = query_spec.get("grouping")
    aggregations = query_spec.get("aggregations", [])
    sorting = query_spec.get("sorting")  # e.g. {"field": "priority", "direction": "DESC"}
    limit = query_spec.get("limit", 50)
    debug = query_spec.get("debug", False)
    fields = query_spec["fields"]
    returned_fields = query_spec["returned_fields"]
    distinct_on = query_spec.get("distinct_on")
    per_group_limit = query_spec.get("per_group_limit")
    
    # NEW: handle semantic_search
    semantic_query = query_spec.get("semantic_search")  # e.g. "X Agent"
    embedding_vector = None
    
    # We'll store the extra SELECT part for similarity if needed
    similarity_select_part = ""
    
    if semantic_query:
        # Suppose you have a function like "get_embedding_for_query"
        embedding_vector = get_embedding_for_query(semantic_query)
        # We'll build e.g. "1 - (i.embedding <=> %s::vector) as similarity_score"
        similarity_select_part = "1 - (i.embedding <=> %s::vector) as similarity_score"
        # If user didn't specify sorting, default to ordering by similarity DESC
        if not sorting:
            sorting = {"field": "similarity_score", "direction": "DESC"}
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                params: List[Any] = []
                
                # ---------------------------------------------------------
                # 1) Build the base query
                # ---------------------------------------------------------
                if grouping:
                    #
                    # GROUPING QUERY
                    #
                    formatted_grouping = format_json_path_for_sql(grouping)
                    select_parts = [f"{formatted_grouping} as group_field"]
                    
                    # Add each aggregation to the SELECT
                    for agg in aggregations:
                        agg_type = agg.get("type", "count").lower()
                        field = agg.get("field", "*")
                        alias = agg.get("alias", f"{agg_type}_{field.replace('*','all').replace('->','_')}")
                        condition = agg.get("condition")
                        
                        if field == "*":
                            formatted_field = "*"
                        else:
                            formatted_field = format_json_path_for_sql(field)
                        
                        if condition:
                            if not isinstance(condition, dict):
                                raise ValueError("Aggregation conditions must use object format.")
                            cond_field = condition.get("field")
                            cond_operator = condition.get("operator", "=")
                            cond_value = condition.get("value")
                            
                            if not cond_field or cond_value is None:
                                raise ValueError("Aggregation condition must specify field and value")
                            
                            formatted_cond_field = format_json_path_for_sql(cond_field)
                            
                            # e.g. priority IN [1,2]
                            if cond_operator.upper() == "IN" and isinstance(cond_value, list):
                                condition_str = f"{formatted_cond_field} = ANY(%s)"
                                params.append(cond_value)
                            else:
                                condition_str = f"{formatted_cond_field} {cond_operator} %s"
                                params.append(cond_value)
                            
                            if agg_type == "count_distinct":
                                agg_expr = f"COUNT(DISTINCT CASE WHEN {condition_str} THEN {formatted_cond_field} END)"
                            elif agg_type == "count" and field == "*":
                                agg_expr = f"COUNT(CASE WHEN {condition_str} THEN 1 END)"
                            else:
                                agg_expr = f"{agg_type.upper()}(CASE WHEN {condition_str} THEN {formatted_field} END)"
                        else:
                            if agg_type == "count_distinct":
                                agg_expr = f"COUNT(DISTINCT {formatted_field})"
                            else:
                                agg_expr = f"{agg_type.upper()}({formatted_field})"
                    
                        if "_" in alias:
                            select_parts.append(f"{agg_expr} as {alias}")
                        elif " " in alias:
                            select_parts.append(f"{agg_expr} as \"{alias}\"")
                        else:
                            select_parts.append(f"{agg_expr} as {alias}")
                    
                    # Base grouping query
                    sql_query = "SELECT " + ", ".join(select_parts) + " FROM embeddings_simplified i WHERE 1=1"
                    
                    # If doing semantic search + grouping: 
                    # There's no typical "average similarity" or anything out of the box. 
                    # But if you need it, you'd do something like an aggregation on that. 
                    # We'll skip it for simplicity because it's unusual to group semantic similarity.
                    
                else:
                    #
                    # NON-GROUPING QUERIES
                    #
                    # We'll incorporate the user "fields" plus the optional "similarity_select_part"
                    base_select_parts = []
                    
                    # If distinct_on + per_group_limit => special window function approach
                    if distinct_on:
                        formatted_distinct = format_json_path_for_sql(distinct_on)
                        
                        if per_group_limit:
                            # We'll build a CTE with row_number partition
                            for f in fields:
                                f_alias = f.split('->')[-1].strip("'")
                                base_select_parts.append(f"{format_json_path_for_sql(f)} as {f_alias}")
                            
                            # If we have semantic search, add it to the SELECT
                            if similarity_select_part:
                                base_select_parts.append(similarity_select_part)
                            
                            # Sorting for the window function
                            if sorting:
                                sort_field = sorting.get("field")
                                sort_direction = sorting.get("direction", "ASC")
                                formatted_sort = "similarity_score" if sort_field == "similarity_score" else format_json_path_for_sql(sort_field)
                            else:
                                sort_field = "created_at"
                                sort_direction = "DESC"
                                formatted_sort = "i.created_at"
                            
                            # Build where parts
                            where_parts = ["1=1"]
                            filter_params = []
                            for flt in filters:
                                ffield = flt.get("field")
                                fop = flt.get("operator", "=")
                                fvalue = flt.get("value")
                                if ffield and fvalue is not None:
                                    ffmt = format_json_path_for_sql(ffield)
                                    if fop.upper() == "IN" and isinstance(fvalue, list):
                                        where_parts.append(f"{ffmt} = ANY(%s)")
                                        filter_params.append(fvalue)
                                    elif fop.upper() == "= ANY":
                                        where_parts.append(f"{ffmt} = ANY(%s)")
                                        filter_params.append(fvalue)
                                    else:
                                        where_parts.append(f"{ffmt} {fop} %s")
                                        filter_params.append(fvalue)
                            
                            # Build final CTE
                            sql_query = f"""
                            WITH ranked AS (
                                SELECT 
                                    {', '.join(base_select_parts)},
                                    ROW_NUMBER() OVER (
                                        PARTITION BY {formatted_distinct}
                                        ORDER BY {formatted_sort} {sort_direction}
                                    ) as row_num
                                FROM embeddings_simplified i
                                WHERE {' AND '.join(where_parts)}
                            )
                            SELECT {', '.join(x.split('->')[-1].strip("'") for x in fields)}
                            {', similarity_score' if similarity_select_part else ''}
                            FROM ranked
                            WHERE row_num <= %s
                            LIMIT %s
                            """
                            
                            # If we have semantic search, push embedding_vector at the front
                            if embedding_vector is not None:
                                params.append(embedding_vector)
                            params.extend(filter_params)
                            params.append(per_group_limit)
                            params.append(limit)
                            
                        else:
                            # DISTINCT ON but NO per_group_limit
                            distinct_parts = [f"DISTINCT ON ({formatted_distinct})"]
                            
                            # user fields
                            for f in fields:
                                alias = f.split('->')[-1].strip("'")
                                distinct_parts.append(f"{format_json_path_for_sql(f)} as {alias}")
                            
                            # add similarity if needed
                            if similarity_select_part:
                                distinct_parts.append(similarity_select_part)
                            
                            sql_query = "SELECT " + " ".join(distinct_parts) + " FROM embeddings_simplified i WHERE 1=1"
                    else:
                        # Regular query
                        for f in fields:
                            alias = f.split('->')[-1].strip("'")
                            base_select_parts.append(f"{format_json_path_for_sql(f)} as {alias}")
                        
                        # add similarity if needed
                        if similarity_select_part:
                            base_select_parts.append(similarity_select_part)
                        
                        sql_query = "SELECT " + ", ".join(base_select_parts) + " FROM embeddings_simplified i WHERE 1=1"
                
                # ---------------------------------------------------------
                # 2) Add filters if we haven't inlined them (grouping or normal queries)
                # ---------------------------------------------------------
                
                # If not grouping or distinct_on+per_group_limit, we add filters below
                if not grouping and not (distinct_on and per_group_limit):
                    for fcond in filters:
                        ffield = fcond.get("field")
                        fop = fcond.get("operator", "=")
                        fvalue = fcond.get("value")
                        if ffield and fvalue is not None:
                            fmt = format_json_path_for_sql(ffield)
                            if fop.upper() == "IN" and isinstance(fvalue, list):
                                sql_query += f" AND {fmt} = ANY(%s)"
                                params.append(fvalue)
                            elif fop.upper() == "= ANY":
                                sql_query += f" AND {fmt} = ANY(%s)"
                                params.append(fvalue)
                            else:
                                sql_query += f" AND {fmt} {fop} %s"
                                params.append(fvalue)
                
                # If grouping, also add the filters here (so they aren't skipped)
                if grouping:
                    for fcond in filters:
                        ffield = fcond.get("field")
                        fop = fcond.get("operator", "=")
                        fvalue = fcond.get("value")
                        if ffield and fvalue is not None:
                            fmt = format_json_path_for_sql(ffield)
                            if fop.upper() == "IN" and isinstance(fvalue, list):
                                sql_query += f" AND {fmt} = ANY(%s)"
                                params.append(fvalue)
                            elif fop.upper() == "= ANY":
                                sql_query += f" AND {fmt} = ANY(%s)"
                                params.append(fvalue)
                            else:
                                sql_query += f" AND {fmt} {fop} %s"
                                params.append(fvalue)
                    sql_query += f" GROUP BY {formatted_grouping}"
                
                # If we have semantic search but haven't used the special CTE path, we need to insert the embedding vector
                # at the front of our param list for the "similarity_score" expression. 
                # We'll do so here, if we haven't appended it already.
                if semantic_query and not (distinct_on and per_group_limit):
                    # The query references %s::vector for i.embedding <=> %s::vector
                    # So we need embedding_vector as a param. We'll put it at the end or front.
                    params.insert(0, embedding_vector)  # or append
                    # Then we must adjust the reference in the SELECT to use that param index 
                    # (But in Postgres, it's fine as placeholders are matched sequentially.)
                
                # ---------------------------------------------------------
                # 3) ORDER BY if not in the special distinct_on+per_group_limit
                # ---------------------------------------------------------
                if sorting and not (distinct_on and per_group_limit):
                    sort_field = sorting.get("field")
                    sort_direction = sorting.get("direction", "ASC")
                    
                    # If user wants to sort by "similarity_score", handle that
                    if sort_field == "similarity_score":
                        sql_query += f" ORDER BY similarity_score {sort_direction}"
                    elif sort_field in [
                        "total_issues", "urgent_issues", "high_priority_issues",
                        "completed_issues", "in_progress", "high_priority", 
                        "total_count", "priority_issues"
                    ]:
                        # Sorting by an aggregation alias
                        sql_query += f" ORDER BY {sort_field} {sort_direction}"
                    elif grouping:
                        # Possibly want to sort by group_field
                        if sort_field == grouping.split('->')[-1]:
                            sql_query += f" ORDER BY group_field {sort_direction}"
                        else:
                            sql_query += f" ORDER BY {sort_field} {sort_direction}"
                    else:
                        # Normal JSON field
                        formatted_sort = format_json_path_for_sql(sort_field)
                        sql_query += f" ORDER BY {formatted_sort} {sort_direction}"
                
                # ---------------------------------------------------------
                # 4) LIMIT if not in distinct_on+per_group_limit branch
                # ---------------------------------------------------------
                if not (distinct_on and per_group_limit):
                    sql_query += " LIMIT %s"
                    params.append(limit)
                
                # Debug logs
                if debug:
                    logger.info(f"Executing SQL query: {sql_query}")
                    logger.info(f"With parameters: {params}")
                
                # Execute
                cur.execute(sql_query, params)
                rows = cur.fetchall()
                
                # ---------------------------------------------------------
                # 5) Format & structure the results
                # ---------------------------------------------------------
                formatted_results = []
                for row in rows:
                    row_dict = {}
                    for idx, colinfo in enumerate(cur.description):
                        row_dict[colinfo.name] = row[idx]
                    formatted_results.append(row_dict)
                
                # Map columns -> your returned_fields
                structured_results = structure_results(formatted_results, returned_fields)
                
                return {
                    "count": len(structured_results),
                    "results": structured_results
                }
            
            except Exception as e:
                logger.error(f"Error in advanced search: {str(e)}")
                logger.error(traceback.format_exc())
                raise

def format_json_path_for_sql(path: str) -> str:
    """
    Your existing function that transforms "team->key" into 
    i.data->'team'->>'key', etc. Also handle integer cast for 'priority'.
    """
    # ... (unchanged from your original approach) ...
    # E.g.:
    if not path:
        return path
        
    table_prefix = "i."
    is_priority = path.lower() == 'priority' or path.endswith('->priority')
    
    if '->' not in path:
        # top-level key in data
        if is_priority:
            return f"({table_prefix}data->>'{path}')::integer"
        else:
            return f"{table_prefix}data->>'{path}'"
    
    parts = path.split('->')
    # Ensure "data" is first
    if parts[0] != 'data':
        parts.insert(0, 'data')
    
    parts[0] = table_prefix + parts[0]
    
    if len(parts) == 2:
        if is_priority:
            return f"({parts[0]}->>'{parts[1].strip()}')::integer"
        return f"{parts[0]}->>'{parts[1].strip()}'"
    else:
        # e.g. data->assignee->name
        middle = [f"'{p.strip()}'" for p in parts[1:-1]]
        result = parts[0]
        if middle:
            result += "->" + "->".join(middle)
        last = parts[-1].strip()
        if is_priority:
            return f"({result}->>'{last}')::integer"
        else:
            return f"{result}->>'{last}'"

def check_first_embedding():
    """Check the first entry in the embeddings_simplified table."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        sql_query = """
        SELECT 
            e.issue_id,
            e.embedding,
            e.content,
            e.data,
            i.full_context
        FROM 
            embeddings_simplified e
        JOIN
            issues_simplified i ON e.issue_id = i.id
        LIMIT 1
        """
        
        cur.execute(sql_query)
        result = cur.fetchone()
        
        if result:
            issue_id, embedding, content, data, full_context = result
            print("\n========== First Entry in Embeddings Table ==========")
            print(f"Issue ID: {issue_id}")
            print(f"Content: {content[:200]}..." if content else "Content: None")
            try:
                data_json = json.loads(data) if isinstance(data, str) else data
                print(f"Data: {json.dumps(data_json, indent=2)}")
            except (json.JSONDecodeError, TypeError):
                print(f"Data (raw): {data}")
            print(f"Full Context: {full_context[:200]}..." if full_context else "Full Context: None")
            print("=" * 50)
        else:
            print("\nNo entries found in embeddings_simplified table")
            
    except Exception as e:
        print(f"Error checking first embedding: {str(e)}")
        traceback.print_exc()
    finally:
        cur.close()
        conn.close()

def main():
    """Run test cases for the advanced_search function."""
    try:
        # Check first embedding entry
        check_first_embedding()
        result = linear_rag_db_import.check_existing_data()
        print(f"embeddings row count: {result['embeddings_count']}")
        
        print("\nRunning test cases for advanced_search function...\n")
        
        # Test Case 1: Basic semantic search
        print("\n========== TEST CASE 1: Basic Semantic Search ==========")
        test_case_1 = {
            "semantic_search": "database migration",
            "fields": ["title", "description", "team->key", "state"],
            "returned_fields": {
                "Title": "title",
                "Description": "description",
                "Team": "team->key",
                "Status": "state"
            },
            "limit": 2
        }
        print(f"Query: {json.dumps(test_case_1, indent=2)}")
        results_1 = advanced_search(test_case_1)
        print(f"Found {results_1['count']} results")
        print("First result:")
        print(json.dumps(results_1['results'][0], indent=2, default=str))
        
        # Test Case 2: Filter by team and state
        print("\n========== TEST CASE 2: Filter by Team and State ==========")
        test_case_2 = {
            "filters": [
                {"field": "team->key", "operator": "=", "value": "ENG"},
                {"field": "state", "operator": "=", "value": "In Progress"}
            ],
            "fields": ["title", "team->key", "state", "assignee->name"],
            "returned_fields": {
                "Title": "title",
                "Team": "team->key",
                "Status": "state",
                "Assignee": "assignee->name"
            },
            "limit": 50
        }
        print(f"Query: {json.dumps(test_case_2, indent=2)}")
        results_2 = advanced_search(test_case_2)
        print(f"Found {results_2['count']} results")
        if results_2['count'] > 0:
            print("First result:")
            print(json.dumps(results_2['results'][0], indent=2, default=str))
        
        # Test Case 3: Select specific fields
        print("\n========== TEST CASE 3: Select Specific Fields ==========")
        test_case_3 = {
            "filters": [
                {"field": "team->key", "operator": "=", "value": "ENG"}
            ],
            "fields": ["title", "assignee->name", "state", "priority"],
            "returned_fields": {
                "Task": "title",
                "Owner": "assignee->name",
                "Status": "state",
                "Priority": "priority"
            },
            "limit": 3
        }
        print(f"Query: {json.dumps(test_case_3, indent=2)}")
        results_3 = advanced_search(test_case_3)
        print(f"Found {results_3['count']} results")
        print("Results (limited to 3):")
        print(json.dumps(results_3['results'], indent=2, default=str))
        
        # Test Case 4: Grouping by assignee
        print("\n========== TEST CASE 4: Grouping by Assignee ==========")
        test_case_4 = {
            "filters": [
                {"field": "team->key", "operator": "=", "value": "ENG"},
                {"field": "cycle->name", "operator": "=", "value": "Cycle 40"}
            ],
            "grouping": "assignee->name",
            "aggregations": [
                {"type": "count", "field": "*", "alias": "total_issues"}
            ],
            "sorting": {"field": "total_issues", "direction": "DESC"},
            "fields": ["assignee->name"],
            "returned_fields": {
                "Assignee": "group_field",
                "Total Issues": "total_issues"
            },
            "limit": 5
        }
        print(f"Query: {json.dumps(test_case_4, indent=2)}")
        results_4 = advanced_search(test_case_4)
        print(f"Found {results_4['count']} results")
        print("Results (limited to 5 groups):")
        print(json.dumps(results_4['results'], indent=2, default=str))
        
        # Test Case 5: Using returned_fields to structure output
        print("\n========== TEST CASE 5: Structure Output with returned_fields (dictionary) ==========")
        test_case_5 =     {
        "fields": ["id", "title", "state", "priority", "assignee->name"],
        "returned_fields": {
            "ID": "id",
            "Title": "title",
            "Status": "state",
            "Priority": "priority",
            "Assignee": "assignee->name"
        },
        "filters": [
            {"field": "team->key", "operator": "=", "value": "ENG"},
            {"field": "priority", "operator": "<=", "value": 2},
        ],
        "order_by": {"field": "priority", "direction": "ASC"},
        "limit": 10
        }
        print(f"Query: {json.dumps(test_case_5, indent=2)}")
        results_5 = advanced_search(test_case_5)
        print(f"Found {results_5['count']} results")
        print("Results (limited to 3):")
        print(json.dumps(results_5['results'], indent=2, default=str))
        
        # Test Case 6: Using returned_fields with list format
        print("\n========== TEST CASE 6: Structure Output with returned_fields (list) ==========")
        test_case_6 = {
            "fields": ["title", "team->key", "state", "assignee->name"],
            "filters": [
                {"field": "team->key", "operator": "=", "value": "ENG"}
            ],
            "returned_fields": {
                "Title": "title",
                "Assignee": "assignee->name",
                "Status": "state",
                "Priority": "priority"
            },
            "limit": 3
        }
        print(f"Query: {json.dumps(test_case_6, indent=2)}")
        results_6 = advanced_search(test_case_6)
        print(f"Found {results_6['count']} results")
        print("Results (limited to 3):")
        print(json.dumps(results_6['results'], indent=2, default=str))
        
        # Test Case 7: Complex query with semantic search, filters and returned_fields
        print("\n========== TEST CASE 7: Complex Query ==========")
        test_case_7 = {
            "semantic_search": "X Agent",
            "filters": [
                {"field": "state", "operator": "!=", "value": "Done"}
            ],
            "fields": ["title", "team->key", "state", "assignee->name", "id"],
            "returned_fields": {
                "ID": "id",
                "Title": "title",
                "Team": "team->key",
                "State": {"field": "state", "format": "capitalize"},
                "Assignee": "assignee->name"
            },
            "limit": 15
        }
        print(f"Query: {json.dumps(test_case_7, indent=2)}")
        results_7 = advanced_search(test_case_7)
        print(f"Found {results_7['count']} results")
        print("Results (limited to 3):")
        print(json.dumps(results_7['results'], indent=2, default=str))
        
        # Test Case 8: Advanced grouping with multiple aggregations
        print("\n========== TEST CASE 8: Advanced Grouping ==========")
        test_case_8 = {
            "grouping": "team->key",
            "filters": [
                {"field": "cycle->name", "operator": "=", "value": "Cycle 40"}
            ],
            "aggregations": [
                {"type": "count", "field": "*", "alias": "total_issues"},
                {
                    "type": "count", 
                    "field": "*", 
                    "condition": {
                        "field": "priority",
                        "operator": "<=",
                        "value": 2
                    }, 
                    "alias": "high_priority_issues"
                },
                {
                    "type": "count", 
                    "field": "*", 
                    "condition": {
                        "field": "state",
                        "operator": "=",
                        "value": "Done"
                    }, 
                    "alias": "completed_issues"
                }
            ],
            "sorting": {"field": "total_issues", "direction": "DESC"},
            "fields": ["team->key"],
            "returned_fields": {
                "Team": "group_field",
                "Total": "total_issues",
                "High Priority": "high_priority_issues",
                "Completed": "completed_issues"
            },
            "limit": 5,
            "debug": True
        }
        print(f"Query: {json.dumps(test_case_8, indent=2)}")
        results_8 = advanced_search(test_case_8)
        print(f"Found {results_8['count']} results")
        print("Results (team stats):")
        print(json.dumps(results_8['results'], indent=2, default=str))
            
    except Exception as e:
        print(f"Error running test cases: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()