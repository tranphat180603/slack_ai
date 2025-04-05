import logging
from linear_rag_search import advanced_search
from tabulate import tabulate
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_linear_search_operator")

def display_results(results, test_name):
    """Display results in a formatted table."""
    print(f"\n========== {test_name} ==========")
    
    if not results:
        logger.warning("No results returned")
        return
        
    results_data = results.get('results', [])
    result_count = len(results_data)
    print(f"Found {result_count} results")
    
    if results_data:
        # Create table from results
        if isinstance(results_data[0], dict):
            headers = results_data[0].keys()
            table = [[row.get(col, '') for col in headers] for row in results_data]
            print(tabulate(table, headers=headers, tablefmt="grid"))
        else:
            print(json.dumps(results_data, indent=2))

def run_tests():
    """Run test cases based on understanding the Linear search operator prompt."""
    
    # Test Case 1: Basic Priority-Based Search
    print("\nTest Case 1: High Priority Issues for Specific Assignee")
    query_1 = {
        "fields": ["id", "title", "state", "priority"],
        "returned_fields": {
            "id": "id",
            "title": "title",
            "state": "state",
            "priority": "priority"
        },
        "filters": [
            {"field": "assignee->name", "operator": "=", "value": "Phát -"},
            {"field": "priority", "operator": "<=", "value": 2},
            {"field": "state", "operator": "!=", "value": "Done"}
        ],
        "sorting": {"field": "priority", "direction": "ASC"},
        "limit": 10
    }
    results_1 = advanced_search(query_1)
    display_results(results_1, "High Priority Issues for Phát")

    # Test Case 2: Team Statistics with Aggregation
    print("\nTest Case 2: Team Statistics for Current Cycle")
    query_2 = {
        "grouping": "team->key",
        "fields": ["team->key"],
        "returned_fields": {
            "team": "group_field",
            "total": "total_issues",
            "high_priority": "high_priority_issues"
        },
        "filters": [
            {"field": "cycle->name", "operator": "=", "value": "Cycle 41"}
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
            }
        ],
        "sorting": {"field": "total_issues", "direction": "DESC"},
        "limit": 5
    }
    results_2 = advanced_search(query_2)
    display_results(results_2, "Team Statistics")

    # Test Case 3: Semantic Search
    print("\nTest Case 3: Semantic Search for AI Agent")
    query_3 = {
        "semantic_search": "AI Agent implementation",
        "fields": ["id", "title", "team->key", "state"],
        "returned_fields": {
            "id": "id",
            "title": "title",
            "team": "team->key",
            "state": "state",
            "relevance": "similarity_score"
        },
        "limit": 5
    }
    results_3 = advanced_search(query_3)
    display_results(results_3, "AI Agent Related Issues")

    # Test Case 4: Multi-Step Query
    print("\nTest Case 4: Active Teams and Their Issues")
    query_4 = [
        {
            "grouping": "team->key",
            "fields": ["team->key"],
            "returned_fields": {
                "team": "group_field",
                "count": "active_issues"
            },
            "filters": [
                {"field": "state", "operator": "=", "value": "In Progress"}
            ],
            "aggregations": [
                {"type": "count", "field": "*", "alias": "active_issues"}
            ],
            "sorting": {"field": "active_issues", "direction": "DESC"},
            "limit": 3,
            "result_variable": "active_teams"
        },
        {
            "fields": ["title", "state", "assignee->name", "team->key"],
            "returned_fields": {
                "title": "title",
                "state": "state",
                "assignee": "assignee->name",
                "team": "team->key"
            },
            "filters": [
                {
                    "field": "team->key",
                    "operator": "= ANY",
                    "value": "{{active_teams.team}}"
                },
                {"field": "state", "operator": "=", "value": "In Progress"}
            ],
            "limit": 10
        }
    ]
    
    # For multi-step query, we need to process each step
    step_results = {}
    for i, step in enumerate(query_4):
        print(f"\nExecuting step {i+1}...")
        result_variable = step.pop("result_variable", None)
        step_result = advanced_search(step)
        if result_variable:
            step_results[result_variable] = step_result.get("results", [])
        if i == len(query_4) - 1:
            display_results(step_result, "Active Teams and Their Issues")

    # Test Case 5: Distinct Issues per Assignee
    print("\nTest Case 5: Sample Issues per Assignee")
    query_5 = {
        "fields": ["title", "assignee->name", "state"],
        "returned_fields": {
            "title": "title",
            "assignee": "assignee->name",
            "state": "state"
        },
        "filters": [
            {"field": "team->key", "operator": "=", "value": "AI"}
        ],
        "distinct_on": "assignee->name",
        "per_group_limit": 2,
        "limit": 10
    }
    results_5 = advanced_search(query_5)
    display_results(results_5, "Sample Issues per Assignee")

if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())