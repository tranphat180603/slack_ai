#!/usr/bin/env python3
"""
Script to test Linear RAG advanced search with example queries from prompts.yaml.
This script tests both single-step and multi-step queries.
"""
import os
import json
import logging
from dotenv import load_dotenv
from linear_rag_search import advanced_search
from tabulate import tabulate
from linear_rag_search import process_variable_references

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_linear_query")

# Load environment variables
load_dotenv()

def display_results(results, test_name):
    """Display results in a formatted table."""
    print(f"\n========== {test_name} ==========")
    
    # Check if results exist
    if not results:
        logger.warning("No results object returned.")
        return
        
    # Get results data and count
    results_data = results.get('results', [])
    result_count = len(results_data)
    logger.info(f"Found {result_count} results")
    
    
    if not results_data:
        logger.warning("No data in results.")
        return
    
    # Handle different result formats
    if isinstance(results_data[0], dict):
        # Get column names from keys of first result
        column_names = list(results_data[0].keys())
        
        # Create table data
        table_data = []
        for row in results_data:
            table_row = []
            for key in column_names:
                value = row.get(key, "")
                # Convert value to string if it's not already
                if isinstance(value, dict):
                    value = json.dumps(value)
                elif value is None:
                    value = ""
                else:
                    value = str(value)
                table_row.append(value)
            table_data.append(table_row)
        
        # Print the table
        if table_data:
            print("\nResults Table:")
            print(tabulate(table_data, headers=column_names, tablefmt="grid"))
        else:
            print("\nNo data to display in table format.")
    else:
        # For non-dictionary results
        print("\nResults:")
        for result in results_data:
            print(result)

def main():
    """Run test queries from prompts.yaml examples."""
    logger.info("Testing Linear RAG advanced search with example queries...")
    
    # # Battle 1A: Semantic search only for Ahmed's agent-related issues
    # battle_1a = {
    #     "fields": ["title", "state", "assignee->name", "description"],
    #     "returned_fields": {
    #         "Title": "title",
    #         "State": "state",
    #         "Assignee": "assignee->name",
    #     },
    #     "semantic_query": "agent",
    #     "limit": 20
    # }

    # # Battle 1B: Semantic search + SQL filters for Ahmed's agent-related issues
    # battle_1b = {
    #     "fields": ["title", "state", "assignee->name", "description"],
    #     "returned_fields": {
    #         "Title": "title",
    #         "State": "state",
    #         "Assignee": "assignee->name",
    #     },
    #     "filters": [
    #         {"field": "assignee->name", "operator": "=", "value": "Ahmed Hamdy"},
    #         {"field": "cycle->name", "operator": "=", "value": "Cycle 41"}
    #     ],
    #     "semantic_query": "agent",
    #     "limit": 20
    # }

    # # Battle 2A: Semantic search only for Data API issues
    # battle_2a = {
    #     "fields": ["title", "state", "assignee->name", "description"],
    #     "returned_fields": {
    #         "Title": "title",
    #         "State": "state",
    #         "Assignee": "assignee->name",
    #     },
    #     "semantic_query": "Data API",
    #     "limit": 20
    # }

    # # Battle 2B: Semantic search + SQL filters for Data API issues
    # battle_2b = {
    #     "fields": ["title", "state", "assignee->name", "description"],
    #     "returned_fields": {
    #         "Title": "title",
    #         "State": "state",
    #         "Assignee": "assignee->name",
    #     },
    #     "filters": [
    #         {"field": "assignee->name", "operator": "=", "value": "Harsh Gautam"},
    #         {"field": "cycle->name", "operator": "=", "value": "Cycle 41"}
    #     ],
    #     "semantic_query": "Data API",
    #     "limit": 20
    # }

    # # Run the battles
    # print("\nBattle 1: Ahmed's Agent-Related Issues")
    # print("\nBattle 1A - Semantic Search Only:")
    # results_1a = advanced_search(battle_1a)
    # display_results(results_1a, "Semantic Search Only (agent)")
    
    # print("\nBattle 1B - Semantic Search + SQL Filters:")
    # results_1b = advanced_search(battle_1b)
    # display_results(results_1b, "Semantic Search + SQL Filters (agent + Ahmed + Cycle 41)")
    
    # print("\nBattle 2: Data API Issues")
    # print("\nBattle 2A - Semantic Search Only:")
    # results_2a = advanced_search(battle_2a)
    # display_results(results_2a, "Semantic Search Only (Data API)")
    
    # print("\nBattle 2B - Semantic Search + SQL Filters:")
    # results_2b = advanced_search(battle_2b)
    # display_results(results_2b, "Semantic Search + SQL Filters (Data API + Harsh + Cycle 41)")


    test_case_1 = {
        "fields": ["assignee->name"],
        "returned_fields": {
            "total_issues": "total_issues",
            "total_achievements": "total_achievements",
            "assignee": "group_field"
        },
        "filters": [
            {
            "field": "assignee->name",
            "operator": "=",
            "value": "Phát -"
            }
        ],
        "grouping": "assignee->name",
        "aggregations": [
            {
            "type": "count",
            "field": "*",
            "alias": "total_issues"
            },
            {
            "type": "count",
            "field": "*",
            "condition": {
                "field": "state",
                "operator": "=",
                "value": "Done"
            },
            "alias": "total_achievements"
            }
        ]
        }








    test_case_2 = {
    "fields": [
        "id",
        "title",
        "state",
        "team->key",
        "assignee->name",
        "cycle->name"
    ],
    "returned_fields": {
        "ID": "id",
        "Title": "title",
        "State": "state",
        "Team": "team->key",
        "Assignee Name": "assignee->name",
    },
    "filters": [
        {
        "field": "assignee->name",
        "operator": "=",
        "value": "Ph\u00e1t -"
        },
        {
        "field": "cycle->name",
        "operator": "=",
        "value": "Cycle 41"
        }
    ],
    "order_by": {
        "field": "created_at",
        "direction": "DESC"
    },
    "limit": 15
    }

    
    # # Test Case 2: Get issue statistics by team with priority breakdown
    # test_case_2 = {
    #     "fields": ["team->key"],
    #     "returned_fields": {
    #         "Team": "group_field",
    #         "Total": "total_issues",
    #         "Urgent": "urgent_issues",
    #         "High": "high_priority_issues",
    #         "Completed": "completed_issues"
    #     },
    #     "grouping": "team->key",
    #     "aggregations": [
    #         {"type": "count", "field": "*", "alias": "total_issues"},
    #         {
    #             "type": "count",
    #             "field": "*",
    #             "condition": {
    #                 "field": "priority",
    #                 "operator": "=",
    #                 "value": 1  # Urgent
    #             },
    #             "alias": "urgent_issues"
    #         },
    #         {
    #             "type": "count",
    #             "field": "*",
    #             "condition": {
    #                 "field": "priority",
    #                 "operator": "=",
    #                 "value": 2  # High
    #             },
    #             "alias": "high_priority_issues"
    #         },
    #         {
    #             "type": "count",
    #             "field": "*",
    #             "condition": {
    #                 "field": "state",
    #                 "operator": "=",
    #                 "value": "Done"
    #             },
    #             "alias": "completed_issues"
    #         }
    #     ],
    #     "limit": 10
    # }
    
    # Test Case 3: Get issue statistics by assignee for current cycle
    test_case_3 = {
        "fields": ["assignee->name"],
        "returned_fields": {
            "Assignee Name": "group_field",
            "Total Issues": "total_issues",
            "High Priority": "high_priority",
            "In Progress": "in_progress",
            "Completed": "completed"
        },
        "filters": [
            {"field": "cycle->name", "operator": "=", "value": "Cycle 41"}
        ],
        "grouping": "assignee->name",
        "aggregations": [
            {"type": "count", "field": "*", "alias": "total_issues"},
            {
                "type": "count",
                "field": "*",
                "condition": {
                    "field": "priority",
                    "operator": "IN",
                    "value": [1, 2]  # Urgent and High
                },
                "alias": "high_priority"
            },
            {
                "type": "count",
                "field": "*",
                "condition": {
                    "field": "state",
                    "operator": "=",
                    "value": "In Progress"
                },
                "alias": "in_progress"
            },
            {
                "type": "count",
                "field": "*",
                "condition": {
                    "field": "state",
                    "operator": "=",
                    "value": "Done"
                },
                "alias": "completed"
            }
        ],
        "sorting": {"field": "total_issues", "direction": "DESC"},
        "limit": 10
    }
    
    # Test Case 4: Two-step query: first find teams with urgent issues, then get team details
    # test_case_4_multi = [
    # {
    #     "fields": ["team->key"],
    #     "returned_fields": {
    #     "Team Key": "group_field",
    #     "Urgent Issues": "urgent_issues"
    #     },
    #     "grouping": "team->key",
    #     "filters": [
    #     {
    #         "field": "priority", 
    #         "operator": "=",
    #         "value": 1
    #     }
    #     ],
    #     "aggregations": [
    #     {
    #         "type": "count",
    #         "field": "*",
    #         "alias": "urgent_issues"
    #     }
    #     ],
    #     "sorting": {
    #     "field": "urgent_issues",
    #     "direction": "DESC"
    #     },
    #     "limit": 5,
    #     "result_variable": "urgent_teams"
    # },
    # {
    #     "fields": ["title", "team->key", "state", "priority", "created_at"],
    #     "returned_fields": {
    #     "Title": "title",
    #     "Team": "team->key",
    #     "Status": "state",
    #     "Priority": "priority"
    #     },
    #     "filters": [
    #     {
    #         "field": "team->key",
    #         "operator": "= ANY",
    #         "value": "{{urgent_teams.group_field}}"
    #     },
    #     {
    #         "field": "priority",
    #         "operator": "=",
    #         "value": 1
    #     }
    #     ],
    #     "distinct_on": "team->key",
    #     "per_group_limit": 2,
    #     "sorting": {
    #     "field": "created_at",
    #     "direction": "DESC"
    #     },
    #     "limit": 10
    # }
    # ]

    test_case_4_multi =     [
      {
        "fields": ["assignee->name"],
        "returned_fields": {
          "assignee": "group_field"
        },
        "filters": [
          {"field": "team->key", "operator": "=", "value": "AI"},
          {"field": "cycle->name", "operator": "=", "value": "Cycle 41"}
        ],
        "grouping": "assignee->name",
        "limit": 20,
        "result_variable": "ai_team_members",
      },
      {
        "fields": ["title", "assignee->name", "state", "priority", "created_at"],
        "returned_fields": {
          "title": "title",
          "assignee": "assignee->name",
          "state": "state",
          "priority": "priority",
          "created": "created_at"
        },
        "filters": [
          {"field": "cycle->name", "operator": "=", "value": "Cycle 41"},
          {
            "field": "assignee->name",
            "operator": "= ANY",
            "value": "{{ai_team_members.assignee}}"
          }
        ],
        "per_group_limit": 2,
        "sorting": {"field": "priority", "direction": "ASC"},
        "limit": 20,
        "distinct_on": "assignee->name",
      }
    ]

    
    # Test Case 6: Multi-step query to get issues for EVERYONE in AI Team in Cycle 41
    test_case_6_multi = [
        # Step 1: Get distinct assignees in AI Team for Cycle 41
        {
            "fields": ["assignee->name"],
            "returned_fields": {
                "Assignee": "group_field"
            },
            "filters": [
                {"field": "team->key", "operator": "=", "value": "AI"},
                {"field": "cycle->name", "operator": "=", "value": "Cycle 41"}
            ],
            "grouping": "assignee->name",
            "limit": 20,
            "result_variable": "ai_team_members",
            "debug": True  # Add debug flag to see SQL query
        },
        # Step 2: Get 1-2 issues per assignee to show what everyone is working on
        {
        "fields": ["title", "assignee->name", "state", "priority", "created_at"],
        "returned_fields": {
            "Title": "title",
            "Assignee": "assignee->name", 
            "Status": "state",
            "Priority": "priority",
            "Created": "created_at"
        },
        "filters": [
            { "field": "cycle->name", "operator": "=", "value": "Cycle 41" },
            {
            "field": "assignee->name",
            "operator": "= ANY",
            "value": "{{ai_team_members.group_field}}"
            }
        ],
            "per_group_limit": 2,  
            "sorting": { "field": "priority", "direction": "ASC" },
            "limit": 20,
            "distinct_on": "assignee->name",
            "debug": True
        }
    ]
    
    # Run single-step queries
    try:
        print("\nTesting Single-Step Queries:")
        
        # Test Case 1
        results_1 = advanced_search(test_case_1)
        display_results(results_1, "Test Case 1: TESING VARIOUS THINGS")
        
        # Test Case 2
        results_2 = advanced_search(test_case_2)
        display_results(results_2, "Test Case 2: Team Statistics with Priority Breakdown")
        
        # Test Case 3
        results_3 = advanced_search(test_case_3)
        display_results(results_3, "Test Case 3: Assignee Statistics")
    
        
        print("\nTesting Multi-Step Queries:")
        
        # Process test_case_4_multi
        step_results = {}
        final_results = None
        
        for i, step_query in enumerate(test_case_4_multi):
            print(f"\nExecuting step {i+1}...")
            
            # Extract the result_variable name if specified
            result_variable = step_query.pop("result_variable", f"query_{i+1}_result")
            
            # Process any variable references in the query
            processed_query = process_variable_references(step_query, step_results)
            
            # Execute the query
            step_result = advanced_search(processed_query)
            
            # Store the results for use by subsequent queries
            step_results[result_variable] = step_result.get("results", [])
            
            # The final step results will be displayed
            if i == len(test_case_4_multi) - 1:
                final_results = step_result
        
        display_results(final_results, "Test Case 4: Active Teams' Urgent/High Priority Issues")
        
        # Process test_case_6_multi (Everyone in AI Team)
        step_results = {}
        final_results = None
        
        for i, step_query in enumerate(test_case_6_multi):
            print(f"\nExecuting step {i+1} for AI Team Members Query...")
            
            # Extract the result_variable name if specified
            result_variable = step_query.pop("result_variable", f"query_{i+1}_result")
            
            # Process any variable references in the query
            processed_query = process_variable_references(step_query, step_results)
            
            # Execute the query and print debug information
            step_result = advanced_search(processed_query)
            
            # Store the results for use by subsequent queries
            step_results[result_variable] = step_result.get("results", [])
            
            # The final step results will be displayed
            if i == len(test_case_6_multi) - 1:
                final_results = step_result
        
        display_results(final_results, "Test Case 6: All AI Team Members' Issues in Cycle 41")
        
    except Exception as e:
        logger.error(f"Error running test cases: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    # # Test Case: Semantic search with grouping and aggregation (the problematic case)
    # semantic_with_grouping = {
    #     "fields": ["state"],
    #     "returned_fields": {
    #         "State": "group_field",
    #         "Total Issues": "total_issues"
    #     },
    #     "semantic_search": "Indices for Token Metrics website",
    #     "aggregations": [
    #         {"type": "count", "field": "*", "alias": "total_issues"}
    #     ],
    #     "grouping": "state",
    #     "limit": 10
    # }
    
    # print("\nTest Case: Semantic Search with Grouping")
    # print("This tests the combination that was causing errors")
    # try:
    #     results_semantic_grouping = advanced_search(semantic_with_grouping)
    #     display_results(results_semantic_grouping, "Semantic Search with Grouping and Aggregation")
    # except Exception as e:
    #     logger.error(f"Error in semantic search with grouping test: {str(e)}")
    #     import traceback
    #     logger.error(traceback.format_exc())
        
    # # Try an alternative approach without semantic search
    # print("\nTrying alternative approach - separate queries")
    
    # # First query - just get the states
    # states_only = {
    #     "fields": ["state"],
    #     "returned_fields": {
    #         "State": "group_field"
    #     },
    #     "grouping": "state",
    #     "limit": 10,
    #     "result_variable": "available_states"
    # }
    
    # # Second query - semantic search for each state
    # semantic_search_by_state = {
    #     "fields": ["title", "state", "id"],
    #     "returned_fields": {
    #         "Title": "title",
    #         "State": "state",
    #         "ID": "id"
    #     },
    #     "semantic_search": "Indices for Token Metrics website",
    #     "limit": 15
    # }
    
    # Create a proper multi-step test case for semantic search with grouping
    semantic_with_grouping_multi = [
        # Step 1: Get all states
        {
            "fields": ["state"],
            "returned_fields": {
                "State": "group_field"
            },
            "grouping": "state",
            "limit": 10,
            "result_variable": "available_states"
        },
        # Step 2: Do semantic search and filter by states from step 1
        {
            "fields": ["title", "state", "id"],
            "returned_fields": {
                "Title": "title", 
                "State": "state",
                "ID": "id"
            },
            "semantic_search": "Indices for Token Metrics website",
            "filters": [
                {
                    "field": "state",
                    "operator": "= ANY",
                    "value": "{{available_states.State}}"
                }
            ],
            "limit": 15
        }
    ]
    
    print("\nTest Case: Multi-Step Semantic Search with Grouping")
    print("This tests a workaround for the problematic case using proper multi-step approach")
    
    # Process the multi-step approach using the same pattern as other multi-step queries
    try:
        step_results = {}
        final_results = None
        
        for i, step_query in enumerate(semantic_with_grouping_multi):
            print(f"\nExecuting step {i+1} for multi-step semantic search...")
            
            # Extract the result_variable name if specified
            result_variable = step_query.pop("result_variable", f"query_{i+1}_result")
            
            # Process any variable references in the query
            processed_query = process_variable_references(step_query, step_results)
            
            # Execute the query
            step_result = advanced_search(processed_query)
            
            # Store the results for use by subsequent queries
            step_results[result_variable] = step_result.get("results", [])
            
            # The final step results will be displayed
            if i == len(semantic_with_grouping_multi) - 1:
                final_results = step_result
        
        display_results(final_results, "Multi-Step Semantic Search with States Filtering")
        
        # Post-process to show results grouped by state
        if final_results and final_results.get("results"):
            # Group results by state
            grouped_results = {}
            for item in final_results.get("results", []):
                state = item.get("State")
                if state not in grouped_results:
                    grouped_results[state] = {
                        "State": state,
                        "Total Issues": 0,
                        "Issues": []
                    }
                grouped_results[state]["Total Issues"] += 1
                grouped_results[state]["Issues"].append(item["Title"])
            
            # Display grouped summary
            print("\nPost-Processed Grouping Summary:")
            for state, data in grouped_results.items():
                print(f"State: {state} - Total Issues: {data['Total Issues']}")
                if data["Issues"]:
                    print("  Sample issues:")
                    for i, title in enumerate(data["Issues"][:3]):  # Show up to 3 examples
                        print(f"  - {title}")
                    if len(data["Issues"]) > 3:
                        print(f"  - ... and {len(data['Issues'])-3} more")
        
    except Exception as e:
        logger.error(f"Error in multi-step approach: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    


if __name__ == "__main__":
    main() 