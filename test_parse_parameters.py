#!/usr/bin/env python3
"""
Test script to parse parameters from an updateIssue JSON payload
"""

import json
import pprint

# Sample JSON data as provided in the query
json_data = """
{
  "updateIssue": {
    "function": "updateIssue",
    "parameters": {
      "team_key": "OPS",
      "issue_number": 2568,
      "title": "Optimize Asynchronous Data Processing Pipelines for Enhanced Throughput and Fault Tolerance",
      "description": "## Context\\n\\nCurrent operational workflows rely heavily on synchronous data processing approaches, which are suboptimal in high-traffic or distributed scenarios. There is a critical need to redesign and optimize the data pipelines to maximize throughput, reduce latency, and ensure robust error handling.\\n\\n## Objectives\\n\\n* Transition from synchronous to fully asynchronous data ingestion and processing models.\\n* Integrate state monitoring and circuit breaker patterns to increase failure resilience.\\n* Benchmark before and after performance (latency, throughput, error rate).\\n* Establish clear rollback and retry procedures.\\n* Document architectural changes and interface contracts for downstream teams.\\n\\n## Acceptance Criteria\\n\\n* End-to-end migration of at least one major pipeline to an asynchronous model.\\n* Implementation of automated health checks and alerting on data pipeline failures.\\n* Demonstrated improvement in throughput (>30%) and fault tolerance in simulated failure tests.\\n* All changes documented and reviewed.\\n\\n## Technical Constraints\\n\\n* Use only currently supported frameworks and libraries.\\n* Solution must be backwards compatible.\\n\\n## Dependencies\\n\\n* Coordination with DevOps for deployment pipeline updates.\\n* Review with QA for automated failover test coverage.\\n\\n---\\n\\nPlease prioritize this as a high-value, strategic operational improvement initiative.",
      "priority": 0,
      "assignee_name": "phat",
      "state_name": "Todo",
      "cycle_number": 43
    },
    "result": null,
    "requires_modal_approval": true,
    "description": "Waiting for approval to updateIssue"
  }
}
"""

def parse_update_issue_parameters(json_string):
    """
    Parse the parameters from an updateIssue JSON payload
    
    Args:
        json_string (str): JSON string to parse
        
    Returns:
        dict: The extracted parameters
    """
    try:
        # Parse the JSON data
        data = json.loads(json_string)
        
        # Extract the parameters field
        parameters = data.get("updateIssue", {}).get("parameters", {})
        
        return parameters
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}

def main():
    # Parse the parameters
    parameters = parse_update_issue_parameters(json_data)
    
    # Print the entire parameters dictionary
    print("==== Full Parameters ====")
    pprint.pprint(parameters)
    
    # Print individual parameter values
    print("\n==== Individual Parameters ====")
    print(f"Team Key: {parameters.get('team_key')}")
    print(f"Issue Number: {parameters.get('issue_number')}")
    print(f"Title: {parameters.get('title')}")
    print(f"Priority: {parameters.get('priority')}")
    print(f"Assignee Name: {parameters.get('assignee_name')}")
    print(f"State Name: {parameters.get('state_name')}")
    print(f"Cycle Number: {parameters.get('cycle_number')}")
    
    # Test if all required parameters are present
    required_fields = ['team_key', 'issue_number', 'title']
    missing_fields = [field for field in required_fields if field not in parameters]
    if missing_fields:
        print(f"\nMissing required fields: {missing_fields}")
    else:
        print("\nAll required fields are present.")
    
    # Test compatibility with both team_key and teamKey
    team_key = parameters.get('team_key') or parameters.get('teamKey')
    print(f"\nResolved team_key: {team_key}")

if __name__ == "__main__":
    main() 