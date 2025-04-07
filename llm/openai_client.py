"""
OpenAI API client
"""

import os
import time
import logging
import traceback
import json
import asyncio
from typing import Dict, List, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
import dotenv

dotenv.load_dotenv()

import openai
from openai import OpenAI

# Configure logger
logger = logging.getLogger("openai_client")


class OpenaiClient(ABC):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def response(self, prompt: str, system_prompt: str = None, stream: bool = False) -> str:
        assert self.model.startswith("gpt"), "Only GPT models are supported"
        messages = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt
                }
            )
        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )
        
        # Log the complete request being sent to OpenAI
        logger.debug(f"===== REQUEST TO OPENAI ({self.model}) =====")
        logger.debug(f"System prompt: {system_prompt}")
        logger.debug(f"User prompt: {prompt}")
        logger.debug("======================================")
        
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            stream=stream,
            temperature=0.7
        )
        if stream:
            full_response = ""
            for chunk in response:
                # Check for ResponseTextDeltaEvent with delta field
                if hasattr(chunk, 'type') and chunk.type == 'response.output_text.delta':
                    if hasattr(chunk, 'delta'):
                        delta = chunk.delta
                        full_response += delta
                        print(delta, end="", flush=True)
            print()  # Add a newline at the end
            
            # Log the full response
            logger.debug(f"===== RESPONSE FROM OPENAI (STREAM) =====")
            logger.debug(f"{full_response}")
            logger.debug("========================================")
            
            return full_response
        else:
            result = response.output[0].content[0].text
            
            # Log the full response
            logger.debug(f"===== RESPONSE FROM OPENAI =====")
            logger.debug(f"{result}")
            logger.debug("==============================")
            
            return result

    def response_reasoning(self, prompt: str, reasoning_effort: str, stream: bool = False) -> str:
        assert self.model.startswith("o"), "Only reasoning models are supported"
        messages = []
        messages.append(
            {
                "role": "system",
                "content": "You are a helpful assistant that can reason about the user's prompt."
            }
        )
        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )
        
        # Log the complete request being sent to OpenAI
        logger.debug(f"===== REQUEST TO OPENAI ({self.model}) =====")
        logger.debug(f"Reasoning effort: {reasoning_effort}")
        logger.debug(f"System prompt: You are a helpful assistant that can reason about the user's prompt.")
        logger.debug(f"User prompt: {prompt}")
        logger.debug("======================================")
        
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            stream=stream,
            reasoning={"effort": reasoning_effort},
        )
        if stream:
            full_response = ""
            for chunk in response:
                # Check for ResponseTextDeltaEvent with delta field
                if hasattr(chunk, 'type') and chunk.type == 'response.output_text.delta':
                    if hasattr(chunk, 'delta'):
                        delta = chunk.delta
                        full_response += delta
                        print(delta, end="", flush=True)
            print()  # Add a newline at the end
            
            # Log the full response
            logger.debug(f"===== RESPONSE FROM OPENAI (STREAM) =====")
            logger.debug(f"{full_response}")
            logger.debug("========================================")
            
            return full_response
        else:
            # Extract text from the output message
            if hasattr(response, 'output') and response.output:
                # Find the message item in the output list
                for item in response.output:
                    if hasattr(item, 'type') and item.type == 'message':
                        # Message item has content array with ResponseOutputText objects
                        if hasattr(item, 'content') and item.content:
                            for content_item in item.content:
                                if hasattr(content_item, 'text'):
                                    result = content_item.text
                                    
                                    # Log the full response
                                    logger.debug(f"===== RESPONSE FROM OPENAI (REASONING) =====")
                                    logger.debug(f"{result}")
                                    logger.debug("========================================")
                                    
                                    return result
                
                # If we got here, we didn't find the expected structure
                logger.warning("Could not find text in the message content, trying output_text attribute")
                if hasattr(response, 'output_text'):
                    return response.output_text
                    
                # Final fallback
                logger.warning("Unusual response structure, returning string representation of output")
                logger.debug(f"Raw response output: {response.output}")
                return str(response.output)
            
            logger.error("Failed to extract text from reasoning response")
            return "Error: Could not extract response text"
        
    def use_tool(self, prompt: str, reasoning_effort: str, tools: List[Dict[str, Any]] = None, stream: bool = False) -> Dict[str, Any]:
        assert self.model.startswith("o"), "Only reasoning models are supported"
        
        # Log the complete request being sent to OpenAI
        logger.debug(f"===== TOOL REQUEST TO OPENAI ({self.model}) =====")
        logger.debug(f"Reasoning effort: {reasoning_effort}")
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Tools: {json.dumps(tools, indent=2)}")
        logger.debug("===========================================")
        
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            stream=stream,
            reasoning={"effort": reasoning_effort},
            tools=tools,
        )
        
        # Find the function call output
        function_call = None
        for output in response.output:
            if hasattr(output, 'type') and output.type == 'function_call':
                function_call = output
                break
        
        if not function_call:
            logger.warning("No function call found in tool response")
            return None
            
        # Parse the arguments string into a dictionary
        try:
            arguments = json.loads(function_call.arguments)
        except json.JSONDecodeError:
            arguments = function_call.arguments
            
        tool_called = {
            "name": function_call.name,
            "parameters": arguments,
        }
        
        # Log the tool response
        logger.debug(f"===== TOOL RESPONSE FROM OPENAI =====")
        logger.debug(f"Tool called: {function_call.name}")
        logger.debug(f"Parameters: {json.dumps(arguments, indent=2)}")
        logger.debug("===================================")
        
        return tool_called

def test():
    client = OpenaiClient(os.getenv("OPENAI_API_KEY"), model="o3-mini")
    tool_schema = [
                    {
                        "type": "function",
                        "name": "filter_linear_issues",
                        "description": "GraphQL-based function to filter Linear issues based on various criteria such as state, priority, assignee, etc.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "team_key": {
                                    "type": "string",
                                    "description": "The team key to filter issues by",
                                    "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
                                },
                                "state": {
                                    "type": "string",
                                    "description": "Filter by issue state (e.g. 'In Progress', 'Todo', 'Done')"
                                },
                                "priority": {
                                    "type": "number",
                                    "description": "Filter by priority level (0.0: None, 1.0: Urgent, 2.0: High, 3.0: Medium, 4.0: Low)",
                                    "enum": [0.0, 1.0, 2.0, 3.0, 4.0]
                                },
                                "assignee_name": {
                                    "type": "string",
                                    "description": "Filter by assignee's display name. Supports exact match. Use assignee_contains for partial matches."
                                },
                                "assignee_contains": {
                                    "type": "string",
                                    "description": "Filter by assignee names containing this text (case-insensitive)"
                                },
                                "title_contains": {
                                    "type": "string",
                                    "description": "Filter issues where title contains this string"
                                },
                                "description_contains": {
                                    "type": "string",
                                    "description": "Filter issues where description contains this string"
                                },
                                "cycle_name": {
                                    "type": "string",
                                    "description": "Filter by cycle name"
                                },
                                "project_name": {
                                    "type": "string",
                                    "description": "Filter by project name"
                                },
                                "label_name": {
                                    "type": "string",
                                    "description": "Filter by label name"
                                },
                                "first": {
                                    "type": "number",
                                    "description": "Limit the number of issues returned"
                                }
                            },
                            "required": ["team_key"],
                            "additionalProperties": False
                        }
                    }
    ]
    
    
    # Try use_tool (this works)
    tool_response = client.response_reasoning("what are the tasks I have to do for this cycle?. My name is Phat", stream=True, reasoning_effort="low")
    

if __name__ == "__main__":
    test()

