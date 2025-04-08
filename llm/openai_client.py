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
import tiktoken

dotenv.load_dotenv()

import openai
from openai import OpenAI

# Configure logger
logger = logging.getLogger("openai_client")


class TokenUsageTracker:
    """Tracks token usage for OpenAI API calls"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset usage statistics"""
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_counts = {}
        self.function_counts = {}
        self.last_tracked_model = None
        self.last_tracked_function = None
    
    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """Count tokens in a string using tiktoken"""
        try:
            # Get the encoding for the model
            if model.startswith("gpt-4"):
                encoding_name = "cl100k_base"  # For GPT-4 models
            elif model.startswith("gpt-3.5"):
                encoding_name = "cl100k_base"  # For GPT-3.5-turbo models
            elif model.startswith("o"):
                encoding_name = "cl100k_base"  # For Claude/Anthropic models (approximation)
            else:
                encoding_name = "cl100k_base"  # Default to cl100k_base
            
            encoding = tiktoken.get_encoding(encoding_name)
            token_count = len(encoding.encode(text))
            return token_count
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}")
            # Fallback to rough character-based estimate
            return len(text) // 4
    
    def add_call(self, function_name: str, model: str, input_tokens: int, output_tokens: int = 0):
        """Record a new API call with token usage"""
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        
        # Track model used
        self.last_tracked_model = model
        self.last_tracked_function = function_name
        
        # Track by model
        if model not in self.model_counts:
            self.model_counts[model] = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        
        self.model_counts[model]["calls"] += 1
        self.model_counts[model]["input_tokens"] += input_tokens
        self.model_counts[model]["output_tokens"] += output_tokens
        
        # Track by function
        if function_name not in self.function_counts:
            self.function_counts[function_name] = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
            
        self.function_counts[function_name]["calls"] += 1
        self.function_counts[function_name]["input_tokens"] += input_tokens
        self.function_counts[function_name]["output_tokens"] += output_tokens
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Get a report of token usage"""
        return {
            "total_calls": self.calls,
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "estimated_cost_usd": self.estimate_cost(),
            "model_breakdown": self.model_counts,
            "function_breakdown": self.function_counts
        }
        
    def estimate_cost(self) -> float:
        """Estimate cost in USD based on token usage (very rough estimate)"""
        cost = 0.0
        
        # Define approximate costs per 1M tokens for different models
        # These are estimates and should be updated as prices change
        costs = {
            "gpt-4o": {"input": 5.0, "output": 15.0},        # $5/1M input, $15/1M output
            "gpt-4o-mini": {"input": 0.15, "output": 0.6},   # $0.15/1M input, $0.6/1M output
            "o1-mini": {"input": 3.0, "output": 15.0},       # Claude 3.5 Sonnet
            "o1": {"input": 8.0, "output": 24.0},            # Claude 3.5 Opus
            "o3-mini": {"input": 0.25, "output": 1.25},      # Claude 3 Haiku
            "o3": {"input": 3.0, "output": 15.0}             # Claude 3 Sonnet
        }
        
        # Default cost rates if specific model not found
        default_cost = {"input": 1.0, "output": 2.0}
        
        for model, usage in self.model_counts.items():
            # Get cost rates for this model, or use default
            model_base = model.split("-")[0]  # Handle model variants
            cost_rates = costs.get(model, default_cost)
            
            # Calculate cost
            input_cost = (usage["input_tokens"] / 1_000_000) * cost_rates["input"]
            output_cost = (usage["output_tokens"] / 1_000_000) * cost_rates["output"]
            cost += input_cost + output_cost
            
        return cost
    
    def log_report(self, log_level=logging.DEBUG):
        """Log token usage report at the specified level"""
        report = self.get_usage_report()
        
        if logger.isEnabledFor(log_level):
            logger.log(log_level, "===== TOKEN USAGE REPORT =====")
            logger.log(log_level, f"Total API calls: {report['total_calls']}")
            logger.log(log_level, f"Total tokens: {report['total_tokens']} (Input: {report['total_input_tokens']}, Output: {report['total_output_tokens']})")
            logger.log(log_level, f"Estimated cost: ${report['estimated_cost_usd']:.5f} USD")
            
            logger.log(log_level, "Model breakdown:")
            for model, usage in report['model_breakdown'].items():
                logger.log(log_level, f"  {model}: {usage['calls']} calls, {usage['input_tokens']} input, {usage['output_tokens']} output")
            
            logger.log(log_level, "Function breakdown:")
            for func, usage in report['function_breakdown'].items():
                logger.log(log_level, f"  {func}: {usage['calls']} calls, {usage['input_tokens']} input, {usage['output_tokens']} output")
            
            logger.log(log_level, "===============================")


class OpenaiClient(ABC):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.token_tracker = TokenUsageTracker()

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
        
        # Count input tokens for logging only (will be replaced with actual count from API)
        input_text = system_prompt or "" + prompt
        estimated_input_tokens = self.token_tracker.count_tokens(input_text, self.model)
        
        # Log the complete request being sent to OpenAI
        logger.debug(f"===== REQUEST TO OPENAI ({self.model}) =====")
        logger.debug(f"System prompt: {system_prompt}")
        logger.debug(f"User prompt: {prompt}")
        logger.debug(f"Estimated input tokens: {estimated_input_tokens}")
        logger.debug("======================================")
        
        # Reset last_tracked model to detect if we'll get usage info
        self.token_tracker.last_tracked_model = None
        self.token_tracker.last_tracked_function = None
        
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
                # Check for done event with usage information
                elif hasattr(chunk, 'type') and chunk.type == 'response.done':
                    if hasattr(chunk, 'response') and hasattr(chunk.respon, 'usage'):
                        # Extract usage information
                        usage = chunk.response.usage
                        input_tokens = usage.input_tokens
                        output_tokens = usage.output_tokens
                        
                        # Track token usage with actual counts from API
                        self.token_tracker.add_call("response", self.model, input_tokens, output_tokens)
                        
                        # Log token usage
                        logger.debug(f"Actual input tokens: {input_tokens}")
                        logger.debug(f"Actual output tokens: {output_tokens}")
            print()  # Add a newline at the end
            
            # If we didn't get usage from API, use estimates as fallback
            if self.token_tracker.last_tracked_function != "response" or self.token_tracker.last_tracked_model != self.model:
                output_tokens = self.token_tracker.count_tokens(full_response, self.model)
                # Track token usage with estimated counts
                self.token_tracker.add_call("response", self.model, estimated_input_tokens, output_tokens)
                logger.debug(f"Using estimated tokens (no API data): input={estimated_input_tokens}, output={output_tokens}")
            
            # Log the full response
            logger.debug(f"===== RESPONSE FROM OPENAI (STREAM) =====")
            logger.debug(f"{full_response}")
            logger.debug("========================================")
            
            # Log token usage if in debug mode
            self.token_tracker.log_report()
            
            return full_response
        else:
            result = response.output[0].content[0].text
            
            # Get actual token usage from API response
            input_tokens = estimated_input_tokens
            output_tokens = self.token_tracker.count_tokens(result, self.model)
            
            # Extract token counts from response if available
            if hasattr(response, 'usage'):
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
            
            # Track token usage with actual counts
            self.token_tracker.add_call("response", self.model, input_tokens, output_tokens)
            
            # Log the full response
            logger.debug(f"===== RESPONSE FROM OPENAI =====")
            logger.debug(f"{result}")
            logger.debug(f"Actual input tokens: {input_tokens}")
            logger.debug(f"Actual output tokens: {output_tokens}")
            logger.debug("==============================")
            
            # Log token usage if in debug mode
            self.token_tracker.log_report()
            
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
        
        # Count input tokens for logging only (will be replaced with actual count from API)
        input_text = "You are a helpful assistant that can reason about the user's prompt." + prompt
        estimated_input_tokens = self.token_tracker.count_tokens(input_text, self.model)
        
        # Log the complete request being sent to OpenAI
        logger.debug(f"===== REQUEST TO OPENAI ({self.model}) =====")
        logger.debug(f"Reasoning effort: {reasoning_effort}")
        logger.debug(f"System prompt: You are a helpful assistant that can reason about the user's prompt.")
        logger.debug(f"User prompt: {prompt}")
        logger.debug(f"Estimated input tokens: {estimated_input_tokens}")
        logger.debug("======================================")
        
        # Reset last_tracked model to detect if we'll get usage info
        self.token_tracker.last_tracked_model = None
        self.token_tracker.last_tracked_function = None
        
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
                # Check for done event with usage information
                elif hasattr(chunk, 'type') and chunk.type == 'response.done':
                    if hasattr(chunk, 'response') and hasattr(chunk.response, 'usage'):
                        # Extract usage information
                        usage = chunk.response.usage
                        input_tokens = usage.input_tokens
                        output_tokens = usage.output_tokens
                        
                        # Track token usage with actual counts from API
                        self.token_tracker.add_call("response_reasoning", self.model, input_tokens, output_tokens)
                        
                        # Log token usage
                        logger.debug(f"Actual input tokens: {input_tokens}")
                        logger.debug(f"Actual output tokens: {output_tokens}")
            print()  # Add a newline at the end
            
            # If we didn't get usage from API, use estimates as fallback
            if self.token_tracker.last_tracked_function != "response_reasoning" or self.token_tracker.last_tracked_model != self.model:
                output_tokens = self.token_tracker.count_tokens(full_response, self.model)
                # Track token usage with estimated counts
                self.token_tracker.add_call("response_reasoning", self.model, estimated_input_tokens, output_tokens)
                logger.debug(f"Using estimated tokens (no API data): input={estimated_input_tokens}, output={output_tokens}")
            
            # Log the full response
            logger.debug(f"===== RESPONSE FROM OPENAI (STREAM) =====")
            logger.debug(f"{full_response}")
            logger.debug("========================================")
            
            # Log token usage if in debug mode
            self.token_tracker.log_report()
            
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
                                    
                                    # Get actual token usage from API response
                                    input_tokens = estimated_input_tokens
                                    output_tokens = self.token_tracker.count_tokens(result, self.model)
                                    
                                    # Extract token counts from response if available
                                    if hasattr(response, 'usage'):
                                        input_tokens = response.usage.input_tokens
                                        output_tokens = response.usage.output_tokens
                                    
                                    # Track token usage with actual counts
                                    self.token_tracker.add_call("response_reasoning", self.model, input_tokens, output_tokens)
                                    
                                    # Log the full response
                                    logger.debug(f"===== RESPONSE FROM OPENAI (REASONING) =====")
                                    logger.debug(f"{result}")
                                    logger.debug(f"Actual input tokens: {input_tokens}")
                                    logger.debug(f"Actual output tokens: {output_tokens}")
                                    logger.debug("========================================")
                                    
                                    # Log token usage if in debug mode
                                    self.token_tracker.log_report()
                                    
                                    return result
                
                # If we got here, we didn't find the expected structure
                logger.warning("Could not find text in the message content, trying output_text attribute")
                if hasattr(response, 'output_text'):
                    result = response.output_text
                    
                    # Get token usage from response if available
                    input_tokens = estimated_input_tokens
                    output_tokens = self.token_tracker.count_tokens(str(result), self.model)
                    
                    # Extract token counts from response if available
                    if hasattr(response, 'usage'):
                        input_tokens = response.usage.input_tokens
                        output_tokens = response.usage.output_tokens
                    
                    # Track token usage
                    self.token_tracker.add_call("response_reasoning", self.model, input_tokens, output_tokens)
                    
                    # Log token usage if in debug mode
                    self.token_tracker.log_report()
                    
                    return result
                    
                # Final fallback
                logger.warning("Unusual response structure, returning string representation of output")
                result = str(response.output)
                
                # Get token usage from response if available
                input_tokens = estimated_input_tokens
                output_tokens = self.token_tracker.count_tokens(result, self.model)
                
                # Extract token counts from response if available
                if hasattr(response, 'usage'):
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                
                # Track token usage
                self.token_tracker.add_call("response_reasoning", self.model, input_tokens, output_tokens)
                
                logger.debug(f"Raw response output: {response.output}")
                
                # Log token usage if in debug mode
                self.token_tracker.log_report()
                
                return result
            
            logger.error("Failed to extract text from reasoning response")
            
            # Track token usage for error case with estimated values
            input_tokens = estimated_input_tokens
            
            # Extract token counts from response if available
            if hasattr(response, 'usage'):
                input_tokens = response.usage.input_tokens
            
            # Track token usage
            self.token_tracker.add_call("response_reasoning", self.model, input_tokens, 0)
            
            # Log token usage if in debug mode
            self.token_tracker.log_report()
            
            return "Error: Could not extract response text"
        
    def use_tool(self, prompt: str, reasoning_effort: str, tools: List[Dict[str, Any]] = None, stream: bool = False) -> Dict[str, Any]:
        assert self.model.startswith("o"), "Only reasoning models are supported"
        
        # Count input tokens for logging only (will be replaced with actual count from API)
        input_text = prompt + json.dumps(tools, ensure_ascii=False)
        estimated_input_tokens = self.token_tracker.count_tokens(input_text, self.model)
        
        # Log the complete request being sent to OpenAI
        logger.debug(f"===== TOOL REQUEST TO OPENAI ({self.model}) =====")
        logger.debug(f"Reasoning effort: {reasoning_effort}")
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Tools: {json.dumps(tools, indent=2)}")
        logger.debug(f"Estimated input tokens: {estimated_input_tokens}")
        logger.debug("===========================================")
        
        # Reset last_tracked model to detect if we'll get usage info
        self.token_tracker.last_tracked_model = None
        self.token_tracker.last_tracked_function = None
        
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
            
            # Get token usage from response if available
            input_tokens = estimated_input_tokens
            
            # Extract token counts from response if available
            if hasattr(response, 'usage'):
                input_tokens = response.usage.input_tokens
            
            # Track token usage for error case
            self.token_tracker.add_call("use_tool", self.model, input_tokens, 0)
            
            # Log token usage if in debug mode
            self.token_tracker.log_report()
            
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
        
        # Count output tokens using the function call length as fallback
        output_str = json.dumps(tool_called, ensure_ascii=False)
        estimated_output_tokens = self.token_tracker.count_tokens(output_str, self.model)
        
        # Get token usage from response if available
        input_tokens = estimated_input_tokens
        output_tokens = estimated_output_tokens
        
        # Extract token counts from response if available
        if hasattr(response, 'usage'):
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        
        # Track token usage with actual counts when available
        self.token_tracker.add_call("use_tool", self.model, input_tokens, output_tokens)
        
        # Log the tool response
        logger.debug(f"===== TOOL RESPONSE FROM OPENAI =====")
        logger.debug(f"Tool called: {function_call.name}")
        logger.debug(f"Parameters: {json.dumps(arguments, indent=2)}")
        logger.debug(f"Actual input tokens: {input_tokens}")
        logger.debug(f"Actual output tokens: {output_tokens}")
        logger.debug("===================================")
        
        # Log token usage if in debug mode
        self.token_tracker.log_report()
        
        return tool_called
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Get the current token usage statistics"""
        return self.token_tracker.get_usage_report()
    
    def reset_token_usage(self):
        """Reset token usage statistics"""
        self.token_tracker.reset()
    
    def log_token_usage(self, log_level=logging.DEBUG):
        """Log token usage report at the specified level"""
        self.token_tracker.log_report(log_level)

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
    
    # Log final token usage report
    client.log_token_usage(logging.INFO)
    

if __name__ == "__main__":
    test()

