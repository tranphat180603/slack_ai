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

# Add a new CancellationError class
class CancellationError(Exception):
    """Exception raised when an operation is cancelled."""
    pass

class TokenUsageTracker:
    """Tracks token usage for OpenAI API calls"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset usage statistics"""
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0  # Add tracking for reasoning tokens
        self.cached_tokens = 0  # Add tracking for cached tokens
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
    
    def add_call(self, function_name: str, model: str, input_tokens: int, output_tokens: int = 0, reasoning_tokens: int = 0, cached_tokens: int = 0):
        """Record a new API call with token usage"""
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.reasoning_tokens += reasoning_tokens  # Track reasoning tokens
        self.cached_tokens += cached_tokens  # Track cached tokens
        
        # Track model used
        self.last_tracked_model = model
        self.last_tracked_function = function_name
        
        # Track by model
        if model not in self.model_counts:
            self.model_counts[model] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}
        
        self.model_counts[model]["calls"] += 1
        self.model_counts[model]["input_tokens"] += input_tokens
        self.model_counts[model]["output_tokens"] += output_tokens
        self.model_counts[model]["reasoning_tokens"] += reasoning_tokens
        self.model_counts[model]["cached_tokens"] += cached_tokens
        
        # Track by function
        if function_name not in self.function_counts:
            self.function_counts[function_name] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}
            
        self.function_counts[function_name]["calls"] += 1
        self.function_counts[function_name]["input_tokens"] += input_tokens
        self.function_counts[function_name]["output_tokens"] += output_tokens
        self.function_counts[function_name]["reasoning_tokens"] += reasoning_tokens
        self.function_counts[function_name]["cached_tokens"] += cached_tokens
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Get a report of token usage"""
        total_tokens = self.input_tokens + self.output_tokens
        
        # Include cached tokens in the total calculation when displaying to user
        # (They're already included in input_tokens, but we display them separately)
        
        return {
            "total_calls": self.calls,
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "total_reasoning_tokens": self.reasoning_tokens,
            "total_cached_tokens": self.cached_tokens,
            "total_tokens": total_tokens,
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
            "gpt-4o": {"input": 2.5, "cached": 1.25, "output": 10.0},    # $2.5/1M input, $1.25/1M cached, $10/1M output
            "gpt-4o-mini": {"input": 0.15, "cached": 0.075, "output": 0.6}, # $0.15/1M input, $0.075/1M cached, $0.6/1M output
            "o1-mini": {"input": 3.0, "cached": 1.5, "output": 15.0},    # Claude 3.5 Sonnet ($3/1M input, $1.5/1M cached, $15/1M output)
            "o1": {"input": 15.0, "cached": 7.5, "output": 60.0},        # Claude 3.5 Opus ($15/1M input, $7.5/1M cached, $60/1M output)
            "o3-mini": {"input": 1.1, "cached": 0.55, "output": 4.4},    # Claude 3 Haiku ($1.1/1M input, $0.55/1M cached, $4.4/1M output)
            "o3": {"input": 3.0, "cached": 1.5, "output": 15.0}          # Claude 3 Sonnet ($3/1M input, $1.5/1M cached, $15/1M output)
        }
        
        # Default cost rates if specific model not found
        default_cost = {"input": 1.0, "cached": 0.5, "output": 2.0}
        
        for model, usage in self.model_counts.items():
            # Get cost rates for this model, or use default
            model_base = model.split("-")[0]  # Handle model variants
            cost_rates = costs.get(model, default_cost)
            
            # Calculate cost - accounting for both regular and cached tokens
            regular_input_tokens = usage["input_tokens"] - usage["cached_tokens"]
            cached_tokens = usage["cached_tokens"]
            output_tokens = usage["output_tokens"]
            
            # Calculate costs for each token type
            input_cost = (regular_input_tokens / 1_000_000) * cost_rates["input"]
            cached_cost = (cached_tokens / 1_000_000) * cost_rates["cached"]
            output_cost = (output_tokens / 1_000_000) * cost_rates["output"]
            
            # Add to total cost
            cost += input_cost + cached_cost + output_cost
            
        return cost
    
    def log_report(self, log_level=logging.DEBUG):
        """Log token usage report at the specified level"""
        report = self.get_usage_report()
        
        if logger.isEnabledFor(log_level):
            logger.log(log_level, "===== TOKEN USAGE REPORT =====")
            logger.log(log_level, f"Total API calls: {report['total_calls']}")
            logger.log(log_level, f"Total tokens: {report['total_tokens']} (Input: {report['total_input_tokens']}, Output: {report['total_output_tokens']})")
            if report['total_cached_tokens'] > 0:
                logger.log(log_level, f"Total cached tokens: {report['total_cached_tokens']} (billed at reduced rate)")
            if report['total_reasoning_tokens'] > 0:
                logger.log(log_level, f"Total reasoning tokens: {report['total_reasoning_tokens']}")
            logger.log(log_level, f"Estimated cost: ${report['estimated_cost_usd']:.5f} USD")
            
            logger.log(log_level, "Model breakdown:")
            for model, usage in report['model_breakdown'].items():
                cached_info = f", cached: {usage.get('cached_tokens', 0)}" if usage.get('cached_tokens', 0) > 0 else ""
                reasoning_info = f", reasoning: {usage.get('reasoning_tokens', 0)}" if usage.get('reasoning_tokens', 0) > 0 else ""
                logger.log(log_level, f"  {model}: {usage['calls']} calls, {usage['input_tokens']} input, {usage['output_tokens']} output tokens{cached_info}{reasoning_info}")
            
            logger.log(log_level, "Function breakdown:")
            for func, usage in report['function_breakdown'].items():
                cached_info = f", cached: {usage.get('cached_tokens', 0)}" if usage.get('cached_tokens', 0) > 0 else ""
                reasoning_info = f", reasoning: {usage.get('reasoning_tokens', 0)}" if usage.get('reasoning_tokens', 0) > 0 else ""
                logger.log(log_level, f"  {func}: {usage['calls']} calls, {usage['input_tokens']} input, {usage['output_tokens']} output tokens{cached_info}{reasoning_info}")
            
            logger.log(log_level, "===============================")


class OpenaiClient(ABC):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.token_tracker = TokenUsageTracker()

    def response(self, prompt: str, system_prompt: str = None, image_data: str = None, stream: bool = False) -> str:
        assert self.model.startswith("gpt"), "Only GPT models are supported"
        messages = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt
                }
            )
        
        # Log image data availability for debugging
        if image_data:
            logger.info(f"Image data is provided, length: {len(image_data)} bytes")
        else:
            logger.warning("No image_data provided to response method or it's empty")
        
        # Handle image input if provided
        if image_data and (
            "vision" in self.model or
            self.model.startswith("gpt-4o") or  # This will match gpt-4o, gpt-4o-mini, gpt-4o-2024-11-20, etc.
            "gpt-4" in self.model  # For other gpt-4 variants with vision
        ):
            logger.info(f"Adding image content to message with model {self.model}")
            
            # Create message with text and image content
            image_content = {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{image_data}"
            }
            
            text_content = {
                "type": "input_text",
                "text": prompt
            }
            
            # Log the structure being created
            logger.debug(f"Text content: {text_content}")
            logger.debug(f"Image content type: {image_content['type']}")
            logger.debug(f"Image URL starts with: {image_content['image_url'][:30]}...")
            
            messages.append({
                "role": "user",
                "content": [text_content, image_content]
            })
            
            logger.info(f"Including image data in response request with model {self.model}")
        else:
            # Check why we're not including image (for debugging)
            if image_data:
                logger.error(f"CRITICAL: Image data provided but not included in message. Model={self.model}")
                
                # Check what condition failed
                if not "vision" in self.model and not self.model.startswith("gpt-4o") and not "gpt-4" in self.model:
                    logger.error(f"Model check failed. Model {self.model} doesn't match any vision-compatible pattern")
                else:
                    logger.error(f"Unexpected error: Model {self.model} should be vision-compatible, but image wasn't added")
            
            # Regular text-only message
            messages.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )
        
        # Count input tokens for logging only (will be replaced with actual count from API)
        input_text = (system_prompt or "") + prompt
        estimated_input_tokens = self.token_tracker.count_tokens(input_text, self.model)
        
        # Add estimate for image tokens if image is provided (roughly 500 tokens per image)
        if image_data:
            estimated_input_tokens += 500
        
        # Reset last_tracked model to detect if we'll get usage info
        self.token_tracker.last_tracked_model = None
        self.token_tracker.last_tracked_function = None
        
        # Debug logging
        logger.debug(f"Using responses.create with model={self.model}")
                
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
                    if hasattr(chunk, 'response') and hasattr(chunk.response, 'usage'):
                        # Extract usage information
                        usage = chunk.response.usage
                        input_tokens = usage.input_tokens
                        output_tokens = usage.output_tokens
                        cached_tokens = 0
                        
                        # Extract cached tokens if available
                        if hasattr(usage, 'input_tokens_details') and hasattr(usage.input_tokens_details, 'cached_tokens'):
                            cached_tokens = usage.input_tokens_details.cached_tokens
                            logger.debug(f"Found {cached_tokens} cached tokens in response")
                        
                        # Track token usage with actual counts from API
                        self.token_tracker.add_call("response", self.model, input_tokens, output_tokens, 0, cached_tokens)
            print()  # Add a newline at the end
            
            # If we didn't get usage from API, use estimates as fallback
            if self.token_tracker.last_tracked_function != "response" or self.token_tracker.last_tracked_model != self.model:
                output_tokens = self.token_tracker.count_tokens(full_response, self.model)
                # Track token usage with estimated counts
                self.token_tracker.add_call("response", self.model, estimated_input_tokens, output_tokens, 0, 0)
            
            return full_response
        else:
            try:
                # For GPT models, extract text content
                result = response.output[0].content[0].text
                
                # Extract token usage from API response
                input_tokens = estimated_input_tokens
                output_tokens = self.token_tracker.count_tokens(result, self.model)
                cached_tokens = 0
                
                if hasattr(response, 'usage'):
                    # Extract actual token usage from the response object
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                    
                    # Extract cached tokens if available
                    if hasattr(response.usage, 'input_tokens_details') and hasattr(response.usage.input_tokens_details, 'cached_tokens'):
                        cached_tokens = response.usage.input_tokens_details.cached_tokens
                        logger.debug(f"Found {cached_tokens} cached tokens in response")
                
                # Track token usage
                self.token_tracker.add_call("response", self.model, input_tokens, output_tokens, 0, cached_tokens)
                
                return result
            except Exception as e:
                # Log full exception details for debugging
                logger.error(f"Error extracting response content: {str(e)}")
                logger.error(f"Response type: {type(response)}")
                
                # Try to extract useful information from the response object
                try:
                    if hasattr(response, 'output'):
                        logger.error(f"Response output type: {type(response.output)}")
                        logger.error(f"Response output: {response.output}")
                    
                    # Extract any error message
                    if hasattr(response, 'error'):
                        logger.error(f"API error: {response.error}")
                except Exception as inner_e:
                    logger.error(f"Error extracting response details: {str(inner_e)}")
                
                # Fallback - use string representation
                result = str(response)
                output_tokens = self.token_tracker.count_tokens(result, self.model)
                self.token_tracker.add_call("response", self.model, estimated_input_tokens, output_tokens, 0, 0)
                
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
        
        # Reset last_tracked model to detect if we'll get usage info
        self.token_tracker.last_tracked_model = None
        self.token_tracker.last_tracked_function = None
        
        # Debug logging
        logger.debug(f"Using responses.create with model={self.model} and reasoning.effort={reasoning_effort}")
        
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
                        reasoning_tokens = 0
                        cached_tokens = 0
                        
                        # Extract cached tokens if available
                        if hasattr(usage, 'input_tokens_details') and hasattr(usage.input_tokens_details, 'cached_tokens'):
                            cached_tokens = usage.input_tokens_details.cached_tokens
                            logger.debug(f"Found {cached_tokens} cached tokens in response")
                        
                        # Extract reasoning tokens if available
                        if hasattr(usage, 'output_tokens_details') and hasattr(usage.output_tokens_details, 'reasoning_tokens'):
                            reasoning_tokens = usage.output_tokens_details.reasoning_tokens
                        
                        # Track token usage with actual counts from API
                        self.token_tracker.add_call("response_reasoning", self.model, input_tokens, output_tokens, reasoning_tokens, cached_tokens)
            print()  # Add a newline at the end
            
            # If we didn't get usage from API, use estimates as fallback
            if self.token_tracker.last_tracked_function != "response_reasoning" or self.token_tracker.last_tracked_model != self.model:
                output_tokens = self.token_tracker.count_tokens(full_response, self.model)
                # Track token usage with estimated counts
                self.token_tracker.add_call("response_reasoning", self.model, estimated_input_tokens, output_tokens, 0, 0)
            
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
                                    reasoning_tokens = 0
                                    cached_tokens = 0
                                    
                                    # Extract token counts from response if available
                                    if hasattr(response, 'usage'):
                                        input_tokens = response.usage.input_tokens
                                        output_tokens = response.usage.output_tokens
                                        
                                        # Extract cached tokens if available
                                        if hasattr(response.usage, 'input_tokens_details') and hasattr(response.usage.input_tokens_details, 'cached_tokens'):
                                            cached_tokens = response.usage.input_tokens_details.cached_tokens
                                            logger.debug(f"Found {cached_tokens} cached tokens in response")
                                        
                                        # Extract reasoning tokens if available
                                        if hasattr(response.usage, 'output_tokens_details') and hasattr(response.usage.output_tokens_details, 'reasoning_tokens'):
                                            reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
                                    
                                    # Track token usage with actual counts
                                    self.token_tracker.add_call("response_reasoning", self.model, input_tokens, output_tokens, reasoning_tokens, cached_tokens)
                                    
                                    return result
                
                # If we got here, we didn't find the expected structure
                logger.warning("Could not find text in the message content, trying output_text attribute")
                if hasattr(response, 'output_text'):
                    result = response.output_text
                    
                    # Get token usage from response if available
                    input_tokens = estimated_input_tokens
                    output_tokens = self.token_tracker.count_tokens(str(result), self.model)
                    reasoning_tokens = 0
                    cached_tokens = 0
                    
                    # Extract token counts from response if available
                    if hasattr(response, 'usage'):
                        input_tokens = response.usage.input_tokens
                        output_tokens = response.usage.output_tokens
                        
                        # Extract cached tokens if available
                        if hasattr(response.usage, 'input_tokens_details') and hasattr(response.usage.input_tokens_details, 'cached_tokens'):
                            cached_tokens = response.usage.input_tokens_details.cached_tokens
                            logger.debug(f"Found {cached_tokens} cached tokens in response")
                        
                        # Extract reasoning tokens if available
                        if hasattr(response.usage, 'output_tokens_details') and hasattr(response.usage.output_tokens_details, 'reasoning_tokens'):
                            reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
                    
                    # Track token usage
                    self.token_tracker.add_call("response_reasoning", self.model, input_tokens, output_tokens, reasoning_tokens, cached_tokens)
                    
                    return result
                    
                # Final fallback
                logger.warning("Unusual response structure, returning string representation of output")
                result = str(response.output)
                
                # Get token usage from response if available
                input_tokens = estimated_input_tokens
                output_tokens = self.token_tracker.count_tokens(result, self.model)
                reasoning_tokens = 0
                cached_tokens = 0
                
                # Extract token counts from response if available
                if hasattr(response, 'usage'):
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                    
                    # Extract cached tokens if available
                    if hasattr(response.usage, 'input_tokens_details') and hasattr(response.usage.input_tokens_details, 'cached_tokens'):
                        cached_tokens = response.usage.input_tokens_details.cached_tokens
                        logger.debug(f"Found {cached_tokens} cached tokens in response")
                    
                    # Extract reasoning tokens if available
                    if hasattr(response.usage, 'output_tokens_details') and hasattr(response.usage.output_tokens_details, 'reasoning_tokens'):
                        reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
                
                # Track token usage
                self.token_tracker.add_call("response_reasoning", self.model, input_tokens, output_tokens, reasoning_tokens, cached_tokens)
                
                return result
            
            logger.error("Failed to extract text from reasoning response")
            
            # Track token usage for error case with estimated values
            self.token_tracker.add_call("response_reasoning", self.model, estimated_input_tokens, 0, 0, 0)
            
            return "Error: Could not extract response text"
        
    def use_tool(self, prompt: str, reasoning_effort: str, tools: List[Dict[str, Any]] = None, stream: bool = False) -> Dict[str, Any]:
        assert self.model.startswith("o"), "Only reasoning models are supported"
        
        # Count input tokens for logging only (will be replaced with actual count from API)
        input_text = prompt + json.dumps(tools, ensure_ascii=False)
        estimated_input_tokens = self.token_tracker.count_tokens(input_text, self.model)
        
        # Reset last_tracked model to detect if we'll get usage info
        self.token_tracker.last_tracked_model = None
        self.token_tracker.last_tracked_function = None
        
        # Debug logging
        logger.debug(f"Using responses.create with model={self.model}, reasoning.effort={reasoning_effort}, and tools={len(tools or [])}")
        
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
            
            # Track token usage for error case
            self.token_tracker.add_call("use_tool", self.model, estimated_input_tokens, 0, 0, 0)
            
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
        reasoning_tokens = 0
        cached_tokens = 0
        
        # Extract token counts from response if available
        if hasattr(response, 'usage'):
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            # Extract cached tokens if available
            if hasattr(response.usage, 'input_tokens_details') and hasattr(response.usage.input_tokens_details, 'cached_tokens'):
                cached_tokens = response.usage.input_tokens_details.cached_tokens
                logger.debug(f"Found {cached_tokens} cached tokens in use_tool response")
            
            # Extract reasoning tokens if available
            if hasattr(response.usage, 'output_tokens_details') and hasattr(response.usage.output_tokens_details, 'reasoning_tokens'):
                reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
        
        # Track token usage with actual counts when available
        self.token_tracker.add_call("use_tool", self.model, input_tokens, output_tokens, reasoning_tokens, cached_tokens)
        
        return tool_called
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Get the current token usage statistics"""
        return self.token_tracker.get_usage_report()
    
    def reset_token_usage(self):
        """Reset token usage statistics"""
        self.token_tracker.reset()
    
    def log_token_usage(self, log_level=logging.DEBUG):
        """Log token usage report at the specified level"""
        if not logger.isEnabledFor(log_level):
            return
            
        # Log usage for the specific model being used
        if self.model in self.token_tracker.model_counts:
            usage = self.token_tracker.model_counts[self.model]
            logger.log(log_level, f"Token usage for {self.model}:")
            logger.log(log_level, f"  Calls: {usage['calls']}")
            logger.log(log_level, f"  Input tokens: {usage['input_tokens']}")
            
            # Always show cached tokens even at higher log levels
            cached_tokens = usage.get('cached_tokens', 0)
            if cached_tokens > 0:
                logger.log(log_level, f"  Cached tokens: {cached_tokens} (billed at reduced rate)")
            
            logger.log(log_level, f"  Output tokens: {usage['output_tokens']}")
            
            if usage.get('reasoning_tokens', 0) > 0:
                logger.log(log_level, f"  Reasoning tokens: {usage['reasoning_tokens']}")
            
            # Get model-specific costs
            model_base = self.model.split("-")[0]
            costs = {
                "gpt-4o": {"input": 2.5, "cached": 1.25, "output": 10.0},
                "gpt-4o-mini": {"input": 0.15, "cached": 0.075, "output": 0.6},
                "o1-mini": {"input": 3.0, "cached": 1.5, "output": 15.0},
                "o1": {"input": 15.0, "cached": 7.5, "output": 60.0},
                "o3-mini": {"input": 1.1, "cached": 0.55, "output": 4.4},
                "o3": {"input": 3.0, "cached": 1.5, "output": 15.0}
            }
            default_cost = {"input": 1.0, "cached": 0.5, "output": 2.0}
            cost_rates = costs.get(self.model, default_cost)
            
            # Calculate regular and cached token costs
            regular_input_tokens = usage["input_tokens"] - usage["cached_tokens"]
            cached_tokens = usage["cached_tokens"]
            output_tokens = usage["output_tokens"]
            
            regular_cost = (regular_input_tokens / 1_000_000) * cost_rates["input"]
            cached_cost = (cached_tokens / 1_000_000) * cost_rates["cached"]
            output_cost = (output_tokens / 1_000_000) * cost_rates["output"]
            total_cost = regular_cost + cached_cost + output_cost
            
            # Log total tokens and cost breakdown
            logger.log(log_level, f"  Total tokens: {usage['input_tokens'] + usage['output_tokens']}")
            if cached_tokens > 0:
                logger.log(log_level, f"  Cost breakdown: ${regular_cost:.6f} (regular input) + ${cached_cost:.6f} (cached input) + ${output_cost:.6f} (output) = ${total_cost:.6f}")
            else:
                logger.log(log_level, f"  Cost breakdown: ${regular_cost:.6f} (input) + ${output_cost:.6f} (output) = ${total_cost:.6f}")
        else:
            # If no usage for this model yet
            logger.log(log_level, f"No token usage recorded for {self.model}")
            
        # Full report at any log level - not just debug
        self.token_tracker.log_report(log_level)

class CancellableOpenAIClient(OpenaiClient):
    """OpenAI client with cancellation support for long-running operations."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", context_id: str = None):
        """
        Initialize a cancellable OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use
            context_id: ID of the context to check for cancellation
        """
        super().__init__(api_key, model)
        self.context_id = context_id
        
    def set_context_id(self, context_id: str):
        """Set the context ID to use for cancellation checks."""
        self.context_id = context_id
        
    def is_cancelled(self) -> bool:
        """Check if the current operation has been cancelled."""
        if not self.context_id:
            return False
        
        try:
            from app.TMAI_slack_agent import context_manager
            context = context_manager.get_context(self.context_id)
            return context and context.get('stop_requested', False)
        except ImportError:
            logger.warning("Could not import context_manager, cancellation checks disabled")
            return False
        except Exception as e:
            logger.error(f"Error checking cancellation status: {e}")
            return False
    
    def response(self, prompt: str, system_prompt: str = None, image_data: str = None, stream: bool = False) -> str:
        """Enhanced response method with cancellation checks."""
        # Check for cancellation before API call
        if self.is_cancelled():
            logger.info(f"Operation cancelled before API call (context_id={self.context_id})")
            raise CancellationError("Operation was cancelled")
            
        # Call parent implementation
        result = super().response(prompt, system_prompt, image_data, stream)
        
        # Check again after API call
        if self.is_cancelled():
            logger.info(f"Operation cancelled after API call (context_id={self.context_id})")
            raise CancellationError("Operation was cancelled")
            
        return result
    
    def response_reasoning(self, prompt: str, reasoning_effort: str, stream: bool = False) -> str:
        """Enhanced reasoning response method with cancellation checks."""
        # Check for cancellation before API call
        if self.is_cancelled():
            logger.info(f"Reasoning operation cancelled before API call (context_id={self.context_id})")
            raise CancellationError("Operation was cancelled")
            
        # Call parent implementation
        result = super().response_reasoning(prompt, reasoning_effort, stream)
        
        # Check again after API call
        if self.is_cancelled():
            logger.info(f"Reasoning operation cancelled after API call (context_id={self.context_id})")
            raise CancellationError("Operation was cancelled")
            
        return result
    
    def use_tool(self, prompt: str, reasoning_effort: str, tools: List[Dict[str, Any]] = None, stream: bool = False) -> Dict[str, Any]:
        """Enhanced tool use method with cancellation checks."""
        # Check for cancellation before API call
        if self.is_cancelled():
            logger.info(f"Tool use operation cancelled before API call (context_id={self.context_id})")
            raise CancellationError("Operation was cancelled")
            
        # Call parent implementation
        result = super().use_tool(prompt, reasoning_effort, tools, stream)
        
        # Check again after API call
        if self.is_cancelled():
            logger.info(f"Tool use operation cancelled after API call (context_id={self.context_id})")
            raise CancellationError("Operation was cancelled")
            
        return result

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
    tool_response = client.response_reasoning("How to solve the problem of creating suitable context for an agent to use? What are the best practices?", stream=False, reasoning_effort="low")
    
    print(tool_response)
    

if __name__ == "__main__":
    test()

