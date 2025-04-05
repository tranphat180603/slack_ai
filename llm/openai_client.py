"""
OpenAI API client module for TMAI Agent.
This module handles all interactions with OpenAI APIs, including:
- Chat completions
- Function calling
- Rate limiting
- Error handling
- Model configuration
"""

import os
import time
import logging
import traceback
import json
from typing import Dict, List, Optional, Union, Any, Callable

import openai
from openai import OpenAI

# Configure logger
logger = logging.getLogger("openai_client")

# Default model settings
DEFAULT_MODEL = os.environ.get("AI_MODEL", "o3-mini")
DEFAULT_TEMPERATURE = 0.7

class OpenAIClient:
    """
    Client for handling OpenAI API calls with rate limiting and error handling.
    
    This class provides methods for:
    1. Chat completions
    2. Function calling with the OpenAI API
    3. Streaming responses
    4. Proper error handling and retries
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: Optional OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
            model: Model ID to use for completions (default: from AI_MODEL env var or gpt-4o)
            temperature: Temperature setting for completions (default: 0.7)
            max_retries: Maximum number of retries for failed requests (default: 3)
            retry_delay: Base delay between retries in seconds (default: 1.0)
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key)
        
        # Store configuration
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Rate limit tracking
        self.last_request_time = 0
        self.min_request_interval = 0.05  # 50ms minimum between requests
        
        logger.info(f"OpenAI client initialized with model: {self.model}")
    
    def _handle_retry(self, attempt: int, exception: Exception) -> float:
        """
        Handle retry logic with exponential backoff.
        
        Args:
            attempt: Current attempt number (0-indexed)
            exception: The exception that triggered the retry
            
        Returns:
            Delay in seconds before next retry
        """
        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
        
        # Add jitter to avoid thundering herd
        import random
        delay = delay * (0.5 + random.random())
        
        # Add longer delay for rate limit errors
        if hasattr(exception, 'code') and exception.code == 'rate_limit_exceeded':
            delay = max(delay, 5.0)
            
        logger.warning(f"Request failed with {type(exception).__name__}: {str(exception)}. "
                      f"Retrying in {delay:.2f}s (attempt {attempt+1}/{self.max_retries})")
        
        return delay

    async def chat_with_history(
        self,
        history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
        model: Optional[str] = None
    ) -> Union[str, Any]:
        """
        Generate a chatbot response based on conversation history.
        
        Args:
            history: List of conversation messages in [{"role": "user", "content": "..."}, ...] format
            system_prompt: Optional system prompt to prepend
            temperature: Temperature for response generation
            stream: Whether to stream the response
            stream_callback: Optional callback function for handling streamed chunks
            model: Optional model override to use for this request
            
        Returns:
            Full response text if stream=False, or streaming response object if stream=True
        """
        # Use provided model or fall back to instance default
        current_model = model or self.model
        
        # Prepare messages based on model type
        messages = []
        
        # Handle different model types differently
        if current_model.startswith("o"):
            # For "o" models like o3-mini that don't support system prompts,
            # we prepend the system prompt to the first user message
            if system_prompt and history:
                modified_history = history.copy()
                if modified_history[0]["role"] == "user":
                    modified_history[0]["content"] = f"{system_prompt}\n\n{modified_history[0]['content']}"
                else:
                    # If the first message isn't from a user, add a new one at the beginning
                    modified_history.insert(0, {
                        "role": "user",
                        "content": f"{system_prompt}\n\nPlease respond to: {modified_history[0]['content']}"
                    })
                messages = modified_history
            else:
                messages = history
        else:
            # For models that support system prompts (e.g., gpt-4)
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history)
        
        logger.info(f"Sending chat request with {len(messages)} messages using model {current_model}")
        
        # Handle streaming vs non-streaming
        if stream:
            return await self.stream_chat_completion(
                messages=messages,
                temperature=temperature,
                model=current_model,
                callback=stream_callback
            )
        else:
            response = await self.chat_completion(
                messages=messages,
                temperature=temperature,
                model=current_model
            )
            
            # Extract and return the content from the new response format
            if hasattr(response, 'output'):
                for output_item in response.output:
                    if hasattr(output_item, 'type') and output_item.type == 'text':
                        return output_item.text
            
            return ""
    
    async def execute_tool_call(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        previous_results: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool call using OpenAI's function calling.
        
        Args:
            query: The user's query or instruction
            tools: List of tool specifications
            system_prompt: Optional system prompt to guide tool execution
            context: Additional context for tool execution
            previous_results: Results from previous tool calls
            temperature: Temperature setting for response generation
            
        Returns:
            Dictionary containing tool call details and parameters
        """


        current_model = model or self.model
        # Build user message with query and any additional context
        user_content = query
        if context:
            user_content = f"{query}\n\nContext:\n{context}"
        
        # Add information about previous results if available
        if previous_results:
            user_content += f"\n\nPrevious tool results:\n{json.dumps(previous_results, indent=2)}"
            
        # Prepare messages based on model type
        messages = []
        
        # Handle different model types differently
        if current_model.startswith("o"):
            # For "o" models like o3-mini that don't support system prompts,
            # we prepend the system prompt to the user message
            if system_prompt:
                user_content = f"{system_prompt}\n\n{user_content}"
            messages.append({"role": "user", "content": user_content})
        else:
            # For models that support system prompts (e.g., gpt-4)
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_content})
        
        logger.info(f"Executing tool call with {len(tools)} available tools")

        logger.debug(f"LLM REQUEST - Tools: {tools}")
        
        # Make the API call
        response = await self.chat_completion(
            messages=messages,
            tools=tools,
            temperature=temperature
        )
        
        # Extract and process the tool call
        if hasattr(response, 'output') and response.output:
            for output_item in response.output:
                if hasattr(output_item, 'type') and output_item.type == 'function_call':
                    return {
                        "tool_name": output_item.name,
                        "parameters": json.loads(output_item.arguments),
                        "id": output_item.call_id
                    }
            
            # If we get here, there was no function call in the response
            # Check if there's text output
            for output_item in response.output:
                if hasattr(output_item, 'type') and output_item.type == 'text':
                    return {
                        "error": "No tool call found in response",
                        "content": output_item.text
                    }
        
        # If no output or no recognizable output format
        return {
            "error": "No tool call found in response",
            "content": None
        }

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None
    ) -> Any:
        """
        Make a chat completion request to OpenAI API.
        
        Args:
            messages: List of conversation messages
            temperature: Optional temperature override
            tools: Optional list of tools/functions to call
            model: Optional model override
            
        Returns:
            OpenAI chat completion response
        """
        try:
            # Respect rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            
            # Use provided values or fall back to instance defaults
            current_model = model or self.model
            current_temp = temperature if temperature is not None else self.temperature
            
            # Prepare API call parameters
            params = {
                "model": current_model,
                "input": messages,
            }
            
            # Only add temperature for non-'o' models
            if not current_model.startswith("o"):
                params["temperature"] = current_temp
            
            # Add tools if provided
            if tools:
                params["tools"] = tools
            
            # Log detailed request info
            logger.debug(f"LLM REQUEST - Model: {current_model}")
            logger.debug(f"LLM REQUEST - Messages: {json.dumps(messages, indent=2)}")
            if tools:
                logger.debug(f"LLM REQUEST - Tools: {json.dumps([t['name'] for t in tools])}")
            
            # Make the API call
            completion = self.client.responses.create(**params)
            
            # Log response details
            logger.debug(f"LLM RESPONSE: {completion}")
            
            # Update rate limit tracking
            self.last_request_time = time.time()
            
            return completion
            
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise 

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> Any:
        """
        Stream a chat completion from OpenAI API.
        
        Args:
            messages: List of conversation messages
            temperature: Optional temperature override
            model: Optional model override
            callback: Optional callback function for handling chunks
            
        Returns:
            Streaming response object
        """
        try:
            # Respect rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            
            # Use provided values or fall back to instance defaults
            current_model = model or self.model
            current_temp = temperature if temperature is not None else self.temperature
            
            # Prepare API call parameters
            params = {
                "model": current_model,
                "input": messages,
                "stream": True
            }
            
            # Only add temperature for non-'o' models
            if not current_model.startswith("o"):
                params["temperature"] = current_temp
            
            # Make the streaming API call
            stream = self.client.responses.stream(**params)
            
            # Update rate limit tracking
            self.last_request_time = time.time()
            
            # Handle the stream
            if callback:
                for chunk in stream:
                    if hasattr(chunk, 'text') and chunk.text:
                        callback(chunk.text)
            
            return stream
            
        except Exception as e:
            logger.error(f"Error in streaming chat completion: {str(e)}")
            raise 