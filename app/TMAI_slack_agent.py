"""
TMAI Slack Agent - Core agent functionality
Implements the three-phase workflow:
1. Planning - Determine which tools to use
2. Sequential Tool Execution - Execute each tool in sequence
3. Response Generation - Generate the final response
"""

import os
import time
import logging
import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Union, Callable

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from llm.openai_client import OpenAIClient
from conversation_db.database import get_db, load_conversation_from_db, save_conversation_to_db
from tools.tool_schema_linear import LINEAR_SCHEMAS
from tools.tool_schema_slack import SLACK_SCHEMAS  
from tools.tool_schema_semantic_search import SEMANTIC_SEARCH_SCHEMAS
from rate_limiter import global_limiter, slack_limiter, linear_limiter, openai_limiter
from linear_db.linear_client import LinearClient
from utils import safe_append, format_for_slack

# Configure logger
logger = logging.getLogger("tmai_agent")

# Only log errors and critical issues in production
if os.environ.get("ENVIRONMENT") == "production":
    logger.setLevel(logging.WARNING)
else:
    logger.setLevel(logging.INFO)

class AIRequest:
    """Model for AI request data received from Slack."""
    def __init__(
        self,
        text: str,
        user_id: str,
        channel_id: str,
        message_ts: Optional[str] = None,
        thread_ts: Optional[str] = None,
        files: Optional[List[Dict[str, Any]]] = None,
        urls: Optional[List[str]] = None,
        sender_name: str = "User"
    ):
        self.text = text
        self.user_id = user_id
        self.channel_id = channel_id
        self.message_ts = message_ts
        self.thread_ts = thread_ts
        self.files = files or []
        self.urls = urls or []
        self.sender_name = sender_name
class ProgressiveMessageHandler:
    """Manages progress updates for Slack messages."""
    
    def __init__(self, slack_client: WebClient, channel_id: str, message_ts: str, thread_ts: str):
        self.slack_client = slack_client
        self.channel_id = channel_id
        self.message_ts = message_ts
        self.thread_ts = thread_ts
        self.current_stage = "initialized"
        self.stages_completed = []
        self.stages_pending = []
    
    async def send_thinking_message(self, initial=True):
        """Send or update the thinking message."""
        blocks = self._build_thinking_blocks()
        
        try:
            if initial:
                response = await asyncio.to_thread(
                    self.slack_client.chat_postMessage,
                    channel=self.channel_id,
                    text="TMAI's neuron firing......",
                    thread_ts=self.thread_ts,
                    blocks=blocks
                )
                self.message_ts = response["ts"]
                return response
            else:
                return await asyncio.to_thread(
                    self.slack_client.chat_update,
                    channel=self.channel_id,
                    ts=self.message_ts,
                    text="TMAI's neuron firing",
                    blocks=blocks
                )
        except SlackApiError as e:
            logger.warning(f"Error updating message: {e.response.get('error', '')}")
    
    def update_stage(self, stage_name, completed=False):
        """Update the current processing stage."""
        self.current_stage = stage_name
        if completed and stage_name not in self.stages_completed:
            if stage_name in self.stages_pending:
                self.stages_pending.remove(stage_name)
            self.stages_completed.append(stage_name)
    
    def add_pending_stages(self, stages):
        """Add stages to the pending list."""
        for stage in stages:
            if stage not in self.stages_completed and stage not in self.stages_pending:
                self.stages_pending.append(stage)
    
    def _build_thinking_blocks(self):
        """Build the blocks for the thinking message."""
        # Construct status text showing completed and pending stages
        status_text = "*Status:*\n```\n"
        
        for stage in self.stages_completed:
            status_text += f"✓ {stage}\n"
        
        if self.current_stage in self.stages_pending:
            status_text += f"➤ {self.current_stage}...\n"
        
        for stage in self.stages_pending:
            if stage != self.current_stage:
                status_text += f"  ➤ {stage}\n"
                
        status_text += "```"
        
        return [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"\n{status_text}"
                },
                "accessory": {
                    "type": "image",
                    "image_url": "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExanRpdDc0enVuOXc3dG9vYWEzOGUyajFkOG03OHB6aTM4aTZhd2kycSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/IUNycHoVqvLDowiiam/giphy.gif",
                    "alt_text": "Processing"
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Harnessing the power of a trillion artificial neurons... and slightly fewer from my creator ◐"
                    }
                ]
            }
        ]
    
    async def send_rate_limit_message(self):
        """Send rate limit exceeded message."""
        try:
            await asyncio.to_thread(
                self.slack_client.chat_update,
                channel=self.channel_id,
                ts=self.message_ts,
                text="⚠️ Rate limit exceeded. Please try again in a minute."
            )
        except SlackApiError as e:
            logger.error(f"Error updating message with rate limit notice: {e.response['error']}")
    
    async def send_error_message(self, error_msg):
        """Send error message."""
        try:
            await asyncio.to_thread(
                self.slack_client.chat_postMessage,
                channel=self.channel_id,
                thread_ts=self.thread_ts,
                text=f"Sorry, I encountered an error: {error_msg}"
            )
            await asyncio.to_thread(
                self.slack_client.chat_delete,
                channel=self.channel_id,
                ts=self.message_ts
            )
        except SlackApiError as e:
            logger.error(f"Error sending error message: {e.response['error']}")
    
    async def delete_thinking_message(self):
        """Delete the thinking message."""
        try:
            await asyncio.to_thread(
                self.slack_client.chat_delete,
                channel=self.channel_id,
                ts=self.message_ts
            )
        except SlackApiError as e:
            logger.warning(f"Could not delete thinking message: {e.response.get('error', '')}")

class ConversationManager:
    """Manages conversation history, loading from memory, database, or Slack API."""
    
    def __init__(self, slack_client: WebClient, conversation_key: str, sender_name: str):
        self.slack_client = slack_client
        self.conversation_key = conversation_key
        self.sender_name = sender_name
        self.history = []
        self.conversation_history = {}  # In-memory cache
    
    async def load_conversation(self, channel_id: str, thread_ts: str, is_direct_message: bool = False):
        """Load conversation history from memory, database, or Slack API."""
        # First try memory cache
        if self.conversation_key in self.conversation_history and self.conversation_history[self.conversation_key]:
            self.history = self.conversation_history[self.conversation_key]
            logger.debug(f"Using in-memory conversation history with {len(self.history)} messages")
            return self._format_history_for_context(self.sender_name)
        
        # Then try database
        db_messages = load_conversation_from_db(channel_id, thread_ts)
        if db_messages:
            self.history = db_messages
            self.conversation_history[self.conversation_key] = db_messages
            logger.debug(f"Loaded conversation history from database with {len(db_messages)} messages")
            return self._format_history_for_context(self.sender_name)
        
        # Finally rebuild from Slack API
        try:
            response = self.slack_client.conversations_replies(
                channel=channel_id,
                ts=thread_ts
            )
            
            if response.get("ok") and response.get("messages"):
                messages = response.get("messages", [])
                
                # Different handling for direct messages vs mentions
                if is_direct_message:
                    # For DMs, reverse messages for chronological order
                    messages.reverse()
                else:
                    # For mentions, skip the first message
                    messages = messages[1:]
                
                self.conversation_history[self.conversation_key] = []
                
                for msg in messages:
                    # Determine if it's a user or bot message
                    is_bot = msg.get("bot_id") is not None
                    text = msg.get("text", "")
                    
                    # Skip empty messages
                    if not text:
                        continue
                        
                    # Skip processing messages
                    if is_bot and ("TMAI's neuron firing..." in text or "is thinking" in text):
                        continue
                    
                    # Skip default thread signal (only in DMs)
                    if is_direct_message and "New Assistant Thread" in text:
                        continue
                    
                    # Add to conversation history
                    self._add_message_internal(
                        "assistant" if is_bot else "user",
                        text,
                        msg.get("ts")
                    )
                
                self.history = self.conversation_history[self.conversation_key]
                logger.debug(f"Rebuilt conversation history with {len(self.history)} messages")
            else:
                logger.warning(f"Failed to get thread replies: {response.get('error', 'Unknown error')}")
                
        except SlackApiError as e:
            logger.error(f"Error getting thread replies: {e.response['error']}")
        
        return self._format_history_for_context(self.sender_name)
    
    async def add_message(self, role: str, content: str, message_ts: Optional[str] = None, metadata: Optional[Dict] = None):
        """Add a message to the conversation history."""
        self._add_message_internal(role, content, message_ts, metadata)
        
        # Save to database
        try:
            parts = self.conversation_key.split(":")
            if len(parts) == 2:
                channel_id, thread_ts = parts
                save_conversation_to_db(channel_id, thread_ts, self.conversation_history[self.conversation_key])
        except Exception as e:
            logger.error(f"Error saving conversation to database: {str(e)}")
    
    def _add_message_internal(self, role: str, content: str, message_ts: Optional[str] = None, metadata: Optional[Dict] = None):
        """Internal method to add message to memory cache."""
        if self.conversation_key not in self.conversation_history:
            self.conversation_history[self.conversation_key] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "message_ts": message_ts,
            "metadata": metadata or {}
        }
        
        self.conversation_history[self.conversation_key].append(message)
        self.history = self.conversation_history[self.conversation_key]
    
    def _format_history_for_context(self, sender_name: str = "User", max_messages: int = 10):
        """Format conversation history for context."""
        if not self.history:
            return []
        
        history_context = []
        
        # Limit to max_messages most recent messages
        messages = self.history
        if len(messages) > max_messages:
            messages = messages[-max_messages:]
            history_context = [f"**Conversation History:** Showing only the {max_messages} most recent messages."]
        else:
            history_context = [f"**Here is the conversation history between you and {sender_name} so far:**"]
        
        # Format the messages for context
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                # Format user messages with sender name
                msg_content = msg['content']
                history_context.append(f"**{sender_name} (#{i} turn):** {msg_content}")
            else:
                # Format assistant messages
                content = msg["content"]
                if len(content) > 500:
                    content = content[:500] + "... (content truncated)"
                history_context.append(f"**Assistant (#{i - 1} turn):** {content}")
        
        return history_context

class TMAISlackAgent:
    """
    Main agent class for the TMAI Slack Assistant.
    Implements the three-phase workflow for processing messages.
    """
    
    def __init__(
        self,
        slack_bot_token: str,
        slack_user_token: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        ai_model: str = "o3-mini",
        prompts: Dict = None,
        model_config: Optional[Dict[str, str]] = None
    ):
        """Initialize the agent with required clients and configuration."""
        # Store tokens
        self.slack_bot_token = slack_bot_token
        self.slack_user_token = slack_user_token
        
        # Initialize clients
        self.slack_client = WebClient(token=slack_bot_token)
        self.slack_user_client = WebClient(token=slack_user_token) if slack_user_token else None
        self.openai_client = OpenAIClient(api_key=openai_api_key, model=ai_model)
        
        # Store default model
        self.default_model = ai_model
        
        # Store model configuration for different phases
        self.model_config = model_config or {
            "planning": ai_model,
            "tool_execution": ai_model,
            "evaluation": ai_model,
            "response_generation": ai_model
        }
        
        # Initialize LinearClient properly
        linear_api_key = os.environ.get("LINEAR_API_KEY", "")
        if not linear_api_key:
            logger.warning("LINEAR_API_KEY not found in environment, LinearClient may not work properly")
        self.linear_client = LinearClient(linear_api_key)
        
        # Initialize SlackClient for tool execution
        from slack_ops.slack_tools import SlackClient
        self.slack_tool_client = SlackClient(slack_bot_token, slack_user_token)
        
        # Load prompts
        self.prompts = prompts or {}
        
        # Tool mapping
        self.tools_map = {
            **LINEAR_SCHEMAS,
            **SLACK_SCHEMAS,
            **SEMANTIC_SEARCH_SCHEMAS
        }
        
        # Initialize conversation store
        self.conversation_history = {}
        
        logger.debug("TMAI Slack Agent initialized")
    
    def parse_user_mentions(self, text: str) -> str:
        """Convert Slack user mentions to actual display names."""
        if not hasattr(self, '_user_cache'):
            self._user_cache = {}
            
        mention_pattern = r'<@([A-Z0-9]+)>'
        mentions = re.findall(mention_pattern, text)
        
        try:
            for user_id in mentions:
                display_name = None
                
                if user_id in self._user_cache:
                    display_name = self._user_cache[user_id]
                else:
                    try:
                        if not slack_limiter.check_rate_limit():
                            logger.warning("Slack API rate limit exceeded, waiting...")
                            slack_limiter.wait_if_needed()
                            
                        user_info = self.slack_client.users_info(user=user_id)
                        if user_info["ok"]:
                            display_name = (
                                user_info["user"]["profile"].get("display_name") or
                                user_info["user"]["profile"].get("real_name") or
                                f"<@{user_id}>"
                            )
                            self._user_cache[user_id] = display_name
                    except SlackApiError as e:
                        logger.error(f"Error getting user info for {user_id}: {str(e)}")
                        display_name = f"<@{user_id}>"
                
                if display_name:
                    text = text.replace(f"<@{user_id}>", display_name)
        except Exception as e:
            logger.error(f"Error in parse_user_mentions: {str(e)}")
            
        return text
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        if not text:
            return []
            
        url_pattern = r'https?://[^\s<>"\']+'
        urls = re.findall(url_pattern, text)
        
        logger.debug(f"Extracted {len(urls)} URLs")
        return urls
    
    async def process_slack_message(self, ai_request: AIRequest, thread_ts: Optional[str] = None, 
                               is_direct_message: bool = False, model_overrides: Optional[Dict[str, str]] = None):
        """
        Unified function to process both direct messages and mentions.
        
        Args:
            ai_request: The AI request object containing user, channel, text
            thread_ts: Optional thread timestamp (required for mentions)
            is_direct_message: Whether this is a direct message
            model_overrides: Optional dict to override models for specific phases
                             e.g. {"planning": "gpt-4o", "response_gener`ation": "claude-3-sonnet-20240229"}
        """
        start_time = time.time()
        logger.info(f"Processing message from user {ai_request.user_id}")
        
        # Apply any model overrides
        model_overrides = model_overrides or {}
        if model_overrides:
            # Create a temporary model config with overrides
            temp_model_config = self.model_config.copy()
            temp_model_config.update(model_overrides)
            
            # Log model overrides
            logger.info(f"Using model overrides: {model_overrides}")
        else:
            temp_model_config = self.model_config
        
        # For direct messages, get thread_ts from request; for mentions, use the provided parameter
        effective_thread_ts = ai_request.thread_ts if is_direct_message else thread_ts
        conversation_key = f"{ai_request.channel_id}:{effective_thread_ts}"
        
        # Initialize progressive message handler
        message_handler = ProgressiveMessageHandler(self.slack_client, ai_request.channel_id, ai_request.message_ts, effective_thread_ts)
        
        try:
            # Check rate limiting
            if not global_limiter.check_rate_limit():
                await message_handler.send_rate_limit_message()
                return
            
            # Parse user mentions in the text
            ai_request.text = self.parse_user_mentions(ai_request.text)
            
            # Initialize conversation manager
            conversation_manager = ConversationManager(self.slack_client, conversation_key, ai_request.sender_name)
            
            # Load conversation history
            history_context = await conversation_manager.load_conversation(
                ai_request.channel_id, 
                effective_thread_ts,
                is_direct_message
            )
            
            # Initialize SlackClient for tool execution AFTER loading conversation history
            # This optimization significantly reduces API rate limit issues
            from slack_ops.slack_tools import SlackClient
            slack_tool = SlackClient(
                bot_token=self.slack_bot_token,
                user_token=self.slack_user_token,
            )
            # Override the slack_tool_client for this request
            self.slack_tool_client = slack_tool
            
            # Send initial thinking message only if message_ts isn't already set (from app.py)
            if not ai_request.message_ts:
                await message_handler.send_thinking_message()
            
            # Extract URLs from text
            urls = self.extract_urls(ai_request.text)
            if urls:
                ai_request.urls = urls
            
            # Execute the three-phase workflow
            # 1. Planning Phase - determine which tools to use
            tools_to_execute = await self._planning_phase(
                ai_request, 
                history_context, 
                message_handler,
                model=temp_model_config.get('planning')
            )
            
            # Check if the LLM decided no tools are needed
            if not tools_to_execute:
                logger.info("No tools needed for this query, generating direct response")
                message_handler.update_stage("Simple query processing", completed=True)
                await message_handler.send_thinking_message(initial=False)
                
                # Skip execution phase and go straight to response generation with empty tool results
                empty_tool_results = {
                    "results": {},
                    "context": f"Current {ai_request.sender_name} query: {ai_request.text}\n"
                }
                final_response = await self._response_generation(
                    empty_tool_results, 
                    ai_request, 
                    history_context, 
                    message_handler,
                    model=temp_model_config.get('response_generation')
                )
            else:
                # 2. Sequential Tool Execution Phase
                tool_results = await self._execution_phase(
                    tools_to_execute, 
                    ai_request, 
                    history_context, 
                    message_handler,
                    model=temp_model_config.get('tool_execution')
                )
                
                # 3. Response Generation Phase
                final_response = await self._response_generation(
                    tool_results, 
                    ai_request, 
                    history_context, 
                    message_handler,
                    model=temp_model_config.get('response_generation')
                )
            
            # Delete the thinking message
            await message_handler.delete_thinking_message()
            
            # Stream the final response to user
            await self._stream_response(final_response, ai_request, effective_thread_ts, ai_request.sender_name)
            
            # Add the current message to history after processing
            await conversation_manager.add_message("user", ai_request.text, ai_request.message_ts)
            
            # Log execution time - keep this as INFO for performance monitoring
            elapsed_time = time.time() - start_time
            logger.info(f"Processed message in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await message_handler.send_error_message(str(e))
    
    async def _planning_phase(self, ai_request: AIRequest, history_context: List[str], 
                          message_handler: ProgressiveMessageHandler, model: Optional[str] = None):
        """Determine which tools to execute based on the query."""
        message_handler.update_stage("Analyzing query")
        
        # Use specified model or fall back to config or default
        model_to_use = model or self.model_config.get("planning") or self.default_model
        
        system_prompt = self.prompts.get("plan_phase", {}).get("system_template", "")
        user_prompt_template = self.prompts.get("plan_phase", {}).get("user_template", "")
        
        user_prompt = user_prompt_template.format(
            text=ai_request.text,
            conversation_history="\n".join(history_context),
            sender_name=ai_request.sender_name
        )
        
        # Determine plan with specified model
        plan_result = await self.openai_client.chat_with_history(
            history=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            model=model_to_use
        )
        
        # Parse plan from JSON response
        try:
            plan = json.loads(plan_result)
            logger.debug(f"Planning phase result: {plan}")
            
            # Set up pending stages based on the plan
            tool_stages = []
            for tool in plan:
                if tool == "search_channel_history":
                    tool_stages.append("Searching Slack history")
                elif tool == "semantic_search_linear":
                    tool_stages.append("Searching Linear semantically")
                elif tool == "filter_linear_issues":
                    tool_stages.append("Filtering Linear issues")
                elif tool == "create_linear_issue":
                    tool_stages.append("Creating Linear issue")
                elif tool.startswith("get_linear_"):
                    tool_stages.append(f"Getting Linear {tool.replace('get_linear_team_', '')}")
            
            if tool_stages:
                message_handler.add_pending_stages(tool_stages)
            else:
                message_handler.add_pending_stages(["Simple query processing"])
                
            message_handler.update_stage("Analyzing query", completed=True)
            await message_handler.send_thinking_message(initial=False)
            
            return plan
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in planning phase result: {plan_result}")
            message_handler.add_pending_stages(["Simple query processing"])
            message_handler.update_stage("Analyzing query", completed=True)
            await message_handler.send_thinking_message(initial=False)
            return []
    
    async def _execution_phase(self, tools_to_execute: List[str], ai_request: AIRequest, history_context: List[str], 
                            message_handler: ProgressiveMessageHandler, model: Optional[str] = None):
        """Execute each tool in sequence, re-planning when context gets too large or plan completes."""
        all_results = {}
        context_parts = [f"Current {ai_request.sender_name} query: {ai_request.text}\n"]
        last_context_length = len(context_parts[0])
        
        # Use specified model or fall back to config or default
        model_to_use = model or self.model_config.get("tool_execution") or self.default_model
        
        # Define mapping from prompts.yaml function names to schema function names
        function_name_mapping = {
            "filter_issues": "filter_linear_issues",
            "create_issue": "create_linear_issue",
            "update_issue": "update_linear_issue",
            "filter_comments": "filter_linear_comments",
            "filter_attachments": "filter_linear_attachments",
            "get_users": "get_linear_team_users",
            "get_projects": "get_linear_team_projects",
            "get_cycles": "get_linear_team_cycles",
            "get_labels": "get_linear_team_labels",
            "get_states": "get_linear_team_states",
            "filter_projects": "filter_linear_projects",
            "filter_cycles": "filter_linear_cycles",
            "create_comment": "create_linear_comment",
            "semantic_search": "semantic_search_linear"
        }
        
        # Main execution loop with re-planning
        plan_iteration = 0
        current_plan = tools_to_execute
        max_iterations = 3  # Prevent infinite loops
        
        while plan_iteration < max_iterations:
            plan_iteration += 1
            logger.debug(f"Execution plan iteration {plan_iteration} with {len(current_plan)} tools")
            
            # Execute current plan
            for i, tool_name in enumerate(current_plan):
                # Map the tool_name to stage name for UI updates
                stage_mapping = {
                    "search_channel_history": "Searching Slack history",
                    "semantic_search_linear": "Searching Linear semantically", 
                    "filter_linear_issues": "Filtering Linear issues",
                    "create_linear_issue": "Creating Linear issue",
                    "get_linear_team_users": "Getting Linear team users",
                    "get_linear_team_projects": "Getting Linear team projects",
                    "get_linear_team_cycles": "Getting Linear team cycles",
                    "get_linear_team_labels": "Getting Linear team labels",
                    "get_linear_team_states": "Getting Linear team states",
                    "filter_linear_projects": "Filtering Linear projects",
                    "filter_linear_cycles": "Filtering Linear cycles"
                }
                
                stage_name = stage_mapping.get(tool_name, tool_name)
                message_handler.update_stage(stage_name)
                await message_handler.send_thinking_message(initial=False)
                
                # Get tool schema
                tool_schema = self.tools_map.get(tool_name)
                if not tool_schema:
                    # Try with mapped name
                    mapped_tool_name = function_name_mapping.get(tool_name, tool_name)
                    tool_schema = self.tools_map.get(mapped_tool_name)
                    if not tool_schema:
                        logger.error(f"Tool schema not found for {tool_name} (tried mapping to {mapped_tool_name})")
                        continue
                
                
                user_prompt_template = self.prompts.get("tool_execution", {}).get("user_template", "")
                
                user_prompt = user_prompt_template.format(
                    text=ai_request.text,
                    conversation_history="\n".join(history_context),
                    sender_name=ai_request.sender_name,
                    previous_results=json.dumps({k: "Result obtained" for k in all_results.keys()}),
                    function_name=tool_name
                )
                
                # Get tool parameters with the specified model
                tool_call_result = await self.openai_client.execute_tool_call(
                    query=user_prompt,
                    tools=[tool_schema],
                    previous_results=all_results,
                    model=model_to_use
                )
                
                if "error" in tool_call_result:
                    logger.error(f"Error executing tool {tool_name}: {tool_call_result['error']}")
                    all_results[tool_name] = {"error": tool_call_result['error']}
                    continue
                    
                # Execute the real tool with the parameters
                tool_result = await self._execute_tool(
                    tool_name, 
                    tool_call_result["parameters"], 
                    ai_request
                )
                all_results[tool_name] = tool_result
                
                # Process tool results for context
                context_update = self._process_tool_result(tool_name, tool_result, ai_request)
                if context_update:
                    context_parts.append(context_update)
                
                # Update progress message
                message_handler.update_stage(stage_name, completed=True)
                await message_handler.send_thinking_message(initial=False)
                
                # Check if context has grown significantly (>500 chars) since last check
                current_context_length = sum(len(part) for part in context_parts)
                context_growth = current_context_length - last_context_length
                
                #when context grows by more than 500 characters, evaluate if we need to continue or change plan
                if context_growth > 500:
                    last_context_length = current_context_length
                    # Evaluate if we need to continue or change plan
                    continue_current_plan = await self._evaluation_phase(
                        ai_request.text,
                        current_plan,
                        current_plan[:i+1],
                        all_results,
                        history_context,
                        ai_request,
                        model=model_to_use
                    )
                    
                    if not continue_current_plan:
                        logger.debug("Evaluation suggested re-planning")
                        # Break out of the for loop to re-plan
                        break
            
            # Check if we completed the full plan (no early breaks)
            if i == len(current_plan) - 1:
                logger.debug("Completed full execution plan")
                break
                
            # Re-plan based on updated context
            if plan_iteration < max_iterations:
                message_handler.update_stage("Re-planning based on new information")
                await message_handler.send_thinking_message(initial=False)
                
                # Add the context we've gathered to the planning prompt
                full_context = "\n".join(context_parts)
                
                # Create a special prompt for re-planning with the results so far
                system_prompt = self.prompts.get("plan_phase", {}).get("system_template", "")
                user_prompt_template = self.prompts.get("plan_phase", {}).get("user_template", "")
                
                user_prompt = user_prompt_template.format(
                    text=ai_request.text,
                    conversation_history="\n".join(history_context) + "\n\n" + full_context,
                    sender_name=ai_request.sender_name
                ) + f"\n\nBased on the information gathered so far:\n{full_context}\n\nWhat additional tools should I use?"
                
                # Get new plan
                new_plan_result = await self.openai_client.chat_with_history(
                    history=[{"role": "user", "content": user_prompt}],
                    system_prompt=system_prompt
                )
                
                try:
                    new_plan = json.loads(new_plan_result)
                    logger.debug(f"Re-planning result (iteration {plan_iteration}): {new_plan}")
                    
                    # Update current plan for next iteration
                    current_plan = new_plan
                    
                    # Update pending stages
                    message_handler.stages_pending = []
                    tool_stages = []
                    for tool in new_plan:
                        stage_name = stage_mapping.get(tool, tool)
                        if stage_name not in message_handler.stages_completed:
                            tool_stages.append(stage_name)
                    
                    if tool_stages:
                        message_handler.add_pending_stages(tool_stages)
                    
                    message_handler.update_stage("Re-planning based on new information", completed=True)
                    await message_handler.send_thinking_message(initial=False)
                    
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in re-planning: {new_plan_result}")
                    # If re-planning fails, exit the loop
                    break
            else:
                logger.debug(f"Reached maximum iterations ({max_iterations}), stopping execution loop")
                break
        
        return {
            "results": all_results,
            "context": "\n".join(context_parts)
        }
    
    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], ai_request: AIRequest):
        """Execute a specific tool with the given parameters using real implementations."""
        try:
            # Map function names if needed (from prompts.yaml schema names to actual implementation names)
            func_name_mapping = {
                "filter_issues": "filter_linear_issues",
                "create_issue": "create_linear_issue",
                "update_issue": "update_linear_issue", 
                "filter_comments": "filter_linear_comments",
                "filter_attachments": "filter_linear_attachments",
                "get_users": "get_linear_team_users",
                "get_projects": "get_linear_team_projects",
                "get_cycles": "get_linear_team_cycles",
                "get_labels": "get_linear_team_labels",
                "get_states": "get_linear_team_states",
                "filter_projects": "filter_linear_projects",
                "filter_cycles": "filter_linear_cycles",
                "create_comment": "create_linear_comment",
                "semantic_search": "semantic_search_linear"
            }
            
            mapped_tool_name = func_name_mapping.get(tool_name, tool_name)
            
            # Handle Slack tools
            if mapped_tool_name == "search_channel_history":
                # Check required parameters
                if "channel_id" not in parameters:
                    parameters["channel_id"] = ai_request.channel_id
                
                # Execute Slack channel search
                return await self.slack_tool_client.search_channel_history(
                    channel_id=parameters["channel_id"],
                    username=parameters.get("username"),
                    time_range=parameters.get("time_range", "days"),
                    time_value=parameters.get("time_value", 7),
                    message_count=parameters.get("message_count", 50)
                )
            
            # Handle Linear semantic search
            elif mapped_tool_name == "semantic_search_linear":
                # Use the semantic_search function from linear_rag_embeddings
                from linear_db.linear_rag_embeddings import semantic_search
                
                # Execute semantic search
                return semantic_search(
                    query=parameters.get("query", ai_request.text),
                    limit=parameters.get("limit", 10),
                    use_reranker=parameters.get("use_reranker", True),
                    candidate_pool_size=parameters.get("candidate_pool_size", 30),
                    team_key=parameters.get("team_key"),
                    object_type=parameters.get("object_type")
                )
            
            # Handle Linear tools using LinearClient methods
            elif mapped_tool_name == "filter_linear_issues":
                # Build filter criteria for GraphQL
                filter_criteria = {}
                
                # Extract team_key
                if "team_key" in parameters:
                    filter_criteria["team"] = {"key": {"eq": parameters["team_key"]}}
                
                # Extract state
                if "state" in parameters:
                    filter_criteria["state"] = {"name": {"eq": parameters["state"]}}
                
                # Extract priority
                if "priority" in parameters:
                    filter_criteria["priority"] = {"eq": parameters["priority"]}
                
                # Extract assignee filters
                if "assignee_name" in parameters:
                    filter_criteria["assignee"] = {"displayName": {"eq": parameters["assignee_name"]}}
                elif "assignee_contains" in parameters:
                    filter_criteria["assignee"] = {"displayName": {"containsIgnoreCase": parameters["assignee_contains"]}}
                
                # Extract text filters
                if "title_contains" in parameters:
                    filter_criteria["title"] = {"contains": parameters["title_contains"]}
                if "description_contains" in parameters:
                    filter_criteria["description"] = {"contains": parameters["description_contains"]}
                
                # Extract cycle filter
                if "cycle_name" in parameters:
                    filter_criteria["cycle"] = {"name": {"eq": parameters["cycle_name"]}}
                
                # Extract project filter
                if "project_name" in parameters:
                    filter_criteria["project"] = {"name": {"eq": parameters["project_name"]}}
                
                # Extract label filter
                if "label_name" in parameters:
                    filter_criteria["labels"] = {"some": {"name": {"eq": parameters["label_name"]}}}
                
                # Get limit parameter
                limit = parameters.get("first", 10)
                
                # Use the filterIssues method from LinearClient
                return self.linear_client.filterIssues(filter_criteria, limit=limit)
            
            elif mapped_tool_name == "create_linear_issue":
                # Prepare the input data for createIssue
                issue_data = {}
                
                # Get team ID from team key
                if "team_key" in parameters:
                    team_key = parameters["team_key"]
                    # Add team ID logic here if needed
                    # For now, we'll assume createIssue can handle team_key
                    issue_data["teamId"] = team_key
                
                # Map parameters to issue_data
                if "title" in parameters:
                    issue_data["title"] = parameters["title"]
                if "description" in parameters:
                    issue_data["description"] = parameters["description"]
                if "priority" in parameters:
                    issue_data["priority"] = parameters["priority"]
                if "estimate" in parameters:
                    issue_data["estimate"] = parameters["estimate"]
                if "assignee_name" in parameters:
                    # This may need additional logic to get assigneeId
                    issue_data["assigneeName"] = parameters["assignee_name"]
                if "state_name" in parameters:
                    # This may need additional logic to get stateId
                    issue_data["stateName"] = parameters["state_name"]
                if "label_names" in parameters:
                    issue_data["labelNames"] = parameters["label_names"]
                if "project_name" in parameters:
                    issue_data["projectName"] = parameters["project_name"]
                if "cycle_name" in parameters:
                    issue_data["cycleName"] = parameters["cycle_name"]
                
                # Use the createIssue method from LinearClient
                return self.linear_client.createIssue(issue_data)
            
            elif mapped_tool_name == "update_linear_issue":
                # Prepare input data for updateIssue
                issue_data = {}
                
                # Map parameters to issue_data
                if "title" in parameters:
                    issue_data["title"] = parameters["title"]
                if "description" in parameters:
                    issue_data["description"] = parameters["description"]
                if "priority" in parameters:
                    issue_data["priority"] = parameters["priority"]
                if "estimate" in parameters:
                    issue_data["estimate"] = parameters["estimate"]
                if "assignee_name" in parameters:
                    issue_data["assigneeName"] = parameters["assignee_name"]
                if "state_name" in parameters:
                    issue_data["stateName"] = parameters["state_name"]
                if "label_names" in parameters:
                    issue_data["labelNames"] = parameters["label_names"]
                if "project_name" in parameters:
                    issue_data["projectName"] = parameters["project_name"]
                if "cycle_name" in parameters:
                    issue_data["cycleName"] = parameters["cycle_name"]
                if "archived" in parameters:
                    issue_data["archived"] = parameters["archived"]
                
                # Get the issue number
                issue_number = parameters.get("issue_number")
                if not issue_number:
                    return {"error": "Issue number is required for updating an issue"}
                
                # Use the updateIssue method from LinearClient
                return self.linear_client.updateIssue(issue_number, issue_data)
            
            elif mapped_tool_name == "get_linear_team_users":
                # Use the getAllUsers method from LinearClient
                team_key = parameters.get("team_key")
                if not team_key:
                    return {"error": "Team key is required"}
                    
                users = self.linear_client.getAllUsers(team_key)
                return {"items": users}
            
            elif mapped_tool_name == "get_linear_team_projects":
                # Use the getAllProjects method from LinearClient
                team_key = parameters.get("team_key")
                if not team_key:
                    return {"error": "Team key is required"}
                    
                projects = self.linear_client.getAllProjects(team_key)
                return {"items": projects}
            
            elif mapped_tool_name == "get_linear_team_cycles":
                # Use the getAllCycles method from LinearClient
                team_key = parameters.get("team_key")
                if not team_key:
                    return {"error": "Team key is required"}
                    
                cycles = self.linear_client.getAllCycles(team_key)
                return {"items": cycles}
            
            elif mapped_tool_name == "get_linear_team_labels":
                # Use the getAllLabels method from LinearClient
                team_key = parameters.get("team_key")
                if not team_key:
                    return {"error": "Team key is required"}
                    
                labels = self.linear_client.getAllLabels(team_key)
                return {"items": labels}
            
            elif mapped_tool_name == "get_linear_team_states":
                # Use the getAllStates method from LinearClient
                team_key = parameters.get("team_key")
                if not team_key:
                    return {"error": "Team key is required"}
                    
                states = self.linear_client.getAllStates(team_key)
                return {"items": states}
            
            elif mapped_tool_name == "filter_linear_projects":
                # Build filter criteria
                filter_criteria = {}
                
                # Extract team_key
                if "team_key" in parameters:
                    filter_criteria["team"] = {"key": {"eq": parameters["team_key"]}}
                
                # Extract name filters
                if "name" in parameters:
                    filter_criteria["name"] = {"eq": parameters["name"]}
                elif "name_contains" in parameters:
                    filter_criteria["name"] = {"contains": parameters["name_contains"]}
                elif "name_contains_ignore_case" in parameters:
                    filter_criteria["name"] = {"containsIgnoreCase": parameters["name_contains_ignore_case"]}
                
                # Extract state filter
                if "state" in parameters:
                    filter_criteria["state"] = {"eq": parameters["state"]}
                
                # Extract lead filter
                if "lead_display_name" in parameters:
                    filter_criteria["lead"] = {"displayName": {"eq": parameters["lead_display_name"]}}
                
                # Use the filterProjects method from LinearClient
                projects = self.linear_client.filterProjects(filter_criteria)
                return {"results": projects}
            
            elif mapped_tool_name == "filter_linear_cycles":
                # Build filter criteria
                filter_criteria = {}
                
                # Extract team_key
                if "team_key" in parameters:
                    filter_criteria["team"] = {"key": {"eq": parameters["team_key"]}}
                
                # Extract name filters
                if "name" in parameters:
                    filter_criteria["name"] = {"eq": parameters["name"]}
                elif "name_contains" in parameters:
                    filter_criteria["name"] = {"contains": parameters["name_contains"]}
                
                # Extract is_active filter
                if "is_active" in parameters:
                    filter_criteria["isActive"] = {"eq": parameters["is_active"]}
                
                # Use the filterCycles method from LinearClient
                cycles = self.linear_client.filterCycles(filter_criteria)
                return {"results": cycles}
            
            elif mapped_tool_name == "filter_linear_comments":
                # Build filter criteria
                filter_criteria = {}
                
                # Extract issue_number
                if "issue_number" in parameters:
                    filter_criteria["issue"] = {"number": {"eq": parameters["issue_number"]}}
                
                # Extract user filter
                if "user_display_name" in parameters:
                    filter_criteria["user"] = {"displayName": {"eq": parameters["user_display_name"]}}
                
                # Extract content filters
                if "contains" in parameters:
                    filter_criteria["body"] = {"contains": parameters["contains"]}
                elif "contains_ignore_case" in parameters:
                    filter_criteria["body"] = {"containsIgnoreCase": parameters["contains_ignore_case"]}
                
                # Use the filterComments method from LinearClient
                comments = self.linear_client.filterComments(filter_criteria)
                return {"results": comments}
            
            elif mapped_tool_name == "filter_linear_attachments":
                # Build filter criteria
                filter_criteria = {}
                
                # Extract issue_number
                if "issue_number" in parameters:
                    filter_criteria["issue"] = {"number": {"eq": parameters["issue_number"]}}
                
                # Extract title filter
                if "title_contains" in parameters:
                    filter_criteria["title"] = {"contains": parameters["title_contains"]}
                
                # Extract creator filter
                if "creator_display_name" in parameters:
                    filter_criteria["creator"] = {"displayName": {"eq": parameters["creator_display_name"]}}
                
                # Use the filterAttachments method from LinearClient
                attachments = self.linear_client.filterAttachments(filter_criteria)
                return {"results": attachments}
                
            elif mapped_tool_name == "create_linear_comment":
                # Extract parameters
                issue_number = parameters.get("issue_number")
                body = parameters.get("body")
                
                if not issue_number:
                    return {"error": "Issue number is required for creating a comment"}
                if not body:
                    return {"error": "Comment body is required"}
                
                # Prepare the comment data
                comment_data = {"body": body}
                
                # Use the createComment method from LinearClient
                return self.linear_client.createComment(issue_number, comment_data)
            
            elif mapped_tool_name == "get_users":
                # Get list of employees with their data
                return self.slack_tool_client.get_employees_data(
                    real_name=parameters.get("real_name", "")
                )
            
            else:
                # Default case for unknown tools
                return {"error": f"Unknown tool: {mapped_tool_name}"}
                
        except Exception as e:
            logger.error(f"Error executing tool {mapped_tool_name}: {str(e)}")
            return {"error": f"Tool execution error: {str(e)}"}
    
    def _process_tool_result(self, tool_name: str, result: Dict[str, Any], ai_request: AIRequest) -> str:
        """Process tool results and format them for context."""
        try:
            # Map function names if needed (from prompts.yaml schema names to actual implementation names)
            func_name_mapping = {
                "filter_issues": "filter_linear_issues",
                "create_issue": "create_linear_issue",
                "update_issue": "update_linear_issue", 
                "filter_comments": "filter_linear_comments",
                "filter_attachments": "filter_linear_attachments",
                "get_users": "get_linear_team_users",
                "get_projects": "get_linear_team_projects",
                "get_cycles": "get_linear_team_cycles",
                "get_labels": "get_linear_team_labels",
                "get_states": "get_linear_team_states",
                "filter_projects": "filter_linear_projects",
                "filter_cycles": "filter_linear_cycles",
                "create_comment": "create_linear_comment",
                "semantic_search": "semantic_search_linear"
            }
            
            mapped_tool_name = func_name_mapping.get(tool_name, tool_name)
            
            # Format Slack channel history
            if mapped_tool_name == "search_channel_history":
                # Check if it's an error
                if "error" in result:
                    return f"\nError searching Slack history: {result['error']}"
                    
                # Format search results
                messages_count = result.get("count", 0)
                formatted_result = f"\nFound {messages_count} relevant Slack messages."
                
                # Add search parameters
                search_params = result.get("search_params", {})
                if search_params:
                    if search_params.get("username"):
                        formatted_result += f"\nFiltered by user: @{search_params['username']}"
                    
                    time_range = f"{search_params.get('time_value', 7)} {search_params.get('time_range', 'days')}"
                    formatted_result += f"\nTime range: {time_range}"
                
                # Add message summaries
                if messages_count > 0:
                    formatted_result += "\n\nMessage summaries:"
                    messages = result.get("messages", [])
                    # Limit to at most 10 message summaries
                    for i, msg in enumerate(messages[:10]):
                        user_id = msg.get("user", "Unknown")
                        text = msg.get("text", "")
                        if len(text) > 200:  # Truncate long messages
                            text = text[:197] + "..."
                        formatted_result += f"\n- Message {i+1} from <@{user_id}>: {text}"
                    
                    # Note if there are more messages than shown
                    if len(messages) > 10:
                        formatted_result += f"\n... and {len(messages) - 10} more messages not shown."
                
                return formatted_result
            
            # Format Linear semantic search
            elif mapped_tool_name == "semantic_search_linear":
                # Check if it's an error
                if "error" in result:
                    return f"\nError searching Linear: {result['error']}"
                    
                # Format search results
                results_count = len(result.get("results", []))
                formatted_result = f"\nFound {results_count} relevant Linear items."
                
                # Add issue summaries
                if results_count > 0:
                    formatted_result += "\n\nIssue summaries:"
                    issues = result.get("results", [])
                    # Limit to at most 5 issue summaries
                    for i, issue in enumerate(issues[:5]):
                        issue_id = issue.get("id", "Unknown")
                        title = issue.get("title", "No title")
                        state = issue.get("state", "No state")
                        priority = issue.get("priority", "No priority")
                        formatted_result += f"\n- Issue {i+1}: {title} (ID: {issue_id}, State: {state}, Priority: {priority})"
                    
                    # Note if there are more issues than shown
                    if len(issues) > 5:
                        formatted_result += f"\n... and {len(issues) - 5} more issues not shown."
                
                return formatted_result
                
            # Format Linear issue filtering
            elif mapped_tool_name == "filter_linear_issues":
                # Check if it's an error
                if "error" in result:
                    return f"\nError filtering Linear issues: {result['error']}"
                    
                # Format filter results
                results_count = len(result.get("results", []))
                formatted_result = f"\nFound {results_count} matching Linear issues."
                
                # Add issue summaries
                if results_count > 0:
                    formatted_result += "\n\nIssue summaries:"
                    issues = result.get("results", [])
                    # Limit to at most 5 issue summaries
                    for i, issue in enumerate(issues[:5]):
                        issue_id = issue.get("id", "Unknown")
                        title = issue.get("title", "No title")
                        state = issue.get("state", {}).get("name", "No state")
                        assignee = issue.get("assignee", {}).get("name", "Unassigned")
                        formatted_result += f"\n- Issue {i+1}: {title} (ID: {issue_id}, State: {state}, Assignee: {assignee})"
                    
                    # Note if there are more issues than shown
                    if len(issues) > 5:
                        formatted_result += f"\n... and {len(issues) - 5} more issues not shown."
                
                return formatted_result
            
            # Format Linear issue creation
            elif mapped_tool_name == "create_linear_issue":
                # Check if it's an error
                if "error" in result:
                    return f"\nError creating Linear issue: {result['error']}"
                
                # Format success response
                issue = result.get("issue", {})
                if issue:
                    formatted_result = f"\nSuccessfully created Linear issue:"
                    formatted_result += f"\nTitle: {issue.get('title', 'No title')}"
                    formatted_result += f"\nURL: {issue.get('url', 'No URL')}"
                    
                    if issue.get('assignee'):
                        formatted_result += f"\nAssigned to: {issue['assignee']}"
                    if issue.get('team'):
                        formatted_result += f"\nTeam: {issue['team']}"
                    if issue.get('priority'):
                        formatted_result += f"\nPriority: {issue['priority']}"
                    
                    return formatted_result
                else:
                    return f"\nIssue created, but details not available."
            
            # Format Linear team data (users, projects, cycles, etc.)
            elif mapped_tool_name.startswith("get_linear_team_"):
                item_type = mapped_tool_name.replace("get_linear_team_", "")
                
                # Check if it's an error
                if "error" in result:
                    return f"\nError getting Linear {item_type}: {result['error']}"
                
                # Format items
                items = result.get("items", [])
                items_count = len(items)
                formatted_result = f"\nFound {items_count} {item_type} in team."
                
                if items_count > 0:
                    formatted_result += f"\n\n{item_type.capitalize()} summary:"
                    for i, item in enumerate(items[:7]):  # Limit to 7 items
                        name = item.get("name", "No name")
                        if item_type == "users":
                            formatted_result += f"\n- User: {name} (Email: {item.get('email', 'No email')})"
                        elif item_type == "projects": 
                            formatted_result += f"\n- Project: {name} (State: {item.get('state', 'No state')})"
                        elif item_type == "cycles":
                            formatted_result += f"\n- Cycle: {name} (Status: {item.get('status', 'No status')})"
                        elif item_type == "labels":
                            formatted_result += f"\n- Label: {name} (Color: {item.get('color', 'No color')})"
                        elif item_type == "states":
                            formatted_result += f"\n- State: {name} (Type: {item.get('type', 'No type')})"
                    
                    # Note if there are more items than shown
                    if items_count > 7:
                        formatted_result += f"\n... and {items_count - 7} more {item_type} not shown."
                
                return formatted_result
            
            # Format default case
            else:
                # For simplicity, convert to JSON string with indentation for readability
                return f"\nResults from {mapped_tool_name}:\n{json.dumps(result, indent=2, default=str)}"
                
        except Exception as e:
            logger.error(f"Error processing tool results for {mapped_tool_name}: {str(e)}")
            return f"\nError formatting results for {mapped_tool_name}: {str(e)}"
    
    async def _evaluation_phase(self, query: str, plan: List[str], executed_tools: List[str], 
                          results: Dict, history_context: List[str], ai_request: AIRequest, model: Optional[str] = None):
        """Evaluate whether to continue with the current plan or change strategy."""
        logger.debug("Starting evaluation phase")
        
        # Use specified model or fall back to config or default
        model_to_use = model or self.model_config.get("evaluation") or self.default_model
        
        system_prompt = self.prompts.get("evaluation_phase", {}).get("system_template", "")
        user_prompt_template = self.prompts.get("evaluation_phase", {}).get("user_template", "")
        
        user_prompt = user_prompt_template.format(
            text=query,
            current_plan=json.dumps(plan),
            executed_functions=json.dumps(executed_tools),
            obtained_results=json.dumps({k: "Result obtained" for k in results.keys()}),
            current_step=len(executed_tools),
            total_steps=len(plan)
        )
        
        # Get evaluation with specified model
        evaluation_result = await self.openai_client.chat_with_history(
            history=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            model=model_to_use
        )
        
        # Parse evaluation
        try:
            evaluation = json.loads(evaluation_result)
            logger.debug(f"Evaluation phase result: {evaluation}")
            
            return evaluation.get("continue_current_plan", True)
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in evaluation phase result: {evaluation_result}")
            return True
    
    async def _response_generation(self, tool_results: Dict[str, Any], ai_request: AIRequest, history_context: List[str], 
                               message_handler: ProgressiveMessageHandler, model: Optional[str] = None):
        """Generate final response using results from executed tools."""
        message_handler.update_stage("Generating response")
        await message_handler.send_thinking_message(initial=False)
        
        # Use specified model or fall back to config or default
        model_to_use = model or self.model_config.get("response_generation") or self.default_model
        
        system_prompt = self.prompts.get("final_response", {}).get("system_template", "")
        user_prompt_template = self.prompts.get("final_response", {}).get("user_template", "")
        
        user_prompt = user_prompt_template.format(
            text=ai_request.text,
            conversation_history="\n".join(history_context),
            sender_name=ai_request.sender_name,
            function_results=json.dumps(tool_results["results"]),
            employee_data=""  # Add employee data if needed
        )
        
        # Generate response with specified model
        final_response = await self.openai_client.chat_with_history(
            history=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            model=model_to_use
        )
        
        return {
            "response": final_response,
            "context": tool_results["context"],
            "history_context": "\n".join(history_context),
            "sender_name": ai_request.sender_name
        }
    
    async def _stream_response(self, final_response: Dict[str, Any], ai_request: AIRequest, thread_ts: str, sender_name: str):
        """Stream the response back to Slack."""
        # This would implement the streaming functionality
        # For now, just send a simple message
        try:
            await asyncio.to_thread(
                self.slack_client.chat_postMessage,
                channel=ai_request.channel_id,
                thread_ts=thread_ts,
                text=final_response["response"]
            )
            
            # Add response to conversation history
            conversation_key = f"{ai_request.channel_id}:{thread_ts}"
            conversation_manager = ConversationManager(self.slack_client, conversation_key, sender_name)
            await conversation_manager.add_message("assistant", final_response["response"])
            
        except SlackApiError as e:
            logger.error(f"Error sending response: {e.response.get('error', '')}") 