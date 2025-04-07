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
import uuid
from typing import Dict, List, Any, Optional, Union, Callable

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from llm.openai_client import OpenaiClient
from ops_conversation_db.conversation_db import get_db, load_conversation_from_db, save_conversation_to_db
from tools.tool_schema_linear import LINEAR_SCHEMAS
from tools.tool_schema_slack import SLACK_SCHEMAS  
from tools.tool_schema_semantic_search import SEMANTIC_SEARCH_SCHEMAS
from rate_limiter import global_limiter, slack_limiter, linear_limiter, openai_limiter
from ops_linear_db.linear_client import LinearClient
from utils import safe_append, format_for_slack
from agent import Commander, Captain, Soldier
from context_manager import context_manager

# Tool configuration
TOOLS_CONFIG = {
    "linear": {
        "enabled": os.getenv("LINEAR_API_KEY") is not None,
        "semantic_search": True
    },
    "slack": {
        "enabled": True
    },
    "github": {
        "enabled": os.getenv("GITHUB_TOKEN") is not None
    },
    "url": {
        "enabled": True
    }
}

# Configure logger
logger = logging.getLogger("tmai_agent")

# Only log errors and critical issues in production
if os.environ.get("ENVIRONMENT") == "production":
    logger.setLevel(logging.INFO)
elif os.environ.get("ENVIRONMENT") == "development":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)

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
        self.stop_requested = False
        self.interaction_ts = None  # Timestamp of the most recent interaction message
    
    async def send_thinking_message(self, initial=True):
        """Send or update the thinking message."""
        blocks = self._build_thinking_blocks()
        
        try:
            if initial:
                response = await asyncio.to_thread(
                    self.slack_client.chat_postMessage,
                    channel=self.channel_id,
                    text="TMAI processing your request...",
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
                    text="TMAI processing your request...",
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
            status_text += f"➤ {self.current_stage}\n"
        
        for stage in self.stages_pending:
            if stage != self.current_stage:
                status_text += f"  {stage}\n"
                
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

    async def send_thinking_message_with_stop(self, initial=True):
        """Send or update the thinking message with a stop button."""
        blocks = self._build_thinking_blocks_with_stop()
        
        try:
            if initial:
                response = await asyncio.to_thread(
                    self.slack_client.chat_postMessage,
                    channel=self.channel_id,
                    text="TMAI processing your request...",
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
                    text="TMAI processing your request...",
                    blocks=blocks
                )
        except SlackApiError as e:
            logger.warning(f"Error updating message: {e.response.get('error', '')}")

    def _build_thinking_blocks_with_stop(self):
        """Build the blocks for the thinking message with a stop button."""
        # Get the basic thinking blocks
        blocks = self._build_thinking_blocks()
        
        # Add a stop button
        stop_button_block = {
            "type": "actions",
            "block_id": "stop_execution",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Stop Processing",
                        "emoji": True
                    },
                    "style": "danger",
                    "value": f"stop_{uuid.uuid4()}",  # Unique identifier
                    "action_id": "stop_processing"
                }
            ]
        }
        
        blocks.append(stop_button_block)
        return blocks

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
            logger.debug(f"Fetching conversation history from Slack API for {channel_id}:{thread_ts}")
            response = self.slack_client.conversations_replies(
                channel=channel_id,
                ts=thread_ts
            )
            
            if response.get("ok") and response.get("messages"):
                messages = response.get("messages", [])
                logger.debug(f"Received {len(messages)} raw messages from Slack API")
                
                # Different handling for direct messages vs mentions
                if is_direct_message:
                    # For DMs, reverse messages for chronological order
                    messages.reverse()
                else:
                    # For mentions, skip the first message
                    messages = messages[1:]
                
                self.conversation_history[self.conversation_key] = []
                seen_ts = set()  # Track seen message timestamps to avoid duplicates
                
                for msg in messages:
                    # Get the timestamp to identify this message
                    msg_ts = msg.get("ts")
                    
                    # Skip if we've already seen this message
                    if msg_ts in seen_ts:
                        logger.debug(f"Skipping duplicate message with ts: {msg_ts}")
                        continue
                    
                    seen_ts.add(msg_ts)
                    
                    # Determine if it's a user or bot message
                    is_bot = msg.get("bot_id") is not None
                    text = msg.get("text", "")
                    
                    # Skip empty messages
                    if not text:
                        logger.debug(f"Skipping empty message with ts: {msg_ts}")
                        continue
                        
                    # Skip processing messages
                    if is_bot and ("TMAI processing your request..." in text or "TMAI's neuron firing" in text or "is thinking" in text):
                        logger.debug(f"Skipping processing message: {text[:30]}...")
                        continue
                    
                    # Skip default thread signal (only in DMs)
                    if is_direct_message and "New Assistant Thread" in text:
                        logger.debug("Skipping 'New Assistant Thread' message")
                        continue
                    
                    # Add to conversation history
                    role = "assistant" if is_bot else "user"
                    logger.debug(f"Adding message: role={role}, content={text[:30]}...")
                    self._add_message_internal(
                        role,
                        text,
                        msg_ts
                    )
                
                self.history = self.conversation_history[self.conversation_key]
                logger.debug(f"Rebuilt conversation history with {len(self.history)} messages")
                
                # Log the formatted history for debugging
                formatted_history = self._format_history_for_context(self.sender_name)
                logger.debug(f"Formatted history has {len(formatted_history)} lines")
                
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
    
    def _format_history_for_context(self, sender_name: str = "User", max_messages: int = 20):
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
        
        # Process messages in chronological order
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                # Format user messages with sender name
                msg_content = msg['content']
                history_context.append(f"**{sender_name}:** {msg_content}")
            else:
                # Format assistant messages
                content = msg["content"]
                if len(content) > 500:
                    content = content[:500] + "... (content truncated)"
                history_context.append(f"**Assistant:** {content}")
        
        return history_context

class TMAISlackAgent:
    """
    Main agent class for the TMAI Slack Assistant.
    Implements the agent workflow for processing messages.
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
        self.openai_client = OpenaiClient(api_key=openai_api_key, model=ai_model)
        
        # Store default model
        self.default_model = ai_model
        
        # Store model configuration for different phases
        self.model_config = model_config or {
            "commander": ai_model,
            "captain": ai_model,
            "soldier": ai_model
        }
        
        # Store prompts
        self.prompts = prompts or {}
        if logger.isEnabledFor(logging.DEBUG):
            if prompts:
                prompt_categories = list(prompts.keys())
                logger.debug(f"TMAISlackAgent loaded with prompts: {prompt_categories}")
                for category in prompt_categories:
                    if isinstance(prompts[category], dict):
                        logger.debug(f"  • {category} has {len(prompts[category])} actions")
            else:
                logger.debug("TMAISlackAgent initialized with empty prompts dictionary")
        
        # Initialize LinearClient if needed
        if TOOLS_CONFIG.get("linear", {}).get("enabled", False):
            LINEAR_API_KEY = os.getenv("LINEAR_API_KEY")
            from ops_linear_db.linear_client import LinearClient
            self.linear_client = LinearClient(LINEAR_API_KEY)
            
            # Import semantic search if needed
            if TOOLS_CONFIG.get("linear", {}).get("semantic_search", False):
                from ops_linear_db.linear_rag_embeddings import semantic_search
                self.semantic_search = semantic_search
            else:
                self.semantic_search = None
        
        # Initialize SlackClient for tool execution
        # Import SlackClient
        try:
            from ops_slack.slack_tools import SlackClient
            slack_client = SlackClient(slack_bot_token, slack_user_token)
            logger.info("SlackClient initialized successfully")
        except Exception as e:
            try:
                from ops_slack.slack_tools import SlackClient
                slack_client = SlackClient(slack_bot_token)
                logger.info("SlackClient initialized with bot token only")
            except Exception as e:
                slack_client = None
                logger.error(f"Failed to import SlackClient: {str(e)}")
                
        from ops_slack.slack_tools import SlackClient
        self.slack_tool_client = SlackClient(slack_bot_token, slack_user_token)
        
        # Initialize agent components
        self.commander = Commander(model=self.model_config.get("commander", ai_model), prompts=self.prompts)
        self.captain = Captain(model=self.model_config.get("captain", ai_model), prompts=self.prompts)
        self.soldier = Soldier(model=self.model_config.get("soldier", ai_model), prompts=self.prompts)
        
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
        Process messages using the Commander-Captain-Soldier agent workflow.
        
        Args:
            ai_request: The AI request object containing user, channel, text
            thread_ts: Optional thread timestamp (required for mentions)
            is_direct_message: Whether this is a direct message
            model_overrides: Optional dict to override models for specific phases
        """
        start_time = time.time()
        logger.info(f"Processing message from user {ai_request.user_id}")
        
        # Apply any model overrides
        model_overrides = model_overrides or {}
        if model_overrides:
            # Create a temporary model config with overrides
            temp_model_config = self.model_config.copy()
            temp_model_config.update(model_overrides)
            logger.info(f"Using model overrides: {model_overrides}")
        else:
            temp_model_config = self.model_config
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using default model config: {self.model_config}")
        
        # Update agent models if overrides provided
        if "commander" in temp_model_config:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Creating Commander with model: {temp_model_config['commander']}")
            self.commander = Commander(model=temp_model_config["commander"], prompts=self.prompts)
        if "captain" in temp_model_config:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Creating Captain with model: {temp_model_config['captain']}")
            self.captain = Captain(model=temp_model_config["captain"], prompts=self.prompts)
        if "soldier" in temp_model_config:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Creating Soldier with model: {temp_model_config['soldier']}")
            self.soldier = Soldier(model=temp_model_config["soldier"], prompts=self.prompts)
        
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
            
            # Create a context for this execution
            context = context_manager.create_context(
                ai_request.channel_id,
                effective_thread_ts,
                ai_request.user_id,
                ai_request.text
            )
            # Initialize stop_requested flag in context
            context_manager.add_context(context['context_id'], 'stop_requested', False)
            
            # Send initial thinking message only if message_ts isn't already set (from app.py)
            if not ai_request.message_ts:
                await message_handler.send_thinking_message()
            
            # Extract URLs from text
            urls = self.extract_urls(ai_request.text)
            if urls:
                ai_request.urls = urls
            
            # Initialize SlackClient for tool execution
            from ops_slack.slack_tools import SlackClient
            slack_tool = SlackClient(
                bot_token=self.slack_bot_token,
                user_token=self.slack_user_token,
            )
            # Override the slack_tool_client for this request
            self.slack_tool_client = slack_tool
            
            # Convert conversation history to format needed for Commander
            history_for_commander = history_context
            
            # If history_for_commander is a list, convert it to a string
            if isinstance(history_for_commander, list):
                history_for_commander = "\n".join(history_for_commander)
            
            # 1. Commander: Assign tasks between platforms
            message_handler.update_stage("Commander assigning tasks")
            await message_handler.send_thinking_message(initial=False)
            
            commander_result = self.commander.assign_tasks(
                user_query= "Current prompt from: " + ai_request.sender_name + ":" + ai_request.text,
                history=history_for_commander
            )
            
            logger.info(f"Commander assigned platforms: {commander_result.get('platform', [])}")
            context_manager.add_context(context['context_id'], 'commander_result', commander_result)
            
            # Extract platforms and order
            platforms = commander_result.get('platform', [])
            # Ensure platforms is always a list (Commander can return a string for a single platform)
            if isinstance(platforms, str):
                platforms = [platforms]
                logger.debug(f"Converted platform string '{platforms[0]}' to list")
            order = commander_result.get('order', ai_request.text)
            
            message_handler.update_stage("Commander assigning tasks", completed=True)
            
            # If no platforms, send direct response
            if not platforms or platforms == ["direct_response"]:
                message_handler.update_stage("Direct response", completed=True)
                await message_handler.send_thinking_message(initial=False)
                
                # Generate direct response
                final_response = self.commander.response(order, {})
                
                # Send final message and delete thinking message
                await self._send_response(final_response, ai_request, effective_thread_ts)
                await message_handler.delete_thinking_message()

                # Add to conversation history
                await conversation_manager.add_message("user", ai_request.text, ai_request.message_ts)
                await conversation_manager.add_message("assistant", final_response)
                
                return
            
            # Main execution loop
            execution_continues = True
            max_iterations = 3  # Prevent infinite loops
            iteration = 0
            
            # Replace regular thinking message with one that has a stop button
            await message_handler.send_thinking_message_with_stop(initial=False)
            
            while execution_continues and iteration < max_iterations:
                iteration += 1
                
                # Check if a stop was requested
                if context_manager.get_context(context['context_id']).get('stop_requested', False):
                    logger.info(f"Stop requested for context {context['context_id']}, breaking execution loop")
                    break
                
                # 2. Captain: Plan execution
                message_handler.update_stage("Captain planning functions")
                await message_handler.send_thinking_message_with_stop(initial=False)
                
                plan = self.captain.plan(order, platforms)
                logger.info(f"Captain planned {len(plan.get('functions', {}).get('ready_to_execute', []))} functions")
                
                # Store plan in context
                context_manager.add_context(context['context_id'], 'current_plan', plan)
                
                message_handler.update_stage("Captain planning functions", completed=True)
                
                # Check if a stop was requested before continuing
                if context_manager.get_context(context['context_id']).get('stop_requested', False):
                    logger.info(f"Stop requested after planning for context {context['context_id']}")
                    break
                
                # Get functions ready to execute
                ready_functions = plan.get('functions', {}).get('ready_to_execute', [])
                
                # If no functions ready, send direct response
                if not ready_functions:
                    message_handler.update_stage("No functions to execute", completed=True)
                    await message_handler.send_thinking_message_with_stop(initial=False)
                    
                    # Generate response from Commander
                    final_response = self.commander.response(order, {})
                    
                    # Send final message and delete thinking message
                    await self._send_response(final_response, ai_request, effective_thread_ts)
                    await message_handler.delete_thinking_message()
                    
                    # Add to conversation history
                    await conversation_manager.add_message("user", ai_request.text, ai_request.message_ts)
                    await conversation_manager.add_message("assistant", final_response)
                    
                    return
                
                # 3. Execute each function
                execution_results = {}
                for function_name in ready_functions:
                    # Check if a stop was requested before each function execution
                    if context_manager.get_context(context['context_id']).get('stop_requested', False):
                        logger.info(f"Stop requested before executing {function_name} for context {context['context_id']}")
                        break
                    
                    message_handler.update_stage(f"Executing {function_name}")
                    await message_handler.send_thinking_message_with_stop(initial=False)
                    
                    # Use previous results as context for subsequent functions
                    result = await self.soldier.execute(
                        plan=plan,
                        function_name=function_name,
                        user_query=order,
                        previous_results=execution_results 
                    )
                    
                    # Add result to execution results
                    execution_results[function_name] = result
                    context_manager.add_context(context['context_id'], 'execution_results', execution_results)
                    
                    message_handler.update_stage(f"Executing {function_name}", completed=True)
                    await message_handler.send_thinking_message_with_stop(initial=False)
                
                # Check if a stop was requested after function execution
                if context_manager.get_context(context['context_id']).get('stop_requested', False):
                    logger.info(f"Stop requested after function execution for context {context['context_id']}")
                    break
                
                # 4. Captain evaluation
                message_handler.update_stage("Captain evaluating results")
                await message_handler.send_thinking_message_with_stop(initial=False)
                
                evaluation = self.captain.evaluate(order, plan, execution_results)
                logger.info(f"Captain evaluation: change_plan={evaluation.get('change_plan', False)}, execution_complete={evaluation.get('execution_complete', False)}, response_ready={evaluation.get('response_ready', False)}")
                
                message_handler.update_stage("Captain evaluating results", completed=True)
                
                # Determine if we should continue execution
                if evaluation.get('response_ready', False):
                    # We have enough information to respond
                    execution_continues = False
                elif evaluation.get('change_plan', False):
                    # Need to completely restart planning
                    # Reset the plan but maintain execution results
                    context_manager.add_context(context['context_id'], 'current_plan', {})
                elif not evaluation.get('execution_complete', False):
                    # More functions need to be executed in the current plan
                    execution_continues = True
                else:
                    # All functions executed but not enough information
                    execution_continues = False
                
                # If the loop will continue, show a message to the user
                if execution_continues and iteration < max_iterations:
                    # Send a message letting the user know we're continuing to process
                    await asyncio.to_thread(
                        self.slack_client.chat_postMessage,
                        channel=ai_request.channel_id,
                        thread_ts=effective_thread_ts,
                        text=f"🔄 *Still working on your request...* (iteration {iteration})\n\nI need to gather more information to fully answer your question. Click the *Stop Processing* button above if you want to interrupt me."
                    )
            
            # Check if we stopped due to user request
            if context_manager.get_context(context['context_id']).get('stop_requested', False):
                # If we have any results so far, generate a partial response
                if execution_results:
                    message_handler.update_stage("Generating partial response")
                    await message_handler.send_thinking_message(initial=False)
                    
                    partial_response = self.commander.response(
                        order=order,
                        execution_results=context_manager.get_context(context['context_id']).get('execution_results', {})
                    )
                    
                    # Send partial response and delete thinking message
                    await self._send_response(partial_response, ai_request, effective_thread_ts)
                    await message_handler.delete_thinking_message()
                    
                    # Add to conversation history
                    await conversation_manager.add_message("user", ai_request.text, ai_request.message_ts)
                    await conversation_manager.add_message("assistant", partial_response)
                
                return
            
            # 5. Generate final response from Commander
            message_handler.update_stage("Commander generating response")
            await message_handler.send_thinking_message(initial=False)
            
            final_response = self.commander.response(
                order=order,
                execution_results=context_manager.get_context(context['context_id']).get('execution_results', {})
            )
            
            message_handler.update_stage("Commander generating response", completed=True)
            
            # 6. Send final message and delete thinking message
            await self._send_response(final_response, ai_request, effective_thread_ts)
            await message_handler.delete_thinking_message()
            
            # 7. Add to conversation history
            await conversation_manager.add_message("user", ai_request.text, ai_request.message_ts)
            await conversation_manager.add_message("assistant", final_response)
            
            # Log execution time
            elapsed_time = time.time() - start_time
            logger.info(f"Processed message in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await message_handler.send_error_message(str(e))
    
    async def _send_response(self, response: str, ai_request: AIRequest, thread_ts: str):
        """Send the response to Slack."""
        try:
            await asyncio.to_thread(
                self.slack_client.chat_postMessage,
                channel=ai_request.channel_id,
                thread_ts=thread_ts,
                text=response
            )
        except SlackApiError as e:
            logger.error(f"Error sending response: {e.response.get('error', '')}")

    async def handle_interaction_payload(self, payload: Dict[str, Any]) -> bool:
        """
        Handle Slack interaction payloads (button clicks, etc.)
        
        Args:
            payload: The interaction payload from Slack
            
        Returns:
            True if a stop was requested, False otherwise
        """
        logger.info(f"Received interaction payload type: {payload.get('type')}")
        
        if payload.get("type") != "block_actions":
            return False
        
        # Extract the action details
        actions = payload.get("actions", [])
        if not actions:
            return False
        
        action = actions[0]
        action_id = action.get("action_id")
        
        if action_id == "stop_processing":
            # Extract context information
            channel_id = payload.get("channel", {}).get("id")
            message_ts = payload.get("message", {}).get("ts")
            user_id = payload.get("user", {}).get("id")
            
            # Find the context associated with this message
            context_id = None
            for ctx_id, ctx in context_manager.contexts.items():
                if ctx.get("channel_id") == channel_id:
                    context_id = ctx_id
                    break
            
            if not context_id:
                logger.warning(f"Could not find context for stop request in channel {channel_id}")
                return False
            
            logger.info(f"Stop requested by user {user_id} for context {context_id}")
            
            # Update the message to show that stopping was requested
            try:
                await asyncio.to_thread(
                    self.slack_client.chat_update,
                    channel=channel_id,
                    ts=message_ts,
                    text="Stopping processing...",
                    blocks=[
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "⚠️ *Processing stopped by user request*"
                            }
                        }
                    ]
                )
                
                # We'll also send a message in the thread to acknowledge
                await asyncio.to_thread(
                    self.slack_client.chat_postMessage,
                    channel=channel_id,
                    thread_ts=context_manager.contexts[context_id].get("thread_ts"),
                    text="Processing stopped by user request. Feel free to ask a new question."
                )
                
                # Mark this context as having a stop requested
                context_manager.add_context(context_id, "stop_requested", True)
                return True
                
            except SlackApiError as e:
                logger.error(f"Error updating stop message: {e.response.get('error', '')}")
                return False
        
        return False 