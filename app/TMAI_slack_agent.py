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
import requests
import base64
from typing import Dict, List, Any, Optional, Union, Callable, Iterable

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from llm.openai_client import OpenaiClient
from ops_conversation_db.conversation_db import get_db, load_conversation_from_db, save_conversation_to_db
from tools.tool_schema_linear import LINEAR_SCHEMAS
from tools.tool_schema_slack import SLACK_SCHEMAS  
from tools.tool_schema_semantic_search import SEMANTIC_SEARCH_SCHEMAS
from rate_limiter import global_limiter, slack_limiter, linear_limiter, openai_limiter
from ops_linear_db.linear_client import LinearClient
from agent import Commander, Captain, Soldier, get_api_call_tracker, log_api_call_report
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
    "website": {
        "enabled": True
    },
    "gdrive": {
        "enabled": os.getenv("GOOGLE_CLIENT_ID") is not None
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
        self.progress_message_ids = []  # Track IDs of progress messages
    
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
                    blocks=blocks,
                    unfurl_links=False,
                )
                self.message_ts = response["ts"]
                return response
            else:
                return await asyncio.to_thread(
                    self.slack_client.chat_update,
                    channel=self.channel_id,
                    ts=self.message_ts,
                    text="TMAI processing your request...",
                    blocks=blocks,
                    unfurl_links=False,
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
        # If no stages completed yet, show a simple thinking message
        if not self.stages_completed and not self.current_stage:
            status_text = "*Working on your request...*"
        else:
            # Otherwise, construct status text showing only completed and current stage
            status_text = "*Status:*\n```\n"
            
            for stage in self.stages_completed:
                status_text += f"✓ {stage}\n"
            
            if self.current_stage:
                status_text += f"➤ {self.current_stage}\n"
                
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
                text=f"Sorry, I encountered an error: {error_msg}",
                unfurl_links=False,
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
            
    async def delete_all_progress_messages(self):
        """Delete all tracked progress messages."""
        try:
            # Delete thinking message first
            await self.delete_thinking_message()
            
            # Delete all progress messages (Commander, Plan, Soldier execute)
            for msg_ts in self.progress_message_ids:
                try:
                    await asyncio.to_thread(
                        self.slack_client.chat_delete,
                        channel=self.channel_id,
                        ts=msg_ts
                    )
                except SlackApiError as e:
                    logger.warning(f"Could not delete progress message {msg_ts}: {e.response.get('error', '')}")
                    
        except Exception as e:
            logger.warning(f"Error deleting progress messages: {str(e)}")

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
                    blocks=blocks,
                    unfurl_links=False,
                )
                self.message_ts = response["ts"]
                return response
            else:
                return await asyncio.to_thread(
                    self.slack_client.chat_update,
                    channel=self.channel_id,
                    ts=self.message_ts,
                    text="TMAI processing your request...",
                    blocks=blocks,
                    unfurl_links=False,
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
            history_result = self._format_history_for_context(self.sender_name)
            logger.debug(f"Returning formatted history with {len(history_result)} lines")
            return history_result
        
        # Then try database
        db_messages = load_conversation_from_db(channel_id, thread_ts)
            
        if db_messages:
            self.history = db_messages
            self.conversation_history[self.conversation_key] = db_messages
            logger.debug(f"Loaded conversation history from database with {len(db_messages)} messages")
            history_result = self._format_history_for_context(self.sender_name)
            logger.debug(f"Returning formatted history with {len(history_result)} lines")
            return history_result
        
        # Finally rebuild from Slack API
        try:
            if is_direct_message:
                # For direct messages, fetch history from the channel
                # But still filter by thread_ts when processing
                logger.debug(f"Fetching DM history from Slack API for {channel_id}")
                response = self.slack_client.conversations_history(
                    channel=channel_id,
                    limit=20  # Fetch last 20 messages
                )
            else:
                # For threaded messages, fetch replies to the specific thread
                logger.debug(f"Fetching threaded conversation from Slack API for {channel_id}:{thread_ts}")
                response = self.slack_client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts
                )
            
            if response.get("ok") and response.get("messages"):
                messages = response.get("messages", [])
                logger.debug(f"Received {len(messages)} raw messages from Slack API")
                
                # Different handling for direct messages vs mentions
                if is_direct_message:
                    # For DMs using conversations_history, we need to filter to the specific thread
                    # or just the main messages (those without a thread_ts)
                    filtered_messages = []
                    for msg in messages:
                        msg_thread_ts = msg.get("thread_ts")
                        # Keep messages from this thread or messages without a thread_ts
                        if msg_thread_ts == thread_ts or (not msg_thread_ts and msg.get("ts") == thread_ts):
                            filtered_messages.append(msg)
                            
                    logger.debug(f"Filtered to {len(filtered_messages)} messages for thread {thread_ts}")
                    messages = filtered_messages
                    # Still need to reverse for chronological order
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
                    if is_bot and ("TMAI's neuron firing" in text or "is thinking" in text):
                        logger.debug(f"Skipping processing message: {text[:30]}...")
                        continue
                    
                    # Skip in-progress messages unless it's a real user message or a final greeting message
                    # This helps with debugging but doesn't affect the conversation
                    if is_bot and "Commander:" in text:
                        logger.debug(f"Skipping intermediate Commander message")
                        continue
                        
                    if is_bot and "Plan:" in text:
                        logger.debug(f"Skipping intermediate Plan message")
                        continue
                        
                    if is_bot and "Soldier execute:" in text:
                        logger.debug(f"Skipping intermediate Soldier message")
                        continue
                    
                    # Ensure we keep actual conversational messages
                    # Check if this is likely a greeting or a final reply (not an intermediate message)
                    is_final_response = is_bot and not any(marker in text for marker in ["Commander:", "Plan:", "Soldier execute:", "neuron firing", "is thinking"])
                    
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
                history_context.append(f"**{sender_name}):** {msg_content}")
            else:
                # Format assistant messages
                content = msg["content"]
                if len(content) > 200:
                    content = content[:200] + "... (content truncated)"
                history_context.append(f"**Assistant):** {content}")
        
        # Add a separator at the end to clearly distinguish history from current query
        if history_context:
            history_context.append("--------------------------------------------------\n")
        
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
        self.ai_model = ai_model
        
        # Store model configuration for different phases
        # Default to using the main model for all roles if not specified
        self.model_config = {
            "commander": model_config.get("commander", ai_model) if model_config else ai_model,
            "captain": model_config.get("captain", ai_model) if model_config else ai_model,
            "soldier": model_config.get("soldier", ai_model) if model_config else ai_model
        }
        
        logger.info(f"Initialized with model config: Commander={self.model_config['commander']}, " +
                    f"Captain={self.model_config['captain']}, Soldier={self.model_config['soldier']}")
        
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
        
        # Initialize agent components with their respective models
        self._initialize_agent_components()
        
        # Initialize conversation store
        self.conversation_history = {}
        
        logger.debug("TMAI Slack Agent initialized")
        
    def _initialize_agent_components(self):
        """Initialize the agent components (Commander, Captain, Soldier)."""
        # These are initialized without context_id as it will be set before use
        self.commander = Commander(
            model=self.model_config.get("commander", self.ai_model),
            prompts=self.prompts
        )
        
        # Use specific models or default to ai_model
        commander_model = self.model_config.get("commander", self.ai_model) if self.model_config else self.ai_model
        captain_model = self.model_config.get("captain", self.ai_model) if self.model_config else self.ai_model 
        soldier_model = self.model_config.get("soldier", self.ai_model) if self.model_config else self.ai_model
        
        self.captain = Captain(
            model=captain_model,
            prompts=self.prompts
        )
        
        self.soldier = Soldier(
            model=soldier_model,
            prompts=self.prompts
        )
        
        logger.info(f"Agent components initialized with models: Commander={commander_model}, Captain={captain_model}, Soldier={soldier_model}")

    def _set_context_for_agents(self, context_id: str):
        """Set the context ID for all agent components to enable cancellation."""
        self.commander.set_context_id(context_id)
        self.captain.set_context_id(context_id)
        self.soldier.set_context_id(context_id)
        logger.debug(f"Set context_id={context_id} for all agent components")

    def update_model_config(self, model_config: Dict[str, str]):
        """Update the model configuration and reinitialize components if needed."""
        changed = False
        
        # Check which components need to be reinitialized
        for component in ["commander", "captain", "soldier"]:
            if component in model_config and model_config[component] != self.model_config.get(component):
                self.model_config[component] = model_config[component]
                changed = True
                
        # Reinitialize components if configuration changed
        if changed:
            self._initialize_agent_components()
            return True
            
        return False
    
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
                                "@" + user_info["user"]["profile"].get("display_name")
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
    
    async def execute_functions_loop(self, plan, ready_functions, user_query, previous_results, context_id):
        """Execute functions sequentially (no batching)."""
        logger.debug(f"Executing {len(ready_functions)} functions sequentially: {ready_functions}")
        
        # Set context ID in all agent components before execution
        self._set_context_for_agents(context_id)
        
        # Check for stop request before starting
        if context_manager.get_context(context_id).get('stop_requested', False):
            logger.info(f"Stop requested before executing {len(ready_functions)} functions")
            return {}
        
        results = {}
        
        # Execute functions one by one
        for function_name in ready_functions:
            # Check for stop request before each function
            if context_manager.get_context(context_id).get('stop_requested', False):
                logger.info(f"Stop requested during execution, stopping after {len(results)} functions")
                return results
            
            try:
                # Execute a single function
                logger.info(f"Executing function: {function_name}")
                result = await self.soldier.execute(
                    plan=plan,
                    function_name=function_name,
                    user_query=user_query,
                    previous_results=previous_results
                )
                
                # Process the result
                if isinstance(result, dict) and result.get("requires_modal_approval", False):
                    logger.warning(f"Found requires_modal_approval flag - this should be handled by Commander")
                    results[function_name] = result
                else:
                    # Add successful result
                    results[function_name] = result # Limit the result to 300 characters
                    
                # Update previous_results after each function to make them available to subsequent functions
                previous_results[function_name] = result # Limit the result to 300 characters
                
            except Exception as e:
                # Handle exceptions
                logger.error(f"Error executing function {function_name}: {str(e)}")
                results[function_name] = {
                    "function": function_name,
                    "error": str(e),
                    "result": None
                }
            
            # Check for stop request after each function
            if context_manager.get_context(context_id).get('stop_requested', False):
                logger.info(f"Stop requested after executing {function_name}")
                return results
                
        return results
        
    def _format_level_results(self, level_functions, level_results):
        """Format level results for display in Slack."""
        execution_output = "```\n"
        for function_name in level_functions:
            if function_name in level_results:
                success = "✓" if level_results[function_name].get("error") is None else "✗"
            else:
                success = "?"
            execution_output += f"{success} {function_name}\n"
        execution_output += "```"
        return execution_output

    async def process_slack_message(self, ai_request: AIRequest, thread_ts: Optional[str] = None, 
                               is_direct_message: bool = False, model_overrides: Optional[Dict[str, str]] = None):
        """
        Process messages using the Commander-Captain-Soldier agent workflow.
        """
        # Start measuring execution time 
        start_time = time.time()
        
        # Determine if this is in a thread
        effective_thread_ts = thread_ts or ai_request.message_ts
        
        # Generate a conversation key - standard format for all types
        conversation_key = f"{ai_request.channel_id}:{effective_thread_ts}"
        logger.debug(f"Generated conversation_key: {conversation_key} (DM: {is_direct_message})")
        
        # Apply any model overrides for this request
        if model_overrides:
            logger.info(f"Applying model overrides: {model_overrides}")
            # Use the new update_model_config method to update models if needed
            self.update_model_config(model_overrides)
        
        # Initialize progress message handler
        message_handler = ProgressiveMessageHandler(
            slack_client=self.slack_client,
            channel_id=ai_request.channel_id,
            message_ts=ai_request.message_ts or "",
            thread_ts=effective_thread_ts
        )

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
            
            # Store the message handler in the context for potential stop requests
            context_manager.add_context(context['context_id'], 'message_handler', message_handler)
            
            # Store the raw conversation history in the context for tool access
            context_manager.add_context(context['context_id'], 'conversation_history', conversation_manager.history)
            
            # Set context ID in all agent components to enable cancellation
            self._set_context_for_agents(context['context_id'])
            
            # Send initial thinking message only if message_ts isn't already set (from app.py)
            if not ai_request.message_ts:
                await message_handler.send_thinking_message()
            
            # Extract URLs from text
            urls = self.extract_urls(ai_request.text)
            if urls:
                ai_request.urls = urls
            
            # Process image files if any
            image_data = None
            if ai_request.files:
                headers = {
                    "Authorization": f"Bearer {self.slack_bot_token}"
                }
                download_url = ai_request.files[0].get("url_private_download")
                response = requests.get(download_url, headers=headers)
                
                # Encode the image bytes to base64 string
                image_data = base64.b64encode(response.content).decode('utf-8')
                logger.info(f"Successfully encoded image to base64, length: {len(image_data)} bytes")
                
                # Update stage for image analysis
                message_handler.update_stage("Analyzing image content")
                await message_handler.send_thinking_message(initial=False)
                
                # Call Commander.analyze_image directly
                image_context = self.commander.analyze_image(image_data)
                
                # Mark image analysis stage as completed
                message_handler.update_stage("Analyzing image content", completed=True)
                
                # Display image analysis results in a code block if available
                if image_context:
                    # Truncate if too long (for UI purposes)
                    display_context = image_context
                    if len(display_context) > 500:
                        display_context = display_context[:497] + "..."
                        
                    image_analysis_output = f"```\nImage Analysis:\n{display_context}\n```"
                    
                    response = await asyncio.to_thread(
                        self.slack_client.chat_postMessage,
                        channel=ai_request.channel_id,
                        thread_ts=effective_thread_ts,
                        text=f"{image_analysis_output}",
                        unfurl_links=False,
                    )
                    # Track message ID for later deletion
                    message_handler.progress_message_ids.append(response['ts'])
            
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
            
            # 1. Update stage BEFORE Commander API call
            message_handler.update_stage("Commander assigning tasks")
            await message_handler.send_thinking_message(initial=False)
            
            # Check if this is an image-only request (empty text but has files)
            image_only_request = not ai_request.text.strip() and len(ai_request.files) > 0
            
            # Prepare the user query
            if image_only_request:
                user_query = f"Current prompt from user: {ai_request.sender_name} : [User uploaded an image]"
                logger.info(f"Processing image-only request from {ai_request.sender_name}")
            else:
                user_query = f"Current prompt from user: {ai_request.sender_name} : {ai_request.text}"
                
                # If text is minimal but there are files, enhance the prompt
                if len(ai_request.text.strip()) < 10 and len(ai_request.files) > 0:
                    logger.info(f"Processing message with minimal text and attached files: '{ai_request.text}'")
                    user_query += " [User also attached an image/file with this message]"
            
            # Make the Commander API call
            commander_result = self.commander.assign_tasks(
                user_query=user_query,
                history=history_for_commander,
                image_data=image_data,
                image_context=image_context if 'image_context' in locals() else None
            )
            
            # Mark Commander stage as completed AFTER API call
            message_handler.update_stage("Commander assigning tasks", completed=True)
            
            # Display Commander results in code block - simplified to show only the order text
            order = commander_result.get('order', ai_request.text)
            commander_output = f"```\n{order}\n```"
            response = await asyncio.to_thread(
                self.slack_client.chat_postMessage,
                channel=ai_request.channel_id,
                thread_ts=effective_thread_ts,
                text=f"{commander_output}",
                unfurl_links=False,
            )
            # Track message ID for later deletion
            message_handler.progress_message_ids.append(response['ts'])
            
            logger.info(f"Commander assigned platforms: {commander_result.get('platform', [])}")
            context_manager.add_context(context['context_id'], 'commander_result', commander_result)
            
            # Extract platforms and order
            platforms = commander_result.get('platform', [])
            direct_response = commander_result.get('direct_response', '')
            
            # Ensure platforms is always a list (Commander can return a string for a single platform)
            if isinstance(platforms, str):
                platforms = [platforms]
                logger.debug(f"Converted platform string '{platforms[0]}' to list")
            
            # If direct_response is provided or no platforms, send direct response
            if direct_response and not platforms:
                message_handler.update_stage("I want to say...")
                await message_handler.send_thinking_message(initial=False)
                
                # Use the direct response if provided, otherwise generate one
                if direct_response:
                    final_response = direct_response
                    logger.debug("Using direct_response field from Commander")
                
                # Mark Direct response as completed
                message_handler.update_stage("I want to say...", completed=True)
                
                # Send final message (at first step) and delete thinking message
                await self._send_response(final_response, ai_request, effective_thread_ts, stream=False)
                # Delete all progress messages instead of just the thinking message
                await message_handler.delete_all_progress_messages()

                # Add to conversation history
                await conversation_manager.add_message("user", ai_request.text, ai_request.message_ts)
                await conversation_manager.add_message("assistant", final_response)
                
                # Log token and API call usage if in debug mode
                if logger.isEnabledFor(logging.DEBUG):
                    for agent in [self.commander, self.captain, self.soldier]:
                        agent.client.log_token_usage(logging.DEBUG)
                    
                    # Log API call report
                    log_api_call_report(logging.DEBUG)
                
                return
            
            # Replace regular thinking message with one that has a stop button
            await message_handler.send_thinking_message_with_stop(initial=False)
            
            # Initialize execution variables
            max_iterations = 3  # Prevent infinite loops
            iteration = 0
            current_order = order
            execution_results = {}
            should_replan = True  # Control whether to call planning in each iteration
            current_plan = None   # Keep track of current plan
            
            # Main execution loop
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"Starting execution iteration {iteration}")
                
                # Check if a stop was requested
                if context_manager.get_context(context['context_id']).get('stop_requested', False):
                    logger.info(f"Stop requested for context {context['context_id']}, breaking execution loop")
                    break
                
                # 2. Captain: Planning step (only when needed)
                if should_replan:
                    # Update stage BEFORE Captain API call
                    message_handler.update_stage(f"Captain planning functions (iteration {iteration})")
                    await message_handler.send_thinking_message_with_stop(initial=False)
                    
                    # Make the Captain API call for planning
                    if iteration > 1 or execution_results:
                        current_plan = self.captain.plan(current_order, platforms, execution_results)
                    else:
                        current_plan = self.captain.plan(current_order, platforms)
                    
                    # Mark Captain planning as completed AFTER API call
                    message_handler.update_stage(f"Captain planning functions (iteration {iteration})", completed=True)
                    
                    # Display Plan results in code block - simplified to show only plan_description
                    plan_description = current_plan.get('plan_description', 'No plan description available')
                    
                    # Clean up Jinja template conditionals from plan description if they exist
                    if "{% if previous_results %}" in plan_description:
                        # Remove the conditional and just keep the main content
                        plan_description = plan_description.split("{% if previous_results %}")[0].strip()
                    
                    plan_output = f"```\n{plan_description}\n```"
                    response = await asyncio.to_thread(
                        self.slack_client.chat_postMessage,
                        channel=ai_request.channel_id,
                        thread_ts=effective_thread_ts,
                        text=f"{plan_output}",
                        unfurl_links=False,
                    )
                    # Track message ID for later deletion
                    message_handler.progress_message_ids.append(response['ts'])
                        
                    # Log plan information
                    if 'function_levels' in current_plan:
                        total_functions = sum(len(level) for level in current_plan['function_levels'])
                        logger.info(f"Captain planned {total_functions} functions across {len(current_plan['function_levels'])} levels")
                    else:
                        logger.warning("Plan doesn't contain function_levels structure")
                    
                    # Check if updateIssue is in the plan without filterIssues before it
                    if 'function_levels' in current_plan:
                        # Find updateIssue in the function levels
                        update_issue_level = None
                        update_issue_index = None
                        filter_issues_before_update = False
                        
                        for level_idx, level in enumerate(current_plan['function_levels']):
                            if 'updateIssue' in level:
                                update_issue_level = level_idx
                                update_issue_index = level.index('updateIssue')
                                
                                # Check if filterIssues appears in the same or earlier level
                                for check_level_idx in range(level_idx + 1):
                                    if check_level_idx == level_idx:
                                        # If in same level, check if it's before updateIssue
                                        if 'filterIssues' in level[:update_issue_index]:
                                            filter_issues_before_update = True
                                            break
                                    else:
                                        # If in earlier level, just check if it exists
                                        if 'filterIssues' in current_plan['function_levels'][check_level_idx]:
                                            filter_issues_before_update = True
                                            break
                                
                                break  # Found updateIssue, no need to check further levels
                        
                        # If we need to add filterIssues
                        if update_issue_level is not None and not filter_issues_before_update:
                            logger.info("Found updateIssue without filterIssues before it - adding filterIssues")
                            
                            # Get parameters for filterIssues from updateIssue parameters
                            update_params = {}
                            for func_name in current_plan['function_levels'][update_issue_level]:
                                if func_name == 'updateIssue' and 'parameters' in current_plan and func_name in current_plan['parameters']:
                                    update_params = current_plan['parameters'][func_name]
                                    break
                            
                            # Extract issue number
                            issue_number = None
                            if update_params:
                                issue_number = update_params.get('issue_number') or update_params.get('issueNumber')
                            
                            # Add filterIssues as a new first level if issue_number is available
                            if issue_number:
                                # Create parameters for filterIssues
                                if 'parameters' not in current_plan:
                                    current_plan['parameters'] = {}
                                
                                current_plan['parameters']['filterIssues'] = {
                                    'number': issue_number,
                                    'limit': 1
                                }
                                
                                # Add filterIssues to a new first level
                                current_plan['function_levels'].insert(0, ['filterIssues'])
                                
                                # Update plan description
                                current_plan['plan_description'] = "First, we need to see the content of that issue. " + current_plan.get('plan_description', '')
                    
                    # Store plan in context
                    context_manager.add_context(context['context_id'], 'current_plan', current_plan)
                    
                    # Reset should_replan flag after planning
                    should_replan = False
                else:
                    logger.info("Reusing existing plan - no need to replan")
                
                # Check if a stop was requested before continuing
                if context_manager.get_context(context['context_id']).get('stop_requested', False):
                    logger.info(f"Stop requested after planning for context {context['context_id']}")
                    break
                
                # Get function levels to execute
                function_levels = current_plan.get('function_levels', [])
                
                # If no functions in plan, generate response with what we have
                if not function_levels or all(len(level) == 0 for level in function_levels):
                    message_handler.update_stage("No functions to execute")
                    await message_handler.send_thinking_message_with_stop(initial=False)
                    
                    # Mark no functions stage as completed
                    message_handler.update_stage("No functions to execute", completed=True)
                    
                    # Generate response from Commander
                    final_response = self.commander.response(order, execution_results, stream=True)
                    
                    # Send final message and delete thinking message
                    final_response_text = await self._send_response(final_response, ai_request, effective_thread_ts, stream=True)
                    # Delete all progress messages instead of just the thinking message
                    await message_handler.delete_all_progress_messages()
                    
                    # Add to conversation history
                    await conversation_manager.add_message("user", ai_request.text, ai_request.message_ts)
                    await conversation_manager.add_message("assistant", final_response_text)
                    
                    # Log token and API call usage if in debug mode
                    if logger.isEnabledFor(logging.DEBUG):
                        for agent in [self.commander, self.captain, self.soldier]:
                            agent.client.log_token_usage(logging.DEBUG)
                        
                        # Log API call report
                        log_api_call_report(logging.DEBUG)
                    
                    return
                
                # 3. Execute functions level by level
                all_functions_executed = True  # Flag to track execution completion
                all_level_results = {}  # Store all level results for display
                
                for level_index, level_functions in enumerate(function_levels):
                    # Skip empty levels
                    if not level_functions:
                        continue
                        
                    # Update stage BEFORE executing functions
                    message_handler.update_stage(f"Executing level {level_index+1}: {len(level_functions)} functions")
                    await message_handler.send_thinking_message_with_stop(initial=False)
                    
                    # Check for stop request before executing level
                    if context_manager.get_context(context['context_id']).get('stop_requested', False):
                        logger.info(f"Stop requested before executing level {level_index+1}")
                        all_functions_executed = False
                        break
                    
                    # Execute all functions in this level in parallel
                    try:
                        level_results = await self.execute_functions_loop(
                            plan=current_plan,
                            ready_functions=level_functions,
                            user_query=current_order,
                            previous_results=execution_results,
                            context_id=context['context_id']  # Pass context_id for stop checks
                        )
                        
                        # Store level results for display
                        execution_output = self._format_level_results(level_functions, level_results)
                        
                        # Add results to the cumulative results
                        execution_results.update(level_results)
                        context_manager.add_context(context['context_id'], 'execution_results', execution_results)
                        
                        # Mark function execution as completed AFTER API calls
                        message_handler.update_stage(f"Executing level {level_index+1}: {len(level_functions)} functions", completed=True)
                        await message_handler.send_thinking_message_with_stop(initial=False)
                        
                        # Display function execution results in code block - simplified to show function names with check marks
                        execution_output = "```\n"
                        for func_name, result in level_results.items():
                            success = "✓" if result.get("error") is None else "✗"
                            execution_output += f"{success} {func_name}\n"
                        execution_output += "```"
                        
                        response = await asyncio.to_thread(
                            self.slack_client.chat_postMessage,
                            channel=ai_request.channel_id,
                            thread_ts=effective_thread_ts,
                            text=f"{execution_output}",
                            unfurl_links=False,
                        )
                        # Track message ID for later deletion
                        message_handler.progress_message_ids.append(response['ts'])
                        
                    except Exception as e:
                        logger.error(f"Error executing level {level_index+1}: {str(e)}")
                        all_functions_executed = False
                        # If a level fails, we should consider replanning
                        should_replan = True
                        break
                    
                    # Check for stop request after executing level
                    if context_manager.get_context(context['context_id']).get('stop_requested', False):
                        logger.info(f"Stop requested after executing level {level_index+1}")
                        all_functions_executed = False
                        break
                
                # 4. Captain evaluation - only if all functions were executed or we need to replan
                if all_functions_executed or should_replan:
                    # Update stage BEFORE Captain evaluation API call
                    message_handler.update_stage("Captain evaluating results")
                    await message_handler.send_thinking_message_with_stop(initial=False)
                    
                    # Make the Captain API call for evaluation
                    evaluation = self.captain.evaluate(order, current_plan, execution_results)
                    
                    # Mark Captain evaluation as completed AFTER API call
                    message_handler.update_stage("Captain evaluating results", completed=True)
                    
                    logger.info(f"Captain evaluation: change_plan={evaluation.get('change_plan', False)}, response_ready={evaluation.get('response_ready', False)}")
                    
                    # Decision based on evaluation
                    if evaluation.get('response_ready', True):
                        # We have enough information to generate a response
                        logger.info("Captain indicates we have enough information to respond")
                        break
                        
                    elif evaluation.get('change_plan', False):
                        # We need to revise the plan
                        logger.info(f"Captain indicates we need to change the plan: {evaluation.get('error_description', 'No error description provided')}")
                        
                        # Enrich the order with error description for next planning iteration
                        error_desc = evaluation.get('error_description', 'Plan requires revision')
                        current_order = f"{order}\n\nPrevious plan had issues: {error_desc}"
                        
                        # Set flag to replan in the next iteration
                        should_replan = True
                        continue
                        
                    else:
                        # All functions executed but we don't have enough information
                        logger.info("Captain indicates we need more information but current plan is exhausted")
                        # We'll exit the loop and generate a response with what we have
                        break
                
                # If we get here without replanning set, it means we should exit the loop
                if not should_replan:
                    break
            
            # Check if we stopped due to user request
            if context_manager.get_context(context['context_id']).get('stop_requested', False):
                message_handler.update_stage("Processing stopped by user")
                
                # If we have any results so far, generate a partial response
                if execution_results:
                    final_response = self.commander.response(
                        order=order,
                        execution_results=execution_results,
                        stream=True
                    )
                    
                    
                    # Send partial response and delete thinking message
                    final_response_text = await self._send_response(final_response, ai_request, effective_thread_ts, stream=True)
                    # Delete all progress messages instead of just the thinking message
                    await message_handler.delete_all_progress_messages()
                    
                    # Add to conversation history
                    await conversation_manager.add_message("user", ai_request.text, ai_request.message_ts)
                    await conversation_manager.add_message("assistant", final_response_text)
                else:
                    await message_handler.delete_all_progress_messages()
                
                # Log token and API call usage if in debug mode
                if logger.isEnabledFor(logging.DEBUG):
                    for agent in [self.commander, self.captain, self.soldier]:
                        agent.client.log_token_usage(logging.DEBUG)
                    
                    # Log API call report
                    log_api_call_report(logging.DEBUG)
                
                return
            
            # 5. Update stage BEFORE Commander response API call
            message_handler.update_stage("Commander generating response")
            await message_handler.send_thinking_message(initial=False)
            
            # Make the Commander API call for the final response
            final_response = self.commander.response(
                order=order,
                execution_results=execution_results,
                stream=True
            )
            
            # Mark Commander response as completed AFTER API call
            message_handler.update_stage("Commander generating response", completed=True)
            
            # Check for Linear actions requiring modal approval
            linear_approval_needed = False
            for func_name, result in execution_results.items():
                if isinstance(result, dict) and result.get("requires_modal_approval", False):
                    linear_approval_needed = True
                    logger.info(f"Found Linear function {func_name} requiring modal approval")
                    
                    # Get context data
                    channel_id = ai_request.channel_id
                    thread_ts = effective_thread_ts
                    
                    
                    # First send a message to get a trigger_id
                    try:
                        # Send the final response first
                        final_response_text = await self._send_response(final_response, ai_request, effective_thread_ts, stream=True)
                        
                        # Then send the approval request
                        response = await asyncio.to_thread(
                            self.slack_client.chat_postMessage,
                            channel=channel_id,
                            thread_ts=thread_ts,
                            text=f"Please approve this Linear action",
                            blocks=[
                                {
                                    "type": "section",
                                    "text": {
                                        "type": "mrkdwn",
                                        "text": f"*Requesting approval*\nPlease review and approve the Linear {func_name} action"
                                    }
                                },
                                {
                                    "type": "actions",
                                    "elements": [
                                        {
                                            "type": "button",
                                            "text": {
                                                "type": "plain_text",
                                                "text": "Open Review Form",
                                                "emoji": True
                                            },
                                            "style": "primary",
                                            "value": json.dumps({
                                                "action": "open_linear_modal",
                                                "function": func_name,
                                                "params_id": f"params_{func_name}_{context['context_id'][-8:]}",
                                                "context_id": context['context_id']
                                            }),
                                            "action_id": "open_linear_modal"
                                        }
                                    ]
                                }
                            ],
                            unfurl_links=False,
                        )
                        
                        # Store the parameters in context for later retrieval
                        params_id = f"params_{func_name}_{context['context_id'][-8:]}"
                        context_manager.add_context(context['context_id'], params_id, result.get('parameters', {}))

                        # Store pending approval details in context
                        context_manager.add_context(
                            context['context_id'], 
                            f"pending_approval_{func_name}", 
                            result
                        )
                        
                        # Delete thinking message since we've already sent the response
                        await message_handler.delete_all_progress_messages()
                        
                        # Add to conversation history
                        await conversation_manager.add_message("user", ai_request.text, ai_request.message_ts)
                        await conversation_manager.add_message("assistant", final_response_text)
                        
                        return
                        
                    except Exception as e:
                        logger.error(f"Error creating approval message: {str(e)}")
                        # Continue with normal flow if we can't send approval
            
            # 6. Send final message and delete thinking message (if no approval needed)
            if not linear_approval_needed:
                final_response_text = await self._send_response(final_response, ai_request, effective_thread_ts, stream=True)
                # Delete all progress messages instead of just the thinking message
                await message_handler.delete_all_progress_messages()
                
                # 7. Add to conversation history
                await conversation_manager.add_message("user", ai_request.text, ai_request.message_ts)
                await conversation_manager.add_message("assistant", final_response_text)
            
            # Log execution time
            elapsed_time = time.time() - start_time
            logger.info(f"Processed message in {elapsed_time:.2f} seconds with {iteration} iterations")
            
            # Log token and API call usage if in debug mode
            if logger.isEnabledFor(logging.DEBUG):
                logger.info("Generating usage reports")
                for agent in [self.commander, self.captain, self.soldier]:
                    logger.debug(f"Token usage for {agent.__class__.__name__} ({agent.model}):")
                    agent.client.log_token_usage(logging.DEBUG)
                
                # Log API call report
                log_api_call_report(logging.DEBUG)
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await message_handler.send_error_message(str(e))
            
            # Log token and API call usage on error if in debug mode
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    for agent in [self.commander, self.captain, self.soldier]:
                        agent.client.log_token_usage(logging.DEBUG)
                    
                    # Log API call report
                    log_api_call_report(logging.DEBUG)
                except Exception as log_error:
                    logger.error(f"Error generating usage reports: {str(log_error)}")
    
    async def _send_response(self, response: str | Iterable[str], ai_request: AIRequest, thread_ts: str, stream: bool = False):
        """Send the response to Slack."""
        full_response = ""
        buffer = ""
        full_response_to_return = ""
        if not stream:
            try:
                await asyncio.to_thread(
                    self.slack_client.chat_postMessage,
                    channel=ai_request.channel_id,
                    thread_ts=thread_ts,
                    text=response,
                    unfurl_links=False,
                )
                return response
            except SlackApiError as e:
                logger.error(f"Error sending response: {e.response.get('error', '')}")
                return response
        elif stream:
            # For streaming responses, create an initial message
            init_response = await asyncio.to_thread(
                self.slack_client.chat_postMessage,
                channel=ai_request.channel_id,
                thread_ts=thread_ts,
                text="Generating response...",
                unfurl_links=False,
            )
            message_ts = init_response["ts"]
            try:
                # Process each chunk from the stream
                for chunk in response:            
                    if hasattr(chunk, 'type') and chunk.type == 'response.output_text.delta':
                        if hasattr(chunk, 'delta'):
                            delta = chunk.delta
                            full_response += delta
                            buffer += delta
                    
                    # because can't update a message that's too long. Will create a new message. Reset full_response every 3000 characters.
                    if len(full_response) >= 3000:
                        # create a chunk for the first 3000 characters                        
                        await asyncio.to_thread(
                            self.slack_client.chat_update,
                            channel=ai_request.channel_id,
                            ts=message_ts,
                            text=full_response[:3000]
                        )
                        full_response_to_return += full_response[:3000]
                        #after we have sent the first 3000 characters, we can reset the full_response
                        full_response = full_response[3000:]
                        # create a new message for the residual response
                        if len(full_response) > 0:
                            next_message = await asyncio.to_thread(
                                self.slack_client.chat_postMessage,
                                channel=ai_request.channel_id,
                                thread_ts=thread_ts,
                                text=full_response,
                                unfurl_links=False,
                            )
                            full_response_to_return += full_response
                        else:
                            next_message = await asyncio.to_thread(
                                self.slack_client.chat_postMessage,
                                channel=ai_request.channel_id,
                                thread_ts=thread_ts,
                                text="...",
                                unfurl_links=False,
                            )
                        # Use the new message TS for any upcoming updates
                        message_ts = next_message["ts"]
                        buffer = ""
                    # Update the message when buffer is large enough. Reset buffer every 500 characters.
                    if len(buffer) >= 500:
                        await asyncio.to_thread(
                            self.slack_client.chat_update,
                            channel=ai_request.channel_id,
                            ts=message_ts,
                            text=full_response
                        )
                        buffer = ""
                
                # Send the final update with the residual response
                if buffer:
                    await asyncio.to_thread(
                        self.slack_client.chat_update,
                        channel=ai_request.channel_id,
                        ts=message_ts,
                        text=full_response,
                    )
                    full_response_to_return += full_response
                # Return the complete response string for conversation history
                return full_response_to_return
                
            except Exception as e:
                logger.error(f"Error during streaming response: {str(e)}")
                # If streaming fails, return what we've got so far
                return full_response or "Sorry, I had an issue generating the response."

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
            thread_ts = payload.get("container", {}).get("thread_ts") or payload.get("message", {}).get("thread_ts")
            
            # Find the context associated with this message
            context_id = None
            
            # First try with the exact thread_ts
            if thread_ts:
                context_id = f"{channel_id}:{thread_ts}"
                if context_id not in context_manager.contexts:
                    context_id = None
            
            # If not found, try a more flexible approach by searching all contexts in this channel
            if not context_id:
                for ctx_id, ctx in context_manager.contexts.items():
                    if ctx.get("channel_id") == channel_id:
                        # Either exact match, or it's the only context in this channel
                        context_id = ctx_id
                        break
            
            if not context_id:
                # If still not found, check if there are any active contexts
                if context_manager.contexts:
                    # As a last resort, mark ALL active contexts as stopped
                    logger.warning(f"Could not find specific context for stop request in channel {channel_id}, stopping all active contexts")
                    for ctx_id in context_manager.contexts:
                        context_manager.add_context(ctx_id, "stop_requested", True)
                    
                    # Use the first context for UI updates
                    context_id = next(iter(context_manager.contexts))
                else:
                    logger.error(f"No active contexts found for stop request in channel {channel_id}")
                    return False
            
            logger.info(f"Stop requested by user {user_id} for context {context_id}")
            
            # Mark this context as stopped
            context_manager.add_context(context_id, "stop_requested", True)
            
            # Find the message handler for this context if it exists in memory
            # This is a quick way to access the message handler without having to recreate it
            message_handler = None
            active_context = context_manager.get_context(context_id)
            if active_context and 'message_handler' in active_context:
                message_handler = active_context.get('message_handler')
            
            # If we found a message handler, use it to delete all progress messages
            if message_handler and hasattr(message_handler, 'delete_all_progress_messages'):
                try:
                    # Delete all progress messages immediately
                    asyncio.create_task(message_handler.delete_all_progress_messages())
                    logger.info(f"Deleted progress messages for context {context_id}")
                except Exception as e:
                    logger.error(f"Error deleting progress messages: {str(e)}")
            else:
                # If we don't have access to the message handler, at least try to delete the current message
                try:
                    await asyncio.to_thread(
                        self.slack_client.chat_delete,
                        channel=channel_id,
                        ts=message_ts
                    )
                except SlackApiError as e:
                    logger.error(f"Error deleting message: {e.response.get('error', '')}")
            
            # We'll send a message in the thread to acknowledge the stop
            try:
                await asyncio.to_thread(
                    self.slack_client.chat_postMessage,
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text="Processing stopped at your request. Feel free to ask a new question.",
                    unfurl_links=False,
                )
                
                return True
                
            except SlackApiError as e:
                logger.error(f"Error sending stop acknowledgment: {e.response.get('error', '')}")
                return False
                
        elif action_id in ["approve_linear_action", "decline_linear_action"]:
            # Extract necessary information
            try:
                channel_id = payload.get("channel", {}).get("id")
                message_ts = payload.get("message", {}).get("ts")
                user_id = payload.get("user", {}).get("id")
                thread_ts = payload.get("container", {}).get("thread_ts") or payload.get("message", {}).get("thread_ts")
                value = json.loads(action.get("value", "{}"))
                
                # Log the value for debugging
                logger.info(f"TRIGGER DEBUG: Button value: {json.dumps(value)}")
                
                # Get the context ID from the value
                context_id = value.get("context_id")
                if not context_id:
                    logger.error("No context_id found in button value")
                    return False
                
                function_name = value.get("function")
                
                # Get the context FIRST
                context = context_manager.get_context(context_id)
                
                # Check if parameters are referenced by ID or included directly
                params_id = value.get("params_id")
                if params_id and context:
                    # Retrieve parameters from context
                    retrieved_params = context.get(params_id, {})
                    if retrieved_params:
                        parameters = retrieved_params
                    # If no parameters were found by ID, we'll fall back to the ones directly included
                else:
                    # Fallback to direct parameters (for backward compatibility)
                    parameters = value.get("parameters", {})
                
                # If no context found, just log a warning but continue
                if not context:
                    logger.warning(f"No context found for ID {context_id}, but will proceed with action anyway")
                    context = {}
                    
                # Update the approval message
                await asyncio.to_thread(
                    self.slack_client.chat_update,
                    channel=channel_id,
                    ts=message_ts,
                    text=f"Linear Action: {action_id == 'approve_linear_action' and 'Approved' or 'Declined'}",
                    blocks=[
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Linear Action {action_id == 'approve_linear_action' and 'Approved ✅' or 'Declined ❌'}*\n{function_name}"
                            }
                        }
                    ]
                )
                
                # If approved, execute the function with the parameters
                if action_id == "approve_linear_action":
                    # Check if parameters are referenced by ID or included directly
                    params_id = value.get("params_id")
                    if params_id:
                        # Retrieve parameters from context
                        parameters = context_manager.get_context(context_id).get(params_id, {})
                    else:
                        # Fallback to direct parameters (for backward compatibility)
                        parameters = value.get("parameters", {})
                        
                    # Add approved flag to parameters
                    parameters["approved"] = True
                    
                    # Execute the function
                    try:
                        result = await self.soldier.execute(
                            plan={},  # Empty plan as we're executing directly
                            function_name=function_name,
                            user_query="",  # No user query needed for direct execution
                            previous_results={},  # No previous results needed
                            function_args=parameters
                        )
                        
                        # Send a message with the result
                        await asyncio.to_thread(
                            self.slack_client.chat_postMessage,
                            channel=channel_id,
                            thread_ts=thread_ts,
                            text=f"Successfully executed {function_name}: {result.get('result', 'No result')}",
                            unfurl_links=False,
                        )
                    except Exception as e:
                        logger.error(f"Error executing approved function {function_name}: {str(e)}")
                        # Send error message
                        await asyncio.to_thread(
                            self.slack_client.chat_postMessage,
                            channel=channel_id,
                            thread_ts=thread_ts,
                            text=f"Error executing {function_name}: {str(e)}",
                            unfurl_links=False,
                        )
                else:
                    # Declined - just send a message
                    await asyncio.to_thread(
                        self.slack_client.chat_postMessage,
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=f"Action {function_name} was declined.",
                        unfurl_links=False,
                    )
                
                # Remove the pending approval from the context
                context_manager.remove_from_context(context_id, f"pending_approval_{function_name}")
                
                return True
            except Exception as e:
                logger.error(f"Error handling Linear approval: {str(e)}")
                return False
        
        elif action_id == "open_linear_modal":
            # Extract necessary information
            try:
                channel_id = payload.get("channel", {}).get("id")
                message_ts = payload.get("message", {}).get("ts")
                user_id = payload.get("user", {}).get("id")
                thread_ts = payload.get("container", {}).get("thread_ts") or payload.get("message", {}).get("thread_ts")
                
                # CRITICAL: Immediately extract the trigger_id first thing since it expires quickly
                trigger_id = payload.get("trigger_id")
                if not trigger_id:
                    logger.error("No trigger_id in payload, can't open modal")
                    return False
                
                # Log the trigger_id right away
                logger.info(f"TRIGGER DEBUG: Received trigger_id: {trigger_id}")
                
                # Extract the remaining information
                value = json.loads(action.get("value", "{}"))
                
                # Log the value for debugging
                logger.info(f"TRIGGER DEBUG: Button value: {json.dumps(value)}")
                
                # Add detailed trigger_id logging
                logger.info(f"TRIGGER DEBUG: Payload timestamp: {payload.get('trigger_id')}")
                logger.info(f"TRIGGER DEBUG: Action timestamp: {payload.get('action_ts')}")
                
                # Get the context ID from the value
                context_id = value.get("context_id")
                if not context_id:
                    logger.error("No context_id found in button value")
                    return False
                
                function_name = value.get("function")
                
                # Get the context FIRST before trying to access parameters
                context = context_manager.get_context(context_id)
                
                # Check if parameters are referenced by ID or included directly
                params_id = value.get("params_id")
                if params_id and context:
                    # Retrieve parameters from context only if context exists
                    retrieved_params = context.get(params_id, {})
                    if retrieved_params:
                        parameters = retrieved_params
                    else:
                        # If no parameters were found by ID, fallback to direct parameters
                        parameters = value.get("parameters", {})
                else:
                    # Fallback to direct parameters (for backward compatibility)
                    parameters = value.get("parameters", {})
                
                # If no context found, just log a warning but continue with the parameters from the button
                if not context:
                    logger.warning(f"No context found for ID {context_id}, but will proceed with opening modal anyway")
                    context = {}
                    
                # IMPORTANT: Use a separate task for initializing SlackModals to minimize delay
                # before using the trigger_id
                slack_modals = None
                
                # Initialize SlackModals
                try:
                    from ops_slack.slack_modals import SlackModals
                    slack_modals = SlackModals(self.slack_client)
                except Exception as e:
                    logger.error(f"Error initializing SlackModals: {str(e)}")
                    return False
                
                # Open appropriate modal based on function
                if function_name == "createIssue":
                    # Log trigger_id right before modal open
                    logger.info(f"TRIGGER DEBUG: About to open createIssue modal with trigger_id: {trigger_id}")
                    
                    # Ensure we pass the exact trigger_id string without any modifications
                    trigger_id_value = str(trigger_id).strip()
                    
                    # Open create issue modal
                    await slack_modals.open_create_issue_modal(
                        trigger_id=trigger_id_value,
                        prefilled_data=parameters,
                        conversation_id=f"{channel_id}:{thread_ts}"
                    )
                    
                    # Update the button message in the background
                    asyncio.create_task(self._update_button_message(
                        channel_id=channel_id,
                        message_ts=message_ts,
                        text="Creating a new Linear issue",
                        message="*Issue creation form opened*\nPlease complete the form to create the issue."
                    ))
                    
                elif function_name == "updateIssue":
                    # Get issue number from parameters
                    issue_number = parameters.get("issue_number") or parameters.get("issueNumber")
                    
                    if not issue_number:
                        logger.error("No issue number found in parameters")
                        return False
                    
                    # Log trigger_id right before modal open
                    logger.info(f"TRIGGER DEBUG: About to open updateIssue modal with trigger_id: {trigger_id}")
                    
                    # Ensure we pass the exact trigger_id string without any modifications
                    trigger_id_value = str(trigger_id).strip()
                    
                    # Open update issue modal - do this immediately before the trigger_id expires
                    await slack_modals.open_update_issue_modal(
                        trigger_id=trigger_id_value,
                        issue_number=issue_number,
                        prefilled_data=parameters,
                        conversation_id=f"{channel_id}:{thread_ts}"
                    )
                    
                    # Update the button message in the background
                    asyncio.create_task(self._update_button_message(
                        channel_id=channel_id,
                        message_ts=message_ts,
                        text="Updating Linear issue",
                        message=f"*Issue update form opened*\nPlease complete the form to update issue #{issue_number}."
                    ))
                    
                return True
                
            except Exception as e:
                logger.error(f"Error opening Linear modal: {str(e)}")
                
                # Try to send an error message
                try:
                    channel_id = payload.get("channel", {}).get("id")
                    thread_ts = payload.get("container", {}).get("thread_ts") or payload.get("message", {}).get("thread_ts")
                    
                    # Provide more helpful error message based on the error type
                    error_message = str(e)
                    if "invalid_arguments" in error_message:
                        error_message = "Error opening form: The form data may be too large or the request timed out. Please try again with a more concise description or try creating the issue with fewer details."
                    elif "trigger_id_invalid" in error_message:
                        error_message = "Error opening form: Your interaction has expired. Please try again by clicking the button."
                    
                    await asyncio.to_thread(
                        self.slack_client.chat_postMessage,
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=error_message,
                        unfurl_links=False,
                    )
                except Exception:
                    pass
                    
                return False
        
        return False

    async def _update_button_message(self, channel_id: str, message_ts: str, text: str, message: str):
        """Helper method to update button messages after modal is opened"""
        try:
            await asyncio.to_thread(
                self.slack_client.chat_update,
                channel=channel_id,
                ts=message_ts,
                text=text,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": message
                        }
                    }
                ]
            )
        except Exception as e:
            logger.error(f"Error updating button message: {str(e)}")
            
        return False 