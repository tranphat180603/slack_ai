"""
Slack Operations Module for TMAI Agent.
This module handles all interactions with the Slack API, including:
- Message formatting
- User information retrieval
- Channel history search
- Parsing user mentions
- URL extraction
- Message posting and updates
"""

import re
import time
import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

import openai
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Configure logger
logger = logging.getLogger("slack_tools")

class SlackClient:
    """
    Tools for interacting with Slack API.
    
    This class provides methods for:
    1. Searching channel history
    2. Get users
    """
    
    def __init__(
        self, 
        bot_token: str,
        user_token: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Slack tools.
        
        Args:
            bot_token: Slack bot token
            user_token: Optional Slack user token for additional permissions
            openai_api_key: OpenAI API key for AI enhancements
            ai_model: AI model to use for enhancements
            max_retries: Maximum number of retries for API calls
            retry_delay: Base delay between retries in seconds
        """
        # Initialize clients
        self.client = WebClient(token=bot_token)
        self.user_client = WebClient(token=user_token) if user_token else None
        
        
        # Store configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Rate limit tracking
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 200ms minimum between requests
        
        logger.info("Slack tools initialized")

    
    def _handle_retry(self, attempt: int, exception: SlackApiError) -> float:
        """
        Handle retry logic with exponential backoff.
        
        Args:
            attempt: Current attempt number (0-indexed)
            exception: The SlackApiError that triggered the retry
            
        Returns:
            Delay in seconds before next retry
        """
        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
        
        # Add jitter to avoid thundering herd
        import random
        delay = delay * (0.5 + random.random())
        
        # Add longer delay for rate limit errors
        if hasattr(exception, 'response') and exception.response.get('error') == 'ratelimited':
            # Get the retry-after time if available
            retry_after = exception.response.get('headers', {}).get('Retry-After')
            if retry_after:
                try:
                    delay = float(retry_after) + 0.5  # Add a small buffer
                except (ValueError, TypeError):
                    delay = max(delay, 5.0)  # Default to 5 seconds if parsing fails
            else:
                delay = max(delay, 5.0)
            
        logger.warning(f"Slack API request failed: {str(exception)}. "
                      f"Retrying in {delay:.2f}s (attempt {attempt+1}/{self.max_retries})")
        
        return delay
    
    def get_employees_data(self, real_name: str) -> List[Dict[str, Any]]:
        users_list = [
        {'display_name': '@Talha', 'real_name': 'Talha Ahmad', 'title': 'Operations Manager', 'team': 'OPS'},
        {'display_name': '@Val', 'real_name': 'Valentine Enedah', 'title': 'CX', 'team': 'PRO'},
        {'display_name': '@Ian Balina', 'real_name': 'Ian Balina', 'title': 'Founder and CEO', 'team': None},
        {'display_name': '@Harsh', 'real_name': 'Harsh', 'title': 'Senior Full Stack Engineer', 'team': 'ENG'},
        {'display_name': '@Andrew Tran', 'real_name': 'Andrew Tran', 'title': 'Data Engineer', 'team': 'AI'},
        {'display_name': '@Ayush Jalan', 'real_name': 'Ayush Jalan', 'title': 'Blockchain Engineer', 'team': 'ENG'},
        {'display_name': '@Drich', 'real_name': 'Raldrich Oracion', 'title': 'Customer Success', 'team': 'PRO'},
        {'display_name': '@Bartosz', 'real_name': 'Bartosz Kusnierczak', 'title': 'Passion never fail', 'team': 'ENG'},
        {'display_name': '@Jake', 'real_name': 'Jake Nguyen', 'title': 'Senior Data Engineer', 'team': 'AI'},
        {'display_name': '@Roshan Ganesh', 'real_name': 'Roshan Ganesh', 'title': 'Marketing Lead', 'team': 'MKT'},
        {'display_name': '@Sam Monac', 'real_name': 'Sam Monac', 'title': 'Chief Product Officer', 'team': None},
        {'display_name': '@Favour', 'real_name': 'Favour Ikwan', 'title': 'Chief Operations Officer', 'team': 'OPS'},
        {'display_name': '@Suleman Tariq', 'real_name': 'Suleman Tariq', 'title': 'Tech Lead', 'team': 'ENG'},
        {'display_name': '@Zaiying Li', 'real_name': 'Zaiying Li', 'title': '', 'team': 'OPS'},
        {'display_name': '@Hemank', 'real_name': 'Hemank', 'title': '', 'team': 'RES'},
        {'display_name': '@Ben', 'real_name': 'Ben Diagi', 'title': 'Product Manager', 'team': 'PRO'},
        {'display_name': '@Chao', 'real_name': 'Chao Li', 'title': 'Quantitative Analyst', 'team': 'AI'},
        {'display_name': '@Abdullah', 'real_name': 'Abdullah', 'title': 'Head Of Investment', 'team': 'RES'},
        {'display_name': '@Manav', 'real_name': 'Manav Garg', 'title': 'Blockchain Engineer', 'team': 'RES'},
        {'display_name': '@Vasilis', 'real_name': 'Vasilis Kotopoulos', 'title': 'AI Team Lead', 'team': 'AI'},
        {'display_name': '@Olaitan Akintunde', 'real_name': 'Olaitan Akintunde', 'title': 'Video Editor and Motion Designer', 'team': 'MKT'},
        {'display_name': '@Chetan Kale', 'real_name': 'Chetan Kale', 'title': '', 'team': 'RES'},
        {'display_name': '@ayo', 'real_name': 'ayo', 'title': '', 'team': 'PRO'},
        {'display_name': '@Özcan İlhan', 'real_name': 'Özcan İlhan', 'title': '', 'team': 'ENG'},
        {'display_name': '@Faith Oladejo', 'real_name': 'Faith Oladejo', 'title': '', 'team': 'PRO'},
        {'display_name': '@Taf', 'real_name': 'Tafcir Majumder', 'title': 'Head Of Business Development', 'team': 'MKT'},
        {'display_name': '@Caleb N', 'real_name': 'Caleb', 'title': '', 'team': 'MKT'},
        {'display_name': '@divine', 'real_name': 'Divine Anthony', 'title': 'Devops', 'team': 'ENG'},
        {'display_name': '@Williams', 'real_name': 'Williams Williams', 'title': 'Senior Fullstack Engineer', 'team': 'ENG'},
        {'display_name': '@Anki Truong', 'real_name': 'Truong An (Anki)', 'title': '', 'team': 'ENG'},
        {'display_name': '@Ryan', 'real_name': 'Ryan Barcelona', 'title': 'Freelancer', 'team': 'MKT'},
        {'display_name': '@Phát -', 'real_name': 'Phát -', 'title': '', 'team': 'OPS'},
        {'display_name': '@AhmedHamdy', 'real_name': 'AhmedHamdy', 'title': 'Senior Data Scientist/ML Engineer', 'team': 'AI'},
        {'display_name': '@Grady', 'real_name': 'Grady', 'title': 'Data Scientist/AI Engineer', 'team': 'AI'},
        {'display_name': '@Khadijah', 'real_name': 'Khadijah Shogbuyi', 'title': '', 'team': 'OPS'},
        {'display_name': '@Talha Cagri', 'real_name': 'Talha Cagri Kotcioglu', 'title': 'Quantitative Analyst', 'team': 'AI'},
        {'display_name': '@Agustín Gamoneda', 'real_name': 'Agustín Gamoneda', 'title': '', 'team': 'MKT'},
        {'display_name': '@Peterson', 'real_name': 'Peterson Nwoko', 'title': 'Sr DevOps/SRE Engineer', 'team': 'ENG'}
        ]
        if real_name:
            return [user for user in users_list if re.match(f"^{re.escape(real_name)}$", user['real_name'], re.IGNORECASE)]
        else:
            return users_list

    
    def extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from text.
        
        Args:
            text: Text to extract URLs from
            
        Returns:
            List of extracted URLs
        """
        if not text:
            return []
            
        # Regex pattern for URLs
        url_pattern = r'https?://[^\s<>"\']+'
        urls = re.findall(url_pattern, text)
        
        logger.info(f"Extracted {len(urls)} URLs")
        return urls
    
    def format_for_slack(self, text: str) -> str:
        """
        Format text to be properly displayed in Slack.
        
        Args:
            text: Text to format
            
        Returns:
            Formatted text for Slack
        """
        # Replace Markdown-style headers with bold text
        # Match # Header, ## Header, etc.
        header_pattern = r'(^|\n)#{1,6}\s+(.*?)(?=\n|$)'
        text = re.sub(header_pattern, r'\1*\2*', text)
        
        # Replace ** bold ** with * bold * (Slack format)
        bold_pattern = r'\*\*(.*?)\*\*'
        text = re.sub(bold_pattern, r'*\1*', text)
        
        # Replace __ italic __ with _ italic _ (Slack format)
        italic_pattern = r'__(.*?)__'
        text = re.sub(italic_pattern, r'_\1_', text)
        
        # Ensure code blocks use Slack format (```code``` -> ```code```)
        # This is a no-op but included for completeness
        
        # Replace HTML tags with their text content
        html_pattern = r'<(?!@|#|!|https?://)(.*?)>'
        text = re.sub(html_pattern, r'\1', text)
        
        return text
    
    async def search_channel_history(
        self,
        channel_id: str,
        username: Optional[str] = None,
        time_range: str = "days",
        time_value: int = 7,
        message_count: int = 50
    ) -> Dict[str, Any]:
        """
        Search channel history with specified parameters.
        Identifies different types of content: text, URLs, images, videos, code blocks, files.
        
        Args:
            channel_id: Slack channel ID
            username: Optional username to filter messages by
            time_range: Time unit to look back ("hours", "days", "weeks")
            time_value: Number of time units to look back (1-30)
            message_count: Maximum number of messages to retrieve (10-100)
            
        Returns:
            Dictionary with search results and metadata
        """
        # Validate parameters
        if time_range not in ["hours", "days", "weeks"]:
            time_range = "days"  # Default to days
            
        if time_value < 1:
            time_value = 1
        elif time_value > 30:
            time_value = 30  # Cap at 30 units
            
        if message_count < 10:
            message_count = 10
        elif message_count > 100:
            message_count = 100  # Cap at 100 messages
            
        # Create search parameters dictionary
        search_params = {
            "username": username,
            "time_range": time_range,
            "time_value": time_value,
            "message_count": message_count
        }
        
        logger.info(f"Searching channel history with parameters: {search_params}")
        
        # Define time range in seconds based on the parameters
        now = datetime.now().timestamp()
        
        if time_range == "hours":
            time_range_seconds = time_value * 3600
        elif time_range == "days":
            time_range_seconds = time_value * 86400
        elif time_range == "weeks":
            time_range_seconds = time_value * 604800
        else:
            time_range_seconds = 86400  # Default to 1 day
        
        # Calculate oldest timestamp to consider
        oldest_ts = str(now - time_range_seconds)
        
        try:
            # Prepare conversation history params
            history_params = {
                "channel": channel_id,
                "limit": min(100, message_count),
                "oldest": oldest_ts,
                "inclusive": True
            }
            
            
            # Fetch channel history
            result = self.client.conversations_history(**history_params)
            
            messages = result["messages"]
            
            # If we need more messages and there are more to fetch
            if message_count > 100 and result.get("has_more", False):
                # Continue fetching older messages
                cursor = result["response_metadata"]["next_cursor"]
                while len(messages) < message_count and cursor:
                    more_result = self.client.conversations_history(
                        channel=channel_id,
                        cursor=cursor,
                        limit=min(100, message_count - len(messages)),
                        inclusive=True
                    )
                    messages.extend(more_result["messages"])
                    if more_result.get("has_more", False):
                        cursor = more_result["response_metadata"]["next_cursor"]
                    else:
                        break
            
            # Apply username filter if specified
            if username:
                logger.info(f"Filtering messages by username: {username}")
                
                # Filter messages directly by user information
                filtered_messages = []
                
                # Get all users first to compare usernames
                try:
                    users_response = self.client.users_list()
                    users = users_response.get("members", [])
                    
                    # Find users that match the username filter
                    matching_user_ids = []
                    for user in users:
                        name = user.get("name", "").lower()
                        real_name = user.get("real_name", "").lower()
                        display_name = user.get("profile", {}).get("display_name", "").lower()
                        
                        # Check for matches in all name fields
                        if (username.lower() == name or 
                            username.lower() == real_name or
                            username.lower() in name or
                            username.lower() in real_name or
                            username.lower() in display_name):
                            matching_user_ids.append(user.get("id"))
                    
                    logger.info(f"Found {len(matching_user_ids)} users matching '{username}'")
                    
                    # Filter messages by those user IDs
                    for msg in messages:
                        user_id = msg.get("user", "")
                        if user_id in matching_user_ids:
                            filtered_messages.append(msg)
                    
                    messages = filtered_messages
                    logger.info(f"Found {len(messages)} messages from users matching '{username}'")
                
                except SlackApiError as e:
                    logger.error(f"Error filtering messages by username: {str(e)}")
                    # Fall back to basic filtering if user lookup fails
                    filtered_messages = []
                    for msg in messages:
                        if msg.get("user"):
                            filtered_messages.append(msg)
                    messages = filtered_messages
                    logger.warning(f"Fell back to basic filtering, found {len(messages)} messages")
            
            # Process messages to identify content types
            processed_messages = []
            for msg in messages:
                processed_msg = {
                    "ts": msg.get("ts", ""),
                    "user": msg.get("user", ""),
                    "text": msg.get("text", ""),
                    "content_types": [],
                    "urls": [],
                    "images": [],
                    "videos": [],
                    "files": [],
                    "code_blocks": [],
                    "reactions": msg.get("reactions", [])
                }
                
                # Check for basic text
                if msg.get("text"):
                    processed_msg["content_types"].append("text")
                
                # Extract URLs from text
                if msg.get("text"):
                    urls = self.extract_urls(msg.get("text", ""))
                    if urls:
                        processed_msg["content_types"].append("url")
                        processed_msg["urls"] = urls
                
                # Check for files
                if "files" in msg:
                    processed_msg["content_types"].append("file")
                    
                    for file in msg["files"]:
                        file_info = {
                            "id": file.get("id", ""),
                            "name": file.get("name", ""),
                            "title": file.get("title", ""),
                            "mimetype": file.get("mimetype", ""),
                            "filetype": file.get("filetype", ""),
                            "url": file.get("url_private", ""),
                            "thumb_url": file.get("thumb_url", ""),
                            "size": file.get("size", 0),
                            "created": file.get("created", 0)
                        }
                        
                        # Add file to the processed message files list
                        processed_msg["files"].append(file_info)
                        
                        # Determine the content type based on mimetype
                        mimetype = file.get("mimetype", "")
                        if mimetype.startswith("image/"):
                            if "image" not in processed_msg["content_types"]:
                                processed_msg["content_types"].append("image")
                            processed_msg["images"].append(file_info)
                        elif mimetype.startswith("video/"):
                            if "video" not in processed_msg["content_types"]:
                                processed_msg["content_types"].append("video")
                            processed_msg["videos"].append(file_info)
                        elif mimetype.startswith("text/") or "code" in file.get("filetype", ""):
                            if "code" not in processed_msg["content_types"]:
                                processed_msg["content_types"].append("code")
                
                # Check for code blocks in text
                if msg.get("text") and "```" in msg.get("text", ""):
                    processed_msg["content_types"].append("code")
                    
                    # Extract code blocks
                    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', msg.get("text", ""), re.DOTALL)
                    if code_blocks:
                        processed_msg["code_blocks"] = code_blocks
                
                # Check for attachments
                if "attachments" in msg:
                    for attachment in msg["attachments"]:
                        # Check for image attachments
                        if attachment.get("image_url"):
                            if "image" not in processed_msg["content_types"]:
                                processed_msg["content_types"].append("image")
                            processed_msg["images"].append({
                                "url": attachment.get("image_url", ""),
                                "thumb_url": attachment.get("thumb_url", ""),
                                "title": attachment.get("title", "Attachment")
                            })
                
                # Check for thread information
                if "thread_ts" in msg:
                    processed_msg["thread_ts"] = msg["thread_ts"]
                    processed_msg["content_types"].append("thread")
                
                # Add the processed message to our list
                processed_messages.append(processed_msg)
            
            # Log content type breakdown
            content_type_counts = {
                "text": sum(1 for m in processed_messages if "text" in m["content_types"]),
                "url": sum(1 for m in processed_messages if "url" in m["content_types"]),
                "image": sum(1 for m in processed_messages if "image" in m["content_types"]),
                "video": sum(1 for m in processed_messages if "video" in m["content_types"]),
                "code": sum(1 for m in processed_messages if "code" in m["content_types"])
            }
            
            logger.info(f"Found {len(processed_messages)} messages with content breakdown: {content_type_counts}")
            
            # Return search parameters along with results
            return {
                "messages": processed_messages,
                "search_params": search_params,
                "count": len(processed_messages)
            }
            
        except SlackApiError as e:
            logger.error(f"Error searching channel history: {e.response['error']}")
            return {
                "messages": [],
                "search_params": search_params,
                "count": 0,
                "error": str(e.response['error'])
            }

