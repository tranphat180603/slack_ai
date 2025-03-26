import uvicorn
import requests
import fastapi
from fastapi import Request, Form, BackgroundTasks, Response
from bs4 import BeautifulSoup
import re
import os
import logging
import datetime
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import tweepy
from fastapi import HTTPException
from dotenv import load_dotenv
import openai
import json
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import base64
import time
import aiohttp
import asyncio
from urllib.parse import urlparse
from github import Github
from linear_client import LinearClient
import sys
import codecs
import yaml
import dataclasses
from collections import defaultdict
from dotenv import load_dotenv
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("slack_ai_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("slack_ai_bot")

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded")

# Configuration variables - loaded from environment variables
OUTPUT_SERVER_URL = os.environ.get("OUTPUT_SERVER_URL", "http://localhost:8080/receive")
TWITTER_API_KEY = os.environ.get("X_API_KEY", "")
TWITTER_API_SECRET = os.environ.get("X_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.environ.get("X_ACCESS_TOKEN", "")
TWITTER_ACCESS_SECRET = os.environ.get("X_ACCESS_TOKEN_SECRET", "")
TWITTER_BEARER_TOKEN = os.environ.get("X_BEARER_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TMAI_ACCOUNT_ID = os.environ.get("TMAI_ACCOUNT_ID", "")
LUCKY_ACCOUNT_ID = os.environ.get("LUCKY_ACCOUNT_ID", "")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")

# Additional environment variables for new integrations
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
LINEAR_API_KEY = os.environ.get("LINEAR_API_KEY", "")
AI_RATE_LIMIT = int(os.environ.get("AI_RATE_LIMIT", "30"))  # Requests per minute
AI_MODEL = os.environ.get("AI_MODEL", "o3-mini")

# Initialize Slack client
slack_client = WebClient(token=SLACK_BOT_TOKEN)
logger.info("Slack client initialized")

# Initialize Slack client with user token for thread history access
SLACK_USER_TOKEN = os.environ.get("SLACK_USER_TOKEN", "")
if SLACK_USER_TOKEN:
    slack_user_client = WebClient(token=SLACK_USER_TOKEN)
    logger.info("Slack user client initialized")
else:
    slack_user_client = None
    logger.warning("No Slack user token provided - thread history access may be limited")

# Initialize additional clients
github_client = None
if GITHUB_TOKEN:
    github_client = Github(GITHUB_TOKEN)
    logger.info("GitHub client initialized")

linear_client = None
if LINEAR_API_KEY:
    try:
        linear_client = LinearClient(LINEAR_API_KEY)
        logger.info("Linear client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Linear client: {str(e)}")

# Load prompts from YAML file
try:
    with open('prompts.yaml', 'r', encoding='utf-8') as file:
        PROMPTS = yaml.safe_load(file)
    logger.info("Successfully loaded prompts from prompts.yaml")
except Exception as e:
    logger.error(f"Error loading prompts: {str(e)}")
    # Fallback to empty dict
    PROMPTS = {}

# Rate limiting tracker
rate_limit_tracker = {
    "last_reset": time.time(),
    "requests": 0
}

# Conversation memory store - keys are channel_id+thread_ts, values are lists of messages
conversation_history = defaultdict(list)
# Maximum number of messages to keep in conversation history
MAX_HISTORY_MESSAGES = 30
# Maximum age of conversation history (in hours)
CONVERSATION_EXPIRY = 24  # hours

app = fastapi.FastAPI()

class ProcessedContent(BaseModel):
    url: str
    text_content: str
    is_tweet: bool
    tweet_id: Optional[str] = None
    user_id: Optional[str] = None
    thread_tweets: Optional[List[dict]] = None

class AIRequest(BaseModel):
    """Model for AI request data received from Slack."""
    text: str  # The text of the request
    user_id: str  # Slack user ID
    channel_id: str  # Slack channel ID
    files: Optional[List[Dict[str, Any]]] = None  # Attached files
    urls: Optional[List[str]] = None  # URLs extracted from text
    message_ts: Optional[str] = None  # Message timestamp for reference

@dataclasses.dataclass
class ContentAnalysisResult:
    """Result of content analysis, with categorization and extracted entities."""
    content_type: str = ""  # Main content type: simple_prompt or prompt_requires_tool_use
    requires_slack_channel_history: bool = False  # Whether we need channel history
    urls: List[str] = None  # URLs mentioned in the query
    perform_RAG_on_linear_issues: bool = False  # Whether to perform RAG on Linear issues
    create_linear_issue: bool = False  # Whether to create a new Linear issue
    text: str = ""  # Original query text

    def __post_init__(self):
        if self.urls is None:
            self.urls = []

def format_for_slack(text: str) -> str:
    """
    Convert standard markdown formatting to Slack-compatible mrkdwn formatting.
    
    Args:
        text: Text with potential markdown formatting
        
    Returns:
        Text with Slack-compatible formatting
    """
    if not text:
        return text
        
    # Replace double asterisks with single (for bold)
    text = re.sub(r'\*\*([^*]+)\*\*', r'*\1*', text)
    
    # Replace double underscores with single (for italic)
    text = re.sub(r'__([^_]+)__', r'_\1_', text)
    
    # Replace markdown headers with bold text
    text = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
    
    # Replace triple backticks with single backticks for inline code
    text = re.sub(r'```([^`]+)```', r'`\1`', text)
    
    # Fix numbered lists (ensure there's a space after the period)
    text = re.sub(r'^(\d+)\.([^\s])', r'\1. \2', text, flags=re.MULTILINE)
    
    # Fix bullet points (ensure there's a space after the asterisk)
    text = re.sub(r'^\*([^\s])', r'* \1', text, flags=re.MULTILINE)
    
    # Remove language specifiers from code blocks
    text = re.sub(r'```[a-zA-Z0-9]+\n', r'```\n', text)
    
    return text

@app.post("/slack/ai_command")
def ai_command(request: Request):
    return {"message": "AI command received"}
#holding for now

@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    """
    Handle Slack events, particularly app_mention events when the bot is tagged.
    This endpoint processes mentions and responds similarly to the slash command.
    """
    # Get the raw request body first
    body = await request.body()
    text_body = body.decode('utf-8')
    logger.info(f"Received raw body: {text_body[:200]}...")  # Log first 200 chars
    
    try:
        # Parse the JSON manually
        payload = json.loads(text_body)
        logger.info(f"Parsed payload type: {payload.get('type', 'unknown')}")
        
        # Handle URL verification challenge by returning the raw challenge value
        if payload.get("type") == "url_verification":
            challenge = payload.get("challenge")
            logger.info(f"Verification challenge received: {challenge}")
            
            # Return the raw challenge value directly as plain text
            return Response(content=challenge, media_type="text/plain")
        
        # Normal event processing continues...
        if payload.get("type") == "event_callback":
            event = payload.get("event", {})
            event_type = event.get("type")
            
            # Handle bot mentions
            if event_type == "app_mention":
                logger.info(f"Bot was mentioned in channel {event.get('channel')}")
                
                # Extract user ID, channel ID, and text
                user_id = event.get("user", "")
                channel_id = event.get("channel", "")
                text = event.get("text", "")
                message_ts = event.get("ts", "")
                
                # Determine if this is a thread reply and get the thread_ts
                # If thread_ts exists, this is a reply in a thread
                # If not, this is a new thread where the current ts becomes the thread_ts
                thread_ts = event.get("thread_ts", message_ts)
                logger.info(f"Message ts: {message_ts}, Thread ts: {thread_ts}")
                
                # Remove the bot mention from the text (matches <@BOTID>)
                text = text.replace(f"<@U08GNQ8F2RH>", "")
                
                if not text:
                    # If there's no text after removing the mention, respond with a help message
                    try:
                        slack_client.chat_postMessage(
                            channel=channel_id,
                            text="Hello! I'm your AI assistant. How can I help you?",
                            thread_ts=thread_ts
                        )
                    except SlackApiError as e:
                        logger.error(f"Error sending help message: {e.response['error']}")
                    return {}
                
                # Create AI request object
                ai_request = AIRequest(
                    text=text,
                    user_id=user_id,
                    channel_id=channel_id,
                    urls=extract_urls(text),
                    message_ts=message_ts
                )
                
                # Send initial "processing" message
                try:
                    initial_response = slack_client.chat_postMessage(
                        channel=channel_id,
                        text="AI is processing...",
                        thread_ts=thread_ts,
                        blocks=[
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"*Input:*\n```\n{text[:50]}{('...' if len(text) > 50 else '')}\n```\n*Status:*\n```\n➤ Analyzing your query...\n```"
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
                                        "text": "_Please wait while I generate a response..._"
                                    }
                                ]
                            }
                        ]
                    )
                    logger.info(f"Posted initial message with ts: {initial_response.get('ts')}")
                    
                    # Update the AI request with the message timestamp for reference
                    ai_request.message_ts = initial_response.get("ts")
                    
                except SlackApiError as e:
                    logger.error(f"Error posting initial message: {e.response['error']}")
                
                # Start processing in the background
                background_tasks.add_task(process_mention_request, ai_request, thread_ts)
                
        # Return an empty 200 response to acknowledge receipt
        return {}
        
    except Exception as e:
        logger.error(f"Error processing Slack event: {str(e)}")
        return {"status": "error", "message": str(e)}

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    if not text:
        return []
        
    # Debug
    logger.info(f"Extracting URLs from: {text[:100]}...")
    
    # Regex pattern for URLs
    url_pattern = r'https?://[^\s<>"\']+'
    urls = re.findall(url_pattern, text)
    
    logger.info(f"Extracted {len(urls)} URLs: {urls}")
    return urls

async def fetch_url_content(url: str) -> Dict[str, Any]:
    """
    Unified function to fetch content from a URL, determining the appropriate method based on URL type.
    Tries GitHub, Twitter/X, and general web content in sequence.
    
    Args:
        url: The URL to fetch content from
        
    Returns:
        Dictionary with the content and metadata
    """
    logger.info(f"Fetching content from URL: {url}")
    
    # Initialize result structure
    result = {
        "url": url,
        "content_type": "unknown",
        "text": "",
        "title": "",
        "metadata": {},
        "error": None
    }
    
    try:
        # Check for GitHub repository URL
        github_pattern = r'github\.com/([^/\s]+/[^/\s]+)'
        github_match = re.search(github_pattern, url)
        
        if github_match:
            # Extract repo owner and name, removing any trailing content
            repo_path = github_match.group(1)
            repo_path = re.sub(r'/(blob|tree|pull|issues).*$', '', repo_path)
            
            result["content_type"] = "github"
            result["metadata"]["repo_path"] = repo_path
            
            # Try to fetch GitHub repo content
            if github_client:
                try:
                    logger.info(f"Fetching GitHub repository: {repo_path}")
                    repo = github_client.get_repo(repo_path)
        
                    # Get basic repo information
                    result["title"] = repo.full_name
                    result["text"] = repo.description or "No description available"
                    result["metadata"].update({
                        "stars": repo.stargazers_count,
                        "forks": repo.forks_count,
                        "language": repo.language,
                        "owner": repo.owner.login,
                        "created_at": repo.created_at.isoformat() if repo.created_at else None,
                        "updated_at": repo.updated_at.isoformat() if repo.updated_at else None
                    })
        
                    # Try to get README content
                    try:
                        readme = repo.get_readme()
                        readme_content = base64.b64decode(readme.content).decode('utf-8')
                        result["text"] += f"\n\nREADME:\n{readme_content}"
                    except Exception as readme_error:
                        logger.warning(f"Could not fetch README for {repo_path}: {str(readme_error)}")
        
                    return result
                except Exception as e:
                    logger.error(f"Error fetching GitHub repository: {str(e)}")
                    result["error"] = f"GitHub API error: {str(e)}"
            else:
                result["error"] = "GitHub client not initialized"
        
        # Check for Twitter/X URL
        twitter_pattern = r'(https?://(twitter\.com|x\.com)/[^\s]+)'
        twitter_match = re.search(twitter_pattern, url)
        
        if twitter_match:
            result["content_type"] = "twitter"
            
            # Try to fetch Twitter content
            if TWITTER_API_KEY and TWITTER_API_SECRET and TWITTER_BEARER_TOKEN:
                try:
                    # This will use your existing Twitter processing function
                    processed_content = process_tweet_url(url)
                    
                    if processed_content:
                        result["text"] = processed_content.text_content
                        result["title"] = f"Tweet from @{processed_content.user_id}" if processed_content.user_id else "Tweet"
                        result["metadata"].update({
                            "tweet_id": processed_content.tweet_id,
                            "user_id": processed_content.user_id,
                            "is_thread": processed_content.thread_tweets is not None,
                            "thread_tweets": processed_content.thread_tweets
                        })
                        
                        return result
                except Exception as e:
                    logger.error(f"Error fetching Twitter content: {str(e)}")
                    result["error"] = f"Twitter API error: {str(e)}"
                    # Continue to try as a regular URL
            else:
                result["error"] = "Twitter API credentials not configured"
    
        # If we're here, try to fetch as a general web URL
        result["content_type"] = "web"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Get the page title
                    title_tag = soup.find("title")
                    if title_tag:
                        result["title"] = title_tag.get_text(strip=True)
                    
                    # Remove scripts, styles, etc.
                    for element in soup(["script", "style", "footer", "nav"]):
                        element.extract()
                    
                    # Get meta description if available
                    meta_desc = soup.find("meta", attrs={"name": "description"})
                    if meta_desc:
                        result["metadata"]["description"] = meta_desc.get("content", "")
                    
                    # Try to get main content
                    main_content = soup.find("main") or soup.find("article") or soup.find("div", class_="content")
                    
                    if main_content:
                        # Get text from main content
                        content_text = main_content.get_text(separator=' ', strip=True)
                        content_text = re.sub(r'\s+', ' ', content_text)
                        result["text"] = content_text[:5000]  # Limit to 5000 chars
                    else:
                        # Get text from body if main content not found
                        body_text = soup.get_text(separator=' ', strip=True)
                        body_text = re.sub(r'\s+', ' ', body_text)
                        result["text"] = body_text[:5000]  # Limit to 5000 chars
                else:
                    result["error"] = f"HTTP error: {response.status}"
                    
        return result
                    
    except Exception as e:
        logger.error(f"Error in process_mention_request: {str(e)}")
        # Send error as a new message in the thread
        try:
            slack_client.chat_postMessage(
                channel=ai_request.channel_id,
                thread_ts=ai_request.message_ts,
                text=f"Sorry, I encountered an error: {str(e)}"
            )
        except Exception as post_error:
            logger.error(f"Error sending error message: {str(post_error)}")


def parse_user_mentions(text):
    """
    Convert Slack user mentions (<@U123ABC>) to their actual display names.
    Returns both the original text and a display-name version.
    """
    # Find all user mentions (<@U123ABC>)
    mention_pattern = r'<@([A-Z0-9]+)>'
    mentions = re.findall(mention_pattern, text)
    
    # Replace each mention with actual username
    for user_id in mentions:
        try:
            user_info = slack_client.users_info(user=user_id)
            if user_info["ok"]:
                # Get user information in order of preference
                display_name = (
                    user_info["user"]["profile"].get("display_name") or
                    f"<@{user_id}>"  # Fallback to original mention if no name found
                )
                
                # Also get title if available
                title = user_info["user"]["profile"].get("title", "")
                
                # Replace the mention with the display name
                text = text.replace(f"<@{user_id}>", display_name)
                
                # Log the successful conversion
                logger.info(f"Converted user mention {user_id} to {display_name}")
                
        except SlackApiError as e:
            logger.error(f"Error getting user info for {user_id}: {str(e)}")
            # Keep the original mention format if we can't get user info
            continue
    
    return text

async def process_mention_request(ai_request: AIRequest, thread_ts: str):
    try:
        logger.info(f"Processing mention request from user {ai_request.user_id}: {ai_request.text[:50]}...")

        # Check rate limits
        if not check_rate_limit():
            try:
                slack_client.chat_postMessage(
                    channel=ai_request.channel_id,
                    thread_ts=thread_ts,
                    text="Rate limit exceeded. Please try again in a minute."
                )
            except Exception as e:
                logger.error(f"Error sending rate limit message: {str(e)}")
            return
        
        # Use the thread_ts parameter as our root thread identifier
        # This is passed from the slack_events function and is the thread's root ts
        root_thread_ts = thread_ts
        
        # Extract thread_ts from the message text if available
        # Sometimes messages contain thread_ts in their metadata
        if hasattr(ai_request, 'metadata') and ai_request.metadata and 'thread_ts' in ai_request.metadata:
            root_thread_ts = ai_request.metadata['thread_ts']
            logger.info(f"Found thread_ts in message metadata: {root_thread_ts}")
        
        # Define conversation key using the root thread_ts
        conversation_key = f"{ai_request.channel_id}:{root_thread_ts}"
        logger.info(f"Using conversation key: {conversation_key}")
        
        # Clean expired conversations periodically
        clean_expired_conversations()
        
        # Parse user mentions if exists
        ai_request.text = parse_user_mentions(ai_request.text)
        
        # Initialize context parts for the final response
        context_parts = []

        #initialize history context
        history_context = []

        #init sender's name
        sender_name = slack_client.users_info(user=ai_request.user_id)
        if sender_name["ok"]:
            sender_name = sender_name["user"]["profile"]["display_name"]
        else:
            sender_name = f"<@{ai_request.user_id}>"
        
        # Add current query
        context_parts.append(f"Current {sender_name} query: {ai_request.text}\n")
        
        # Add conversation history if it exists
        if conversation_key in conversation_history and conversation_history[conversation_key]:
            history_context = ["**Here is the conversation history between you and " + sender_name + " so far:**"]
            for i, msg in enumerate(conversation_history[conversation_key]):
                # Add message number for better context
                if msg["role"] == "user":
                    history_context.append(f"**{sender_name} (#{i+1} turn):** {msg['content']}")
                else:
                    # Truncate assistant responses in history to keep context manageable
                    content = msg["content"]
                    if len(content) > 300:
                        content = content[:300] + "... (content truncated)"
                    history_context.append(f"**Assistant (#{i+1} turn):** {content}")
            
            context_parts.append("\n".join(history_context))
            context_parts.append("")  # Empty line after history
        else:
            history_context = ["**Conversation History:** This is the first message in the conversation between you and " + sender_name + "."]
        
        # Step 1: Analyze the content to determine what we need to do
        try:
            content_analysis = await analyze_content(ai_request.text, "\n".join(history_context), sender_name=sender_name)
            logger.info(f"Content analysis: {content_analysis}")
        except Exception as e:
            error_msg = f"Error analyzing content: {str(e)}"
            logger.error(error_msg)
            context_parts.append(f"\n❌ {error_msg}")
            # Set default content analysis
            content_analysis = ContentAnalysisResult()
            content_analysis.content_type = "simple_prompt"
        
        # Try to rebuild conversation history from thread if needed
        if root_thread_ts and (conversation_key not in conversation_history or not conversation_history[conversation_key]):
            try:
                # Use slack_user_client to get thread history if available
                if slack_user_client:
                    # Log the thread_ts we're using to fetch replies
                    logger.info(f"Fetching thread messages using root ts: {root_thread_ts}")
                    thread_info = slack_user_client.conversations_replies(
                        channel=ai_request.channel_id,
                        ts=root_thread_ts,  # This is the root message of the thread
                        limit=MAX_HISTORY_MESSAGES  # Reasonable limit
                    )
                    # Safely log the response data, not the SlackResponse object itself
                    if thread_info and thread_info.get("ok"):
                        logger.info(f"Raw conversations_replies response: messages count: {len(thread_info.get('messages', []))}, has_more: {thread_info.get('has_more', False)}")
                    else:
                        logger.info(f"Raw conversations_replies response error: {thread_info.get('error', 'unknown error')}")
                else:
                    # Fallback to bot client (which may not work for channels)
                    logger.info(f"Using bot client to fetch thread with ts: {root_thread_ts}")
                    thread_info = slack_client.conversations_replies(
                        channel=ai_request.channel_id,
                        ts=root_thread_ts,
                        limit=MAX_HISTORY_MESSAGES
                    )
                    # Safely log the response data, not the SlackResponse object itself
                    if thread_info and thread_info.get("ok"):
                        logger.info(f"Raw conversations_replies response (bot client): messages count: {len(thread_info.get('messages', []))}, has_more: {thread_info.get('has_more', False)}")
                    else:
                        logger.info(f"Raw conversations_replies response error (bot client): {thread_info.get('error', 'unknown error')}")
                
                if thread_info and "messages" in thread_info:
                    logger.info(f"Found {len(thread_info['messages'])} messages in thread")
                    
                    # Initialize or clear the conversation history for this thread
                    conversation_history[conversation_key] = []
                    
                    # Process all messages in the thread in chronological order
                    for msg in thread_info["messages"]:
                        # Skip the current message being processed
                        if msg.get('ts') == ai_request.message_ts:
                            logger.info(f"Skipping current message being processed: {msg.get('text', '')[:30]}...")
                            continue
                        
                        # Skip processing messages
                        if "AI is processing" in msg.get("text", "") or "Generating response" in msg.get("text", "") or "⌛ Generating response" in msg.get("text", ""):
                            logger.info(f"Skipping processing message: {msg.get('text', '')[:30]}...")
                            continue
                        
                        # Determine if this is a bot message
                        is_bot = msg.get("bot_id") is not None
                        
                        # Add to conversation history with appropriate role
                        conversation_history[conversation_key].append({
                            "role": "assistant" if is_bot else "user",
                            "content": msg.get("text", ""),
                            "timestamp": float(msg.get("ts", time.time()))
                        })
                        # logger.info(f"Added {'assistant' if is_bot else 'user'} message to history: {msg.get('text', '')[:30]}...")
                    
                    logger.info(f"Rebuilt conversation history with {len(conversation_history[conversation_key])} messages")
                else:
                    logger.info(f"No messages found in thread. Response: {thread_info}")
            except Exception as e:
                logger.error(f"Error getting thread info: {str(e)}")
                # Print full traceback for debugging
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Step 2: Process the request based on content analysis
        
        # If we need to search Slack channel history
        if content_analysis.requires_slack_channel_history:
            try:
                # Update message to show slack search stage
                try:
                    slack_client.chat_update(
                        channel=ai_request.channel_id,
                        text="",
                        ts=ai_request.message_ts,
                        blocks=[
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"*Input:*\n```\n{ai_request.text[:50]}{('...' if len(ai_request.text) > 50 else '')}\n```\n*Status:*\n```\n✓ Query analyzed\n➤ Searching Slack history...\n```"
                                },
                                "accessory": {
                                    "type": "image",
                                    "image_url": "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExanRpdDc0enVuOXc3dG9vYWEzOGUyajFkOG03OHB6aTM4aTZhd2kycSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/IUNycHoVqvLDowiiam/giphy.gif",
                                    "alt_text": "Processing"
                                }
                            }
                        ]
                    )
                except SlackApiError as e:
                    error_msg = f"Could not update processing message: {e.response.get('error', '')}"
                    logger.warning(error_msg)
                    context_parts.append(f"\n⚠️ {error_msg}")
                
                logger.info("Searching Slack channel history")
                
                # Search channel history with AI-enhanced parameters
                search_results = await search_channel_history(ai_request.channel_id, search_params, ai_request.text, history_context=history_context)
                
                if search_results.get("error"):
                    error_msg = f"Error searching Slack history: {search_results['error']}"
                    logger.error(error_msg)
                    context_parts.append(f"\n❌ {error_msg}")
                elif search_results["count"] > 0:
                    # Include search parameters used in the context
                    ai_search_params = search_results.get("search_params", {})
                    search_context = []
                    
                    if ai_search_params.get("username"):
                        username = ai_search_params['username']
                        search_context.append(f"User: @{username}")
                    
                    time_range = f"{ai_search_params.get('time_value', 7)} {ai_search_params.get('time_range', 'days')}"
                    search_context.append(f"Time range: {time_range}")
                    search_context.append(f"Message limit: {ai_search_params.get('message_count', 50)}")
                    
                    context_parts.append(f"\nFound {search_results['count']} messages in channel history.")
                    
                    # Format relevant messages for context with user information
                    relevant_messages = []
                    for msg in search_results['messages']:
                        try:
                            # Try to get user information for better display
                            user_id = msg.get('user', '')
                            try:
                                user_info = slack_client.users_info(user=user_id)
                                user_name = user_info['user'].get('real_name', user_id)
                            except:
                                user_name = f"<@{user_id}>"
                                
                            relevant_messages.append(f"Message from {user_name}: {msg['text']}")
                            
                            # Include information about other content types
                            content_types = msg.get("content_types", [])
                            if "image" in content_types:
                                relevant_messages.append(f"- Contains {len(msg.get('images', []))} images")
                            if "url" in content_types:
                                relevant_messages.append(f"- Contains URLs: {', '.join(msg.get('urls', [])[:3])}")
                            if "code" in content_types:
                                relevant_messages.append(f"- Contains code blocks")
                        except Exception as msg_error:
                            error_msg = f"Error processing message: {str(msg_error)}"
                            logger.warning(error_msg)
                            relevant_messages.append(f"⚠️ {error_msg}")
                    
                    context_parts.append("Relevant channel messages:\n" + "\n".join(relevant_messages))
                else:
                    context_parts.append("\nNo relevant messages found in channel history.")
                    if "search_params" in search_results:
                        time_range = f"{search_results.get('search_params', {}).get('time_value', 7)} {search_results.get('search_params', {}).get('time_range', 'days')}"
                        context_parts.append(f"Searched the last {time_range}.")
            except Exception as e:
                error_msg = f"Error searching Slack channel history: {str(e)}"
                logger.error(error_msg)
                context_parts.append(f"\n❌ {error_msg}")
        
        # If we need to perform RAG on Linear issues
        if content_analysis.perform_RAG_on_linear_issues:
            try:
                # Update message to show linear search stage
                try:
                    linear_stage_text = "*Status:*\n```\n✓ Query analyzed\n"
                    if content_analysis.requires_slack_channel_history:
                        linear_stage_text += "✓ Slack history searched\n"
                    linear_stage_text += "➤ Searching Linear issues...\n```"
                    
                    slack_client.chat_update(
                        channel=ai_request.channel_id,
                        text="",
                        ts=ai_request.message_ts,
                        blocks=[
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"*Input:*\n```\n{ai_request.text[:50]}{('...' if len(ai_request.text) > 50 else '')}\n```\n{linear_stage_text}"
                                },
                                "accessory": {
                                    "type": "image",
                                    "image_url": "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExanRpdDc0enVuOXc3dG9vYWEzOGUyajFkOG03OHB6aTM4aTZhd2kycSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/IUNycHoVqvLDowiiam/giphy.gif",
                                    "alt_text": "Processing"
                                }
                            }
                        ]
                    )
                except SlackApiError as e:
                    error_msg = f"Could not update processing message: {e.response.get('error', '')}"
                    logger.warning(error_msg)
                    context_parts.append(f"\n⚠️ {error_msg}")
                
                logger.info("Performing RAG search on Linear issues")
                
                # Call with the user's query and a reasonable limit
                linear_results = await perform_linear_rag_search(ai_request.text, limit=15, history_context=history_context, sender_name=sender_name)
                
                # Format and add Linear results to context
                linear_context = format_linear_search_results(linear_results)
                context_parts.extend(linear_context)
                
            except Exception as e:
                error_msg = f"Error performing Linear RAG search: {str(e)}"
                logger.error(error_msg)
                context_parts.append(f"\n❌ {error_msg}")
        
        # Process URLs if present
        if content_analysis.urls:
            logger.info(f"Processing {len(content_analysis.urls)} URLs")
            
            url_results = []
            url_errors = []
            for url in content_analysis.urls:
                try:
                    result = await fetch_url_content(url)
                    url_results.append(result)
                    if result.get("error"):
                        url_errors.append(f"Error processing {url}: {result['error']}")
                except Exception as e:
                    error_msg = f"Error processing {url}: {str(e)}"
                    logger.error(error_msg)
                    url_errors.append(error_msg)
            
            if url_results:
                context_parts.append("\nURL content:")
                
                # Add any errors first
                if url_errors:
                    context_parts.append("\n❌ URL Processing Errors:")
                    for error in url_errors:
                        context_parts.append(f"- {error}")
                    context_parts.append("")  # Empty line after errors

                for result in url_results:
                    context_parts.append(f"\nURL: {result['url']}")
                    context_parts.append(f"Type: {result['content_type']}")
                    
                    if result.get("error"):
                        context_parts.append(f"Error: {result['error']}")
                    elif result.get("title"):
                        context_parts.append(f"Title: {result['title']}")
                        
                        # Add metadata based on content type
                        if result["content_type"] == "github":
                            if "metadata" in result and "stars" in result["metadata"]:
                                context_parts.append(f"Stars: {result['metadata']['stars']}")
                                context_parts.append(f"Language: {result['metadata'].get('language', 'Unknown')}")
                        
                        # Add text content (limited to keep context reasonable)
                        if result.get("text"):
                            text = result["text"]
                            if len(text) > 1000:
                                text = text[:1000] + "... (content truncated)"
                            context_parts.append(f"Content: {text}")

        # if content_analysis.create_linear_issue == False:
        #     content_analysis.create_linear_issue = True
        if content_analysis.create_linear_issue == False:
            # Check if the query contains keywords that suggest creating a Linear issue
            issue_creation_keywords = [
                "create issue", "create a issue", "create an issue", 
                "make issue", "make a issue", "make an issue",
                "new issue", "add issue", "add a issue", "add an issue",
                "create task", "create a task", "new task"
            ]
            
            # Convert to lowercase for case-insensitive matching
            text_lower = content_analysis.text.lower()
            
            # Check if any of the keywords are in the text
            if any(keyword in text_lower for keyword in issue_creation_keywords):
                content_analysis.create_linear_issue = True
                print(f"Manually overriding create_linear_issue to True based on keyword detection")

        if content_analysis.create_linear_issue:
            try:
                create_issue_text = "*Status:*\n```\n✓ Query analyzed\n"
                create_issue_text += "➤Crafting your Linear issue...\n```"
                try:
                    slack_client.chat_update(
                        channel=ai_request.channel_id,
                        text="",
                        ts=ai_request.message_ts,
                        blocks=[
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"*Input:*\n```\n{ai_request.text[:50]}{('...' if len(ai_request.text) > 50 else '')}\n```\n{create_issue_text}"
                                },
                                "accessory": {
                                    "type": "image",
                                    "image_url": "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExanRpdDc0enVuOXc3dG9vYWEzOGUyajFkOG03OHB6aTM4aTZhd2kycSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/IUNycHoVqvLDowiiam/giphy.gif",
                                    "alt_text": "Processing"
                                }
                            }
                        ]
                    )
                except SlackApiError as e:
                    error_msg = f"Could not update processing message: {e.response.get('error', '')}"
                    logger.warning(error_msg)
                    context_parts.append(f"\n⚠️ {error_msg}")
                    
                # Generate issue parameters using AI
                result = await generate_linear_issue(ai_request.text, history_context, sender_name)
                
                if result is None:
                    error_msg = "Failed to generate Linear issue - no response from API"
                    logger.error(error_msg)
                    context_parts.append(f"\n❌ {error_msg}")
                elif isinstance(result, str) and "Error message:" in result:
                    error_msg = result.replace("Error message: ", "")
                    logger.error(f"Linear issue creation error: {error_msg}")
                    context_parts.append(f"\n❌ Error creating Linear issue: {error_msg}")
                elif result.get("success"):
                    issue_info = result.get("issue", {})
                    success_msg = [
                        f"\n✅ Successfully created Linear issue:",
                        f"Title: {issue_info.get('title', 'No title')}",
                        f"URL: {issue_info.get('url', 'No URL')}",
                    ]
                    if issue_info.get('assignee'):
                        success_msg.append(f"Assigned to: {issue_info['assignee']}")
                    if issue_info.get('team'):
                        success_msg.append(f"Team: {issue_info['team']}")
                    if issue_info.get('priority'):
                        success_msg.append(f"Priority: {issue_info['priority']}")
                    
                    context_parts.extend(success_msg)
                else:
                    error_msg = result.get('error', 'Unknown error occurred')
                    logger.error(f"Linear issue creation failed: {error_msg}")
                    context_parts.append(f"\n❌ Error creating Linear issue: {error_msg}")
                
            except Exception as e:
                error_msg = f"Error in Linear issue creation process: {str(e)}"
                logger.error(error_msg)
                context_parts.append(f"\n❌ {error_msg}")
                result = {"success": False, "error": str(e)}  # Set default result for error case
        
        # Update message to show final response generation stage
        try:
            response_stage_text = "*Status:*\n```\n✓ Query analyzed\n"
            if content_analysis.requires_slack_channel_history:
                response_stage_text += "✓ Slack history searched\n"
            if content_analysis.perform_RAG_on_linear_issues:
                response_stage_text += "✓ Linear issues searched\n"
            if content_analysis.create_linear_issue:
                if result.get("success"):
                    response_stage_text += f"✓ Linear issue created\n"
                    response_stage_text += f"Title: {result['issue']['title']}\n"
                    response_stage_text += f"URL: {result['issue']['url']}\n"
                    if result['issue'].get('assignee'):
                        response_stage_text += f"Assigned to: {result['issue']['assignee']}\n"
                else:
                    response_stage_text += f"❌ Error creating Linear issue: {result.get('error', 'Unknown error')}\n"
            response_stage_text += "➤ Generating response...\n```"
            
            slack_client.chat_update(
                channel=ai_request.channel_id,
                text="",
                ts=ai_request.message_ts,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Input:*\n```\n{ai_request.text[:50]}{('...' if len(ai_request.text) > 50 else '')}\n```\n{response_stage_text}"
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
                                "text": "_Please wait while I generate a response..._"
                            }
                        ]
                    }
                ]
            )
        except SlackApiError as e:
            logger.warning(f"Could not update processing message: {e.response.get('error', '')}")
        
        # Step 3: Call AI with the full context including conversation history
        logger.info("Building final context for AI call with conversation history")
        full_context = "\n".join(context_parts)
        logger.info(f"Full context: {full_context}")
        
        # Add the current user message to conversation history before generating the response
        conversation_history[conversation_key].append({
            "role": "user",
            "content": ai_request.text,
            "timestamp": time.time()
        })
        
        # Get the message with streaming for a better user experience
        message_ts = await stream_ai_response(full_context, ai_request, thread_ts, sender_name=sender_name)
        
        # If we got a valid response, log the conversation history
        if message_ts:
            # Note: Both the user message and bot response are already stored in the conversation history
            
            # Log conversation history for debugging
            logger.info(f"Updated conversation history for {conversation_key}, " + 
                       f"now contains {len(conversation_history[conversation_key]) - 1} messages")
            
            # Limit to last MAX_HISTORY_MESSAGES messages
            warning_message = f"⚠️ *Notice:* The conversation history has exceeded {MAX_HISTORY_MESSAGES} messages. To maintain optimal performance, I'll reset my memory here, please provide full information about what you want to ask. Previous messages will be forgotten."
            if len(conversation_history[conversation_key]) > MAX_HISTORY_MESSAGES and warning_message not in conversation_history[conversation_key]:
                # Send warning message about history reset only if it's not already in the history
                try:
                    slack_client.chat_postMessage(
                        channel=ai_request.channel_id,
                        thread_ts=thread_ts,
                        text=warning_message
                    )
                except SlackApiError as e:
                    logger.warning(f"Could not send history reset warning: {e.response.get('error', '')}")
                conversation_history[conversation_key] = conversation_history[conversation_key][-MAX_HISTORY_MESSAGES:]
                logger.info(f"Trimmed conversation history to {MAX_HISTORY_MESSAGES} messages")
        
    except Exception as e:
        logger.error(f"Error in process_mention_request: {str(e)}")
        # Send error as a new message in the thread
        try:
            slack_client.chat_postMessage(
                channel=ai_request.channel_id,
                thread_ts=thread_ts,
                text=f"Sorry, I encountered an error: {str(e)}"
            )
            slack_client.chat_delete(
                channel=ai_request.channel_id,
                ts=ai_request.message_ts
            )
        except Exception as post_error:
            logger.error(f"Error sending error message: {str(post_error)}")
        return

def check_rate_limit():
    """Check if the request is within rate limits."""
    global rate_limit_tracker
    
    current_time = time.time()
    # Reset the counter every minute
    if current_time - rate_limit_tracker["last_reset"] > 60:
        rate_limit_tracker = {
            "last_reset": current_time,
            "requests": 1
        }
        return True
    
    # Check if we've hit the limit
    if rate_limit_tracker["requests"] >= AI_RATE_LIMIT:
        return False
    
    # Increment the counter
    rate_limit_tracker["requests"] += 1
    return True

def clean_expired_conversations():
    """Remove expired conversations from memory"""
    current_time = time.time()
    expiry_seconds = CONVERSATION_EXPIRY * 3600  # Convert hours to seconds
    
    keys_to_remove = []
    for key, messages in conversation_history.items():
        if not messages:
            keys_to_remove.append(key)
            continue
            
        # Check if the last message is older than the expiry time
        last_msg = messages[-1]
        if current_time - last_msg.get("timestamp", 0) > expiry_seconds:
            keys_to_remove.append(key)
    
    # Remove expired conversations
    for key in keys_to_remove:
        del conversation_history[key]
    
    if keys_to_remove:
        logger.info(f"Cleaned {len(keys_to_remove)} expired conversations")

async def analyze_content(text: str, history_context: List[str], sender_name: str) -> ContentAnalysisResult:
    """
    Analyze the content of a message to determine its type and extract relevant information.
    
    Args:
        text: The text of the message
        history_context: The conversation history between the user and the assistant
    Returns:
        ContentAnalysisResult object with analysis results
    """
    logger.info("Analyzing message content")
    
    # Initialize with default values
    result = ContentAnalysisResult()
    result.text = text  # Store the original text
    
    # Extract URLs from the text
    if "https://" in text or "t.me/" in text:
        urls = extract_urls(text)
        result.urls = urls if urls else []
    
    
    # Use AI classification with our prompt from YAML
    try:
        # Get the prompt templates from YAML
        system_prompt = PROMPTS.get("analyze_content", {}).get("system", "")
        user_prompt_template = PROMPTS.get("analyze_content", {}).get("user_template", "")
    
        # Format the user prompt with properly joined history context
        # history_text = "\n".join(history_context) if history_context else "No previous conversation"
        user_prompt = user_prompt_template.format(
            text=text,
            conversation_history=history_context,
            sender_name=sender_name
        )

        # Call OpenAI
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        if AI_MODEL.startswith("gpt"):
            response = client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
        elif AI_MODEL.startswith("o"):
            prompt = system_prompt + "\n\n" + user_prompt
            logger.info(f"User prompt at analyze_content: {prompt}")
            response = client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        
        # Parse the response
        analysis = json.loads(response.choices[0].message.content)
        
        # Update the result with the AI analysis
        result.content_type = analysis.get("content_type", "simple_prompt")
        result.requires_slack_channel_history = analysis.get("requires_slack_channel_history", False)
        result.perform_RAG_on_linear_issues = analysis.get("perform_RAG_on_linear_issues", False)
        
        # If URLs were identified in the analysis, update our list
        if "urls" in analysis and analysis["urls"]:
            result.urls = analysis["urls"]
        
        logger.info(f"AI determined content type: {result.content_type}, requires_channel_history: {result.requires_slack_channel_history}, perform_RAG: {result.perform_RAG_on_linear_issues}, create_linear_issue: {result.create_linear_issue}")
        return result
        
    except Exception as e:
        logger.error(f"Error in AI content analysis: {str(e)}")
        # Fallback to simple_prompt for any error
        result.content_type = "simple_prompt"
        result.requires_slack_channel_history = False
        result.perform_RAG_on_linear_issues = False
        logger.info("Falling back to simple_prompt type due to error")
        return result

async def search_channel_history(channel_id: str, search_params: Dict[str, Any], query: str = None, history_context: List[str] = []):
    """
    Search channel history with AI-enhanced parameters for relevance.
    Uses AI to determine optimal search parameters based on the query.
    Identifies different types of content: text, URLs, images, videos, code blocks, files.
    
    Args:
        channel_id: Slack channel ID
        search_params: Initial search parameters (may be overridden by AI)
        query: The user's original query text
    
    Returns:
        List of processed messages with content type information
    """
    # If a query is provided, use AI to determine search parameters
    if query:
        try:
            logger.info(f"Using AI to determine search parameters for query: {query}")
            
            # Get the prompt templates from YAML
            system_prompt = PROMPTS.get("slack_search_operator", {}).get("system", "")
            user_prompt_template = PROMPTS.get("slack_search_operator", {}).get("user_template", "")
        
            if system_prompt and user_prompt_template:
                # Format the user prompt
                user_prompt = user_prompt_template.format(
                    text=query,
                    history_context="\n".join(history_context)
                )


                logger.info(f"User prompt at slack_search_operator: {user_prompt}")
        
                # Call OpenAI to analyze the query
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                if AI_MODEL.startswith("gpt"):   
                    response = client.chat.completions.create(
                        model=AI_MODEL,
                        messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                        ],
                        response_format={"type": "json_object"}
                    )
                elif AI_MODEL.startswith("o"):
                    prompt = system_prompt + "\n\n" + user_prompt
                    response = client.chat.completions.create(
                        model=AI_MODEL,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
        
                # Parse the response
                ai_params = json.loads(response.choices[0].message.content)
                logger.info(f"AI determined search parameters: {ai_params}")
                
                # Update search parameters with AI-determined values
                if "username" in ai_params and ai_params["username"]:
                    search_params["username"] = ai_params["username"]
                if "time_range" in ai_params:
                    search_params["time_range"] = ai_params["time_range"]
                if "time_value" in ai_params:
                    search_params["time_value"] = ai_params["time_value"]
                if "message_count" in ai_params:
                    search_params["message_count"] = ai_params["message_count"]
                    
                logger.info(f"Using AI-determined parameters: {search_params}")
        except Exception as e:
            logger.error(f"Error in AI-determined search parameters: {str(e)}")
            # Continue with default parameters
    else:
        logger.warning("No query provided, using default search parameters")
    
    logger.info(f"Searching channel history with parameters: {search_params}")
    
    # Define time range in seconds based on the parameters
    now = datetime.datetime.now().timestamp()
    time_unit = search_params.get("time_range", "days")
    time_value = search_params.get("time_value", 1)
    
    if time_unit == "hours":
        time_range_seconds = time_value * 3600
    elif time_unit == "days":
        time_range_seconds = time_value * 86400
    elif time_unit == "weeks":
        time_range_seconds = time_value * 604800
    else:
        time_range_seconds = 86400  # Default to 1 day
    
    # Calculate oldest timestamp to consider
    oldest_ts = str(now - time_range_seconds)
    
    try:
        # Maximum number of messages to fetch
        max_messages = search_params.get("message_count", 50)
        
        # Prepare conversation history params
        history_params = {
            "channel": channel_id,
            "limit": min(100, max_messages),
            "oldest": oldest_ts,
            "inclusive": True
        }
        
        # If username is specified, we'll filter later in Python
        username_filter = search_params.get("username")
        
        # Fetch channel history
        result = slack_client.conversations_history(**history_params)
        
        messages = result["messages"]
        
        # If we need more messages and there are more to fetch
        if max_messages > 100 and result.get("has_more", False):
            # Continue fetching older messages
            cursor = result["response_metadata"]["next_cursor"]
            while len(messages) < max_messages and cursor:
                more_result = slack_client.conversations_history(
                    channel=channel_id,
                    cursor=cursor,
                    limit=min(100, max_messages - len(messages)),
                    inclusive=True
                )
                messages.extend(more_result["messages"])
                if more_result.get("has_more", False):
                    cursor = more_result["response_metadata"]["next_cursor"]
                else:
                    break
        
        # Apply username filter if specified
        if username_filter:
            logger.info(f"Filtering messages by username: {username_filter}")
            
            # Filter messages directly by user information
            filtered_messages = []
            
            # Get all users first to compare usernames
            try:
                users_response = slack_client.users_list()
                users = users_response.get("members", [])
                
                # Find users that match the username filter
                matching_user_ids = []
                for user in users:
                    name = user.get("name", "").lower()
                    real_name = user.get("real_name", "").lower()
                    display_name = user.get("profile", {}).get("display_name", "").lower()
                    
                    # Check for matches in all name fields
                    if (username_filter.lower() == name or 
                        username_filter.lower() == real_name or
                        username_filter.lower() in name or
                        username_filter.lower() in real_name or
                        username_filter.lower() in display_name):
                        matching_user_ids.append(user.get("id"))
                
                logger.info(f"Found {len(matching_user_ids)} users matching '{username_filter}'")
                
                # Filter messages by those user IDs
                for msg in messages:
                    user_id = msg.get("user", "")
                    if user_id in matching_user_ids:
                        filtered_messages.append(msg)
                
                messages = filtered_messages
                logger.info(f"Found {len(messages)} messages from users matching '{username_filter}'")
            
            except Exception as e:
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
                urls = extract_urls(msg.get("text", ""))
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
        
        logger.info(f"Found {len(processed_messages)} messages with content breakdown: " + 
                   f"text: {sum(1 for m in processed_messages if 'text' in m['content_types'])}, " +
                   f"urls: {sum(1 for m in processed_messages if 'url' in m['content_types'])}, " +
                   f"images: {sum(1 for m in processed_messages if 'image' in m['content_types'])}, " +
                   f"videos: {sum(1 for m in processed_messages if 'video' in m['content_types'])}, " +
                   f"code: {sum(1 for m in processed_messages if 'code' in m['content_types'])}")
        
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

async def perform_linear_rag_search(query: Optional[str] = None, limit: int = 10, history_context: List[str] = [], sender_name: str = "") -> Dict[str, Any]:
    """
    Perform RAG (Retrieval Augmented Generation) search on Linear issues.
    Uses the linear_rag_search.py module to retrieve relevant issues.
    If query is provided, first analyzes the query with AI to determine optimal search parameters.
    Then performs search using these parameters if needed, or returns available company data directly.
    
    Args:
        query: The search query (optional)
        limit: Maximum number of results to return (required)
        
    Returns:
        Dictionary with search results and metadata, or company information
    """
    logger.info(f"Performing Linear RAG search with query: {query if query else 'None'}")
    
    try:
        # Import the linear_rag_search module
        from linear_rag_search import search_issues, get_available_teams_and_cycles, advanced_search
        
        # Get available teams and cycles
        available_data = get_available_teams_and_cycles()
        teams = available_data.get("teams", [])
        cycles = available_data.get("cycles", [])
        
        # Default filter to exclude titles containing 'call'
        default_filter = {
            "field": "title",
            "operator": "NOT ILIKE",
            "value": "%call%"
        }
        
        # Use static list for Linear users - these are separate from Slack users
        # This is the accurate list of Linear users with their correct names
        linear_users = get_linear_names()
        
        # Define known issue states and priorities for the prompt
        issue_states = ["Todo", "In Progress", "In Review", "Done", "Canceled"]
        issue_priorities = [
            "0 - No priority",
            "1 - Urgent",
            "2 - High",
            "3 - Medium",
            "4 - Low"
        ]
        
        # Use AI to determine optimal search parameters only if query is provided
        if query:
            logger.info("Using AI to determine optimal search parameters")
            system_prompt = PROMPTS.get("linear_search_operator", {}).get("system", "")
            user_prompt_template = PROMPTS.get("linear_search_operator", {}).get("user_template", "")
            
            if system_prompt and user_prompt_template:
                # Format the user prompt
                user_prompt = user_prompt_template.format(
                    text=query,
                    teams=", ".join([f"{team['name']} ({team['key']})" for team in teams]),
                    cycles=", ".join([cycle['name'] for cycle in cycles]),
                    cycles_data="\n".join([f"{cycle['name']}: {cycle['issue_count']}" for cycle in cycles]),
                    users=linear_users,
                    states=", ".join(issue_states),
                    priorities=", ".join(issue_priorities),
                    history_context="\n".join(history_context),
                    sender_name=sender_name
                )

                #use different model for linear search
                AI_MODEL = "o3-mini"
                # Call OpenAI to analyze the query
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                try:
                    logger.info("About to call OpenAI API")
                    if AI_MODEL.startswith("gpt"):
                        response = client.chat.completions.create(
                            model=AI_MODEL,  # Using smaller model for faster response
                            messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                            ],
                            response_format={"type": "json_object"}
                        )
                    elif AI_MODEL.startswith("o"):
                        prompt = system_prompt + "\n\n" + user_prompt
                        response = client.chat.completions.create(
                            model=AI_MODEL,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
                    logger.info("Successfully received OpenAI API response")
                except Exception as e:
                    logger.error(f"Error calling OpenAI: {e}")
                    logger.error(f"Exception details: {traceback.format_exc()}")
                    # Re-raise to be caught by outer try/except
                    raise
                # Parse the response
                try:
                    result = response.choices[0].message.content
                    logger.info(f"Raw AI-determined search parameters: {result}")
                    if result.startswith("```json"):
                        result = result.strip("```json")
                    elif result.startswith("```"):
                        result = result.strip("```")
                    result = result.strip()  # Remove any extra whitespace
                    
                    try:
                        search_params = json.loads(result)
                        
                        # Add debug logging to see what's in the search params
                        logger.info(f"Parsed search parameters: {json.dumps(search_params, indent=2)}")
                        
                    except json.JSONDecodeError as json_err:
                        logger.error(f"Failed to parse AI response as JSON: {json_err}")
                        logger.error(f"Raw response content: {result}")
                        # Fall back to basic search
                        return {
                            "requires_linear_search": True,
                            "results": search_issues(query=query, limit=limit),
                            "available_teams": teams,
                            "available_cycles": cycles,
                            "search_parameters": {
                                "team_key": None,
                                "cycle_name": None,
                                "assignee": None,
                                "query": query,
                                "limit": limit
                            },
                            "query": query,
                            "error": f"Failed to parse AI search parameters: {str(json_err)}",
                            "fallback": "Using basic search due to parsing error"
                        }
                    
                    logger.info(f"AI determined search parameters: {search_params}")
                    
                    # Check if we're using the legacy format or the new advanced format
                    if "requires_linear_search" in search_params:
                        # Legacy format - convert to advanced format
                        if not search_params.get("requires_linear_search", True):
                            # If search is not required, return the company data directly
                            logger.info("AI determined that Linear search is not needed, returning company data directly")
                            return {
                                "requires_linear_search": False,
                                "results": [],
                                "available_teams": teams,
                                "available_cycles": cycles,
                                "search_parameters": {
                                    "team_key": None,
                                    "cycle_name": None,
                                    "assignee": None,
                                    "query": query,
                                    "limit": None
                                },
                                "query": query,
                                "result_count": 0,
                                "related_existing_company_data": search_params.get("related_existing_company_data")
                            }
                        
                        # Add default filter to legacy search parameters
                        search_params["filters"] = search_params.get("filters", [])
                        search_params["filters"].append(default_filter)
                        
                        # Log the legacy search parameters
                        team_key = search_params.get("team_key")
                        cycle_name = search_params.get("cycle_name")
                        assignee_name = search_params.get("assignee_name")
                        search_limit = search_params.get("limit", limit)
                        search_query = search_params.get("search_query", query)
                        
                        logger.info(f"Using legacy search with parameters: team={team_key}, cycle={cycle_name}, assignee={assignee_name}, limit={search_limit}, query={search_query}")
                        
                        # Perform the legacy search
                        search_results = search_issues(
                            query=search_query,
                            team_key=team_key,
                            cycle_name=cycle_name,
                            assignee_name=assignee_name,
                            limit=search_limit
                        )
                        
                        # Return the results and metadata directly
                        return {
                            "requires_linear_search": True,
                            "results": search_results,
                            "available_teams": teams,
                            "available_cycles": cycles,
                            "search_parameters": {
                                "team_key": team_key,
                                "cycle_name": cycle_name,
                                "assignee": assignee_name,
                                "query": search_query,
                                "limit": search_limit
                            },
                            "query": query,
                            "result_count": len(search_results),
                            "related_existing_company_data": search_params.get("related_existing_company_data")
                        }
                    else:
                        # Check if this is a multi-step query (an array of queries)
                        is_multi_step = isinstance(search_params, list)
                        
                        if is_multi_step:
                            logger.info(f"Executing multi-step query with {len(search_params)} steps")
                            
                            # Process multi-step queries
                            all_results = []
                            step_results = {}
                            final_results = None
                            
                            for i, step_query in enumerate(search_params):
                                logger.info(f"Executing step {i+1} of multi-step query")
                                
                                # Extract the result_variable name if specified
                                result_variable = step_query.pop("result_variable", f"query_{i+1}_result")
                                
                                # Process any variable references in the query
                                processed_query = process_variable_references(step_query, step_results)
                                
                                # Execute the query
                                step_result = advanced_search(processed_query)
                                
                                # Store the results for use by subsequent queries
                                step_results[result_variable] = step_result.get("results", [])
                                
                                # Store all step results
                                all_results.append({
                                    "step": i+1, 
                                    "results": step_result.get("results", []),
                                    "count": step_result.get("count", 0),
                                    "result_variable": result_variable
                                })
                                
                                # The final step results will be returned as the main results
                                if i == len(search_params) - 1:
                                    final_results = step_result
                            
                            # Return results from all steps
                            return {
                                "requires_linear_search": True,
                                "is_multi_step": True,
                                "results": final_results.get("results", []),
                                "all_steps": all_results,
                                "available_teams": teams,
                                "available_cycles": cycles,
                                "search_parameters": search_params,
                                "query": query,
                                "result_count": final_results.get("count", 0) if final_results else 0,
                                "justification": "Multi-step query executed successfully",
                                "column_names": final_results.get("column_names", []) if final_results else []
                            }
                        else:
                            # Regular advanced format (single query)
                            logger.info(f"Using advanced search with query: {search_params}")
                            
                            # Add default filter
                            search_params["filters"] = search_params.get("filters", [])
                            search_params["filters"].append(default_filter)
                            
                            # Execute the advanced search
                            advanced_results = advanced_search(search_params)
                            logger.info(f"Advanced search results: {advanced_results}")
                            
                            # Return the results and metadata
                            return {
                                "requires_linear_search": True,
                                "results": advanced_results.get("results", []),
                                "available_teams": teams,
                                "available_cycles": cycles,
                                "search_parameters": search_params,
                                "query": query,
                                "result_count": advanced_results.get("count", 0),
                                "query_type": search_params.get("query_type"),
                                "justification": search_params.get("justification"),
                                "column_names": advanced_results.get("column_names", [])
                            }
                        
                except Exception as parse_error:
                    logger.error(f"Error parsing AI response: {parse_error}")
                    logger.error(f"Raw AI response: {response.choices[0].message.content}")
            else:
                # Fallback to defaults if prompts are not available
                logger.warning("Linear search operator prompts not found, using defaults")
        
        # If we get here, execute a basic search with default parameters
        logger.info(f"Executing basic search with query={query}, limit={limit}")
        search_results = search_issues(query=query, limit=limit)
        
        # Return the results and metadata
        return {
            "requires_linear_search": True,
            "results": search_results,
            "available_teams": teams,
            "available_cycles": cycles,
            "search_parameters": {
                "team_key": None,
                "cycle_name": None,
                "assignee": None,
                "query": query,
                "limit": limit
            },
            "query": query,
            "result_count": len(search_results),
            "related_existing_company_data": None
        }
    
    except Exception as e:
        logger.error(f"Error performing Linear search: {str(e)}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        # Log more details about the exception
        if hasattr(e, "__dict__"):
            logger.error(f"Exception details: {e.__dict__}")
        return {
            "error": str(e),
            "results": [],
            "query": query,
            "result_count": 0,
            "requires_linear_search": True,
            "related_existing_company_data": None
        }

def process_variable_references(query, step_results):
    """
    Process variable references in query values.
    
    Args:
        query: The query spec to process
        step_results: Dictionary of previous query results
        
    Returns:
        Processed query with variable references resolved
    """
    processed_query = json.loads(json.dumps(query))  # Deep copy
    
    # Process filters that might contain variable references
    if "filters" in processed_query:
        # Create a list to keep track of filters to remove (if their value lists are empty)
        filters_to_remove = []
        
        for i, filter_item in enumerate(processed_query["filters"]):
            if "value" in filter_item and isinstance(filter_item["value"], str):
                # Check if this is a variable reference
                if filter_item["value"].startswith("{{") and filter_item["value"].endswith("}}"):
                    # Extract variable reference
                    ref = filter_item["value"][2:-2]  # Remove {{ }}
                    var_name, field_name = ref.split(".")
                    
                    if var_name in step_results:
                        # Replace with actual values from previous results
                        values = [item.get(field_name) for item in step_results[var_name] if field_name in item]
                        values = [v for v in values if v is not None]  # Filter out None values
                        
                        # For IN operators, we need to handle PostgreSQL array parameters correctly
                        if filter_item.get("operator", "=") == "IN":
                            if values:
                                # Change from 'IN' to '= ANY' for proper PostgreSQL array handling
                                processed_query["filters"][i]["operator"] = "= ANY"
                                processed_query["filters"][i]["value"] = values
                            else:
                                # If no values found, mark this filter for removal
                                logger.warning(f"Empty value list for reference {ref}, will skip this filter")
                                filters_to_remove.append(i)
                        # For other operators, take the first value if available
                        elif values:
                            processed_query["filters"][i]["value"] = values[0]
                        else:
                            logger.warning(f"No values found for reference {ref}, will skip this filter")
                            filters_to_remove.append(i)
                    else:
                        logger.warning(f"Variable {var_name} not found in results, will skip this filter")
                        filters_to_remove.append(i)
        
        # Remove filters with empty value lists (in reverse order to avoid index shifting)
        for i in sorted(filters_to_remove, reverse=True):
            processed_query["filters"].pop(i)
    
    # Could extend this to process other fields that might contain references
    return processed_query

async def generate_linear_issue(query: str, history_context: str, sender_name: str) -> Dict[str, Any]:
    """
    Generate Linear issue based on the user's query.
    """
    try:
        linear_names = get_linear_names()
        create_issue_system_prompt = PROMPTS.get("linear_issue_creator", {}).get("system", "").format(
            linear_names=linear_names
        )
        if not create_issue_system_prompt:
            create_issue_system_prompt = """You are a Linear issue creator. You must respond with valid JSON containing the following fields:
            {
                "title": "issue title",
                "description": "issue description",
                "team_key": "team key (required)",
                "priority": priority number (0-4, optional, defaults to 0)
            }"""

        create_issue_user_prompt = PROMPTS.get("linear_issue_creator", {}).get("user_template", "").format(
            text=query,
            history_context=history_context,
            sender_name=sender_name
        )
        
        # Call OpenAI to analyze the query
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        if AI_MODEL.startswith("gpt"):
            response = client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": create_issue_system_prompt},
                    {"role": "user", "content": create_issue_user_prompt}
                ],
                response_format={"type": "json_object"}
            )
        elif AI_MODEL.startswith("o"):
            prompt = create_issue_system_prompt + "\n\n" + create_issue_user_prompt
            response = client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

        # Parse the response
        result = response.choices[0].message.content
        logger.info(f"AI for creating linear issue: {result}")
        if result.startswith("```json"):
            result = result.strip("```json")
        elif result.startswith("```"):
            result = result.strip("```")
        result = result.strip()  # Remove any extra whitespace
        
        linear_issue_args = json.loads(result)
        
        if linear_issue_args.get("team_key") is None:
            return {"success": False, "error": "Must specify the team key for the issue."}

        # Create issue with default priority if not specified
        created_issue = await linear_client.create_linear_issue(
            title=linear_issue_args["title"],
            description=linear_issue_args.get("description", ""),  # Default to empty string if missing
            team_key=linear_issue_args["team_key"],
            priority=linear_issue_args.get("priority", 0)  # Default to 0 if missing
        )
        
        return created_issue

    except Exception as e:
        logger.error(f"Error generating Linear issue: {str(e)}")
        return {"success": False, "error": str(e)}

async def stream_ai_response(context: str, ai_request: AIRequest, thread_ts: str, sender_name: str) -> Optional[str]:
    try:
        # Use existing message_ts from the AI request instead of creating a new message
        message_ts = ai_request.message_ts

        # Get system prompts
        slack_users = get_slack_users_list()
        draft_system_prompt = PROMPTS.get("draft_response", {}).get("system_text", "").format(
            slack_users=slack_users,
            sender_name=sender_name
        )
        if not draft_system_prompt:
            draft_system_prompt = "You are an AI assistant for Token Metrics."

        final_system_prompt = PROMPTS.get("final_response", {}).get("system_text", "")
        if not final_system_prompt:
            final_system_prompt = "Refine the draft response for Slack formatting."

        # Get conversation history for this thread
        conversation_key = f"{ai_request.channel_id}:{thread_ts}"
        messages = []
        
        # Add system message for draft
        messages.append({"role": "system", "content": draft_system_prompt})
        
        # Add conversation history if it exists
        if conversation_key in conversation_history and conversation_history[conversation_key]:
            for msg in conversation_history[conversation_key]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current query
        messages.append({"role": "user", "content": context})

        # STEP 1: Generate draft response with OpenAI
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.info(f"Sending {len(messages)} messages to OpenAI for draft generation")
        
        # Generate draft response
        if AI_MODEL.startswith("gpt"):
            draft_response = client.chat.completions.create(
                model=AI_MODEL,
                messages=messages,
            )
        elif AI_MODEL.startswith("o"):
            prompt = draft_system_prompt + "\n\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            draft_response = client.chat.completions.create(
                model=AI_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
        
        # Get the draft content
        draft_content = draft_response.choices[0].message.content
        logger.info(f"Draft response generated ({len(draft_content)} chars)")
        
        # STEP 2: Refine the draft with final_response prompt
        final_messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": f"Draft response to refine:\n\n{draft_content}"}
        ]
        
        logger.info("Sending draft to refinement")
        
        # Stream the final response
        full_response = ""
        
        # Stream the refined response
        stream = client.chat.completions.create(
            model="gpt-4o-mini", #this is easy, should be fast
            messages=final_messages,
            stream=True
        )
        
        # Create a new message in the thread for the actual response
        initial_response = slack_client.chat_postMessage(
            channel=ai_request.channel_id,
            thread_ts=thread_ts,
            text="..."
        )
        response_message_ts = initial_response["ts"]
        
        buffer = ""
        last_update_time = time.time()
        first_chunk = True
        message_parts = []  # Store message parts if we need to split
        current_part = ""
        
        for chunk in stream:
            current_time = time.time()
            
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                buffer += content
                current_part += content
                
                # Delete the acknowledge message as soon as we receive first chunk
                if first_chunk:
                    first_chunk = False
                    try:
                        # Try to delete the processing message
                        slack_client.chat_delete(
                            channel=ai_request.channel_id,
                            ts=message_ts
                        )
                        logger.info(f"Deleted processing message with ts: {message_ts}")
                    except SlackApiError as e:
                        logger.warning(f"Could not delete processing message: {e.response.get('error', '')}")
                
                # Format the buffer for Slack before sending updates
                formatted_buffer = format_for_slack(buffer)
                formatted_full_response = format_for_slack(full_response)
                
                # Check if current part is approaching Slack's limit (3000 chars to be safe)
                if len(current_part) > 3000:
                    message_parts.append(current_part)
                    current_part = ""
                
                # Update message every 2.9 seconds or when buffer reaches 100 chars
                if current_time - last_update_time > 2.9 or len(buffer) > 100:
                    try:
                        # If we have multiple parts, update the last message and create new ones
                        if message_parts:
                            # Update existing messages
                            for i, part in enumerate(message_parts[:-1]):
                                formatted_part = format_for_slack(part)
                                if i == 0:
                                    slack_client.chat_update(
                                        channel=ai_request.channel_id,
                                        ts=response_message_ts,
                                        text=formatted_part,
                                        mrkdwn=True
                                    )
                                else:
                                    # Create new message for each additional part
                                    slack_client.chat_postMessage(
                                        channel=ai_request.channel_id,
                                        thread_ts=thread_ts,
                                        text=formatted_part,
                                        mrkdwn=True
                                    )
                            # Clear processed parts
                            message_parts = []
                        
                        # Update or create message with current content
                        if current_part:
                            formatted_current = format_for_slack(current_part)
                            slack_client.chat_update(
                                channel=ai_request.channel_id,
                                ts=response_message_ts,
                                text=formatted_current,
                                mrkdwn=True
                            )
                        
                        buffer = ""
                        last_update_time = current_time
                    except SlackApiError as e:
                        if e.response["error"] == "msg_too_long":
                            # If message is too long, split it and create new messages
                            message_parts.append(current_part)
                            current_part = ""
                        else:
                            logger.warning(f"Failed to update message: {str(e)}")
        
        # Handle any remaining content
        if current_part:
            message_parts.append(current_part)
        
        # Send all remaining parts
        try:
            for i, part in enumerate(message_parts):
                formatted_part = format_for_slack(part)
                if i == 0:
                    slack_client.chat_update(
                        channel=ai_request.channel_id,
                        ts=response_message_ts,
                        text=formatted_part,
                        mrkdwn=True
                    )
                else:
                    # Create new message for each additional part
                    slack_client.chat_postMessage(
                        channel=ai_request.channel_id,
                        thread_ts=thread_ts,
                        text=formatted_part,
                        mrkdwn=True
                    )
            
            # Store the full response in conversation history
            conversation_key = f"{ai_request.channel_id}:{thread_ts}"
            conversation_history[conversation_key].append({
                "role": "assistant",
                "content": full_response,
                "timestamp": float(response_message_ts)
            })
            logger.info(f"Stored bot response in conversation history")
            
        except SlackApiError as e:
            logger.warning(f"Failed to send final messages: {str(e)}")
                
        return response_message_ts
        
    except Exception as e:
        logger.error(f"Error in streaming AI response: {str(e)}")
        try:
            slack_client.chat_postMessage(
                channel=ai_request.channel_id,
                thread_ts=thread_ts,
                text=f"Sorry, I encountered an error: {str(e)}"
            )
        except:
            pass
        return None

def format_linear_search_results(linear_results: Dict[str, Any]) -> List[str]:
    """
    Format Linear search results for inclusion in the AI context.
    
    Args:
        linear_results: Results from Linear RAG search
        
    Returns:
        List of formatted context strings
    """
    context_parts = []
    
    if "error" in linear_results:
        context_parts.append(f"\nError searching Linear issues: {linear_results['error']}")
        return context_parts
        
    if not linear_results.get("requires_linear_search", True) and linear_results.get("related_existing_company_data"):
        # If no search is needed but we have company data to share
        context_parts.append(f"\nCompany Information:")
        context_parts.append(linear_results.get("related_existing_company_data"))
        return context_parts
        
    # Check if we have a query type and it's an analytical query
    query_type = linear_results.get("query_type")
    if query_type in ["status_report", "performance_metrics", "workload_distribution", "time_based_analysis"]:
        # Format analytical results
        context_parts.append(f"\nLinear {query_type.replace('_', ' ')} results:")
        
        if linear_results.get("justification"):
            context_parts.append(f"Analysis: {linear_results.get('justification')}")
            
        results = linear_results.get("results", [])
        column_names = linear_results.get("column_names", []) or []  # Set empty list if column_names is None
        
        if not results:
            context_parts.append("No data found for this analysis.")
            return context_parts
            
        # Add table header for analytical queries with column names
        if column_names and query_type != "status_report":
            header = " | ".join(column_names)
            context_parts.append(f"\n{header}")
            context_parts.append("-" * len(header))
            
        for i, issue in enumerate(results, 1):
            # Check if this is a standard issue or not
            if isinstance(issue, dict) and "title" in issue:
                # Standard issue format - only include essential information
                issue_id = issue.get('issue_id', 'No ID')
                context_parts.append(f"{i}. {issue.get('title', 'Untitled')} ({issue_id})")
                context_parts.append(f"Description: {issue.get('description', 'No description available')}")
                
                # Add assignee info if available
                assignee = issue.get('assignee', '')
                if assignee:
                    context_parts.append(f"   Assignee: {assignee}")
                
                # Add state if available
                state = issue.get('state', '')
                if state:
                    context_parts.append(f"   State: {state}")
                
                # Add priority if available (but not other metadata)
                priority = issue.get('priority', '')
                if priority:
                    context_parts.append(f"   Priority: {priority}")
                
                # Add estimate if available
                estimate = issue.get('estimate', '')
                if estimate:
                    context_parts.append(f"   Estimated time (in hours): {estimate}")
                
                context_parts.append("--------------------------------")
            else:
                # Non-standard format, just output as is but filter out sensitive fields
                if isinstance(issue, dict):
                    # Filter out excluded fields
                    filtered_issue = {k: v for k, v in issue.items() if k not in 
                                    ['comments', 'team', 'created_at', 'updated_at', 
                                    'completed_at', 'full_context']}
                    context_parts.append(f"{i}. {filtered_issue}")
                else:
                    context_parts.append(f"{i}. {issue}")
        
        return context_parts

    elif linear_results.get("query_type") == "basic_filter":
        # For basic filters, status reports without grouping, or other query types
        # Include search parameters used in the context
        search_params = linear_results.get("search_parameters", {})
        search_context = []
        
        # Check if we're using legacy or advanced search parameters
        if isinstance(search_params, dict) and "team_key" in search_params:
            # Legacy format
            if search_params.get("team_key"):
                search_context.append(f"Team: {search_params['team_key']}")
            if search_params.get("cycle_name"):
                search_context.append(f"Cycle: {search_params['cycle_name']}")
            if search_params.get("assignee"):
                search_context.append(f"Assignee: {search_params['assignee']}")
        else:
            # Advanced format - extract filters
            filters = search_params.get("filters", [])
            for filter_item in filters:
                field = filter_item.get("field", "")
                value = filter_item.get("value", "")
                operator = filter_item.get("operator", "=")
                
                # Format the field name
                if "team->key" in field:
                    field_name = "Team"
                elif "cycle->name" in field:
                    field_name = "Cycle"
                elif "assignee->name" in field:
                    field_name = "Assignee"
                elif "state" in field:
                    field_name = "State"
                elif "priority" in field:
                    field_name = "Priority"
                else:
                    field_name = field.split("->")[-1].capitalize()
                    
                search_context.append(f"{field_name} {operator} {value}")
        
        if search_context:
            context_parts.append(f"\nSearch filters applied: {', '.join(search_context)}")
        
        # Handle standard search results
        results = linear_results.get("results", [])
        if not results:
            context_parts.append("\nNo relevant Linear issues found.")
            return context_parts
        
        context_parts.append(f"\nFound {len(results)} relevant Linear issues:")
        
        # Format Linear issues for context - exclude unnecessary data
        for i, issue in enumerate(results, 1):
            # Check if this is a standard issue or not
            if isinstance(issue, dict) and "title" in issue:
                # Standard issue format - only include essential information
                issue_id = issue.get('issue_id', 'No ID')
                context_parts.append(f"{i}. {issue.get('title', 'Untitled')} ({issue_id})")
                context_parts.append(f"Description: {issue.get('description', 'No description available')}")
                
                # Add assignee info if available
                assignee = issue.get('assignee', '')
                if assignee:
                    context_parts.append(f"   Assignee: {assignee}")
                
                # Add state if available
                state = issue.get('state', '')
                if state:
                    context_parts.append(f"   State: {state}")
                
                # Add priority if available (but not other metadata)
                priority = issue.get('priority', '')
                if priority:
                    context_parts.append(f"   Priority: {priority}")
                
                # Add estimate if available
                estimate = issue.get('estimate', '')
                if estimate:
                    context_parts.append(f"   Estimated time (in hours): {estimate}")
                
                context_parts.append("--------------------------------")
            else:
                # Non-standard format, just output as is but filter out sensitive fields
                if isinstance(issue, dict):
                    # Filter out excluded fields
                    filtered_issue = {k: v for k, v in issue.items() if k not in 
                                    ['comments', 'team', 'created_at', 'updated_at', 
                                    'completed_at', 'full_context']}
                    context_parts.append(f"{i}. {filtered_issue}")
                else:
                    context_parts.append(f"{i}. {issue}")
    else:
        # Default handler for any other query type
        results = linear_results.get("results", [])
        if not results:
            context_parts.append("\nNo relevant Linear issues found.")
            return context_parts
        
        context_parts.append(f"\nFound {len(results)} relevant Linear issues:")
        
        # Format Linear issues for context - exclude unnecessary data
        for i, issue in enumerate(results, 1):
            # Check if this is a standard issue or not
            if isinstance(issue, dict) and "title" in issue:
                # Standard issue format - only include essential information
                issue_id = issue.get('issue_id', 'No ID')
                context_parts.append(f"{i}. {issue.get('title', 'Untitled')} ({issue_id})")
                
                # Add basic information if available
                assignee = issue.get('assignee', '')
                if assignee:
                    context_parts.append(f"   Assignee: {assignee}")
                
                state = issue.get('state', '')
                if state:
                    context_parts.append(f"   State: {state}")
                
                priority = issue.get('priority', '')
                if priority:
                    context_parts.append(f"   Priority: {priority}")
                
                # Add any custom fields that were specifically requested in returned_fields
                returned_fields = linear_results.get("search_parameters", {}).get("returned_fields", {})
                if isinstance(returned_fields, dict):
                    for key, display_name in returned_fields.items():
                        if key not in ["title", "data->assignee->name", "data->state", "data->priority"] and key in issue:
                            context_parts.append(f"   {display_name}: {issue[key]}")
                
                context_parts.append("--------------------------------")
            else:
                # Non-standard format, just output as is
                context_parts.append(f"{i}. {issue}")
        
        return context_parts
        
    return context_parts

def get_slack_users_list():
    """
    Get the list of all users in the Slack workspace.
    """
    try:
        valid_users = []
        response = slack_client.users_list()
        members = response.get("members")
        for member in members:
            if member.get("deleted") == False and member.get("is_bot") == False and member.get("is_email_confirmed") == True and member.get("is_primary_owner") == False:
                valid_user = {
                    "display_name": "",
                    "real_name": "",
                    "title": "",
                    "team": ""
                }
                valid_user["display_name"] = member.get("profile", {}).get("display_name")
                valid_user["real_name"] = member.get("profile", {}).get("real_name")
                valid_user["title"] = member.get("profile", {}).get("title")
                valid_user["team"] = member.get("team")
                valid_users.append(valid_user)
        return valid_users
    except SlackApiError as e:
        logger.error(f"Slack API error: {e}")
        return []

def get_linear_names():
    """
    Get the list of all users in the Linear workspace.
    """
    linear_users = []
    raw_list =  [
            "@agustin: agustin@tokenmetrics.com: MKT",
            "@ahmedhamdy: Ahmed Hamdy: AI",
            "@andrew: Andrew Tran: AI",
            "@ankit: Dao Truong An: ENG",
            "@ashutosh: Ashutosh: ENG",
            "@ayo: Ayo: PRO",
            "@ayush: Ayush Jalan",
            "@bartosz: Bartosz Kusnierczak: ENG",
            "@ben: Ben Diagi: PRO",
            "@caleb: Caleb Nnamani: MKT",
            "@chao: Chao Li: AI",
            "@chetan: Chetan Kale",
            "@divine: Divine Anthony: ENG",
            "@faith: Faith Oladejo: PRO",
            "@favour: Favour Ikwan: OPS",
            "@grady: Grady Matthias Oktavian: AI",
            "@harshg: Harsh Gautam: ENG",
            "@hemank: Hemank Sharma",
            "@ian: Ian Balina",
            "@jake: Jake Nguyen: AI",
            "@khadijah: khadijah@tokenmetrics.com: OPS",
            "@manav: Manav Garg: ENG",
            "@noel: Emanuel Cruz: MKT",
            "@olaitan: Olaitan Akintunde: MKT",
            "@ozcan: Ozcan Ilhan: ENG",
            "@peterson: Peterson Nwoko: ENG",
            "@phat: Phát -: OPS",
            "@raldrich: Raldrich Oracion: PRO",
            "@roshan1: Roshan Ganesh: MKT",
            "@salman: Salman Haider",
            "@sam: Sam Monac",
            "@suleman: Suleman Tariq: ENG",
            "@tafcirm: Tafcir Majumder: MKT",
            "@talha: Talha Ahmad: OPS",
            "@talhacagri: Talha Çağrı Kotcioglu: AI",
            "@val: Valentine Enedah: PRO",
            "@vasilis: Vasilis Kotopoulos: AI",
            "@williams: Williams Cherechi: ENG",
            "@zaiying: Zaiying Li: OPS"
        ]
    for user in raw_list:
        linear_user = {
            "username": "",
            "real_name": "",
            "team": ""  
        }
        linear_user["username"] = user.split(":")[0].strip()
        linear_user["real_name"] = user.split(":")[1].strip()
        if len(user.split(":")) > 2:
            linear_user["team"] = user.split(":")[2].strip()
        else:
            linear_user["team"] = None
        linear_users.append(linear_user)
    return linear_users

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)