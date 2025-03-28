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
from database import get_db, check_db_connection, init_db, cleanup_old_conversations
from models import Conversation, Message, Base
from rate_limiter import global_limiter, slack_limiter, linear_limiter, openai_limiter
from logging.handlers import RotatingFileHandler

from linear_rag_search import process_variable_references

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("slack_ai_bot.log", maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'),  # 10MB per file, keep 5 backups
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("slack_ai_bot")

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded")

# Configuration variables - loaded from environment variables
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
AI_MODEL = os.environ.get("AI_MODEL", "gpt-4o-mini")

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

# Rate limiting tracker - to be replaced by rate_limiter classes
# rate_limit_tracker = {
#     "last_reset": time.time(),
#     "requests": 0
# }

# Conversation memory store - keys are channel_id+thread_ts, values are lists of messages
conversation_history = defaultdict(list)
# Maximum age of conversation history in memory (in hours)
CONVERSATION_EXPIRY = 24  # hours

slack_users = [
    {'display_name': 'Talha', 'real_name': 'Talha Ahmad', 'title': 'Operations Manager', 'team': 'OPS'},
    {'display_name': 'Val', 'real_name': 'Valentine Enedah', 'title': 'CX', 'team': 'PRO'},
    {'display_name': 'Ian Balina', 'real_name': 'Ian Balina', 'title': 'Founder and CEO', 'team': None},
    {'display_name': 'Harsh', 'real_name': 'Harsh', 'title': 'Senior Full Stack Engineer', 'team': 'ENG'},
    {'display_name': '', 'real_name': 'Andrew Tran', 'title': 'Data Engineer', 'team': 'AI'},
    {'display_name': 'Ayush Jalan', 'real_name': 'Ayush Jalan', 'title': 'Blockchain Engineer', 'team': 'ENG'},
    {'display_name': 'Drich', 'real_name': 'Raldrich Oracion', 'title': 'Customer Success', 'team': 'PRO'},
    {'display_name': 'Bartosz', 'real_name': 'Bartosz Kusnierczak', 'title': 'Passion never fail', 'team': 'ENG'},
    {'display_name': 'Jake', 'real_name': 'Jake Nguyen', 'title': 'Senior Data Engineer', 'team': 'AI'},
    {'display_name': 'Roshan Ganesh', 'real_name': 'Roshan Ganesh', 'title': 'Marketing Lead', 'team': 'MKT'},
    {'display_name': 'Sam Monac', 'real_name': 'Sam Monac', 'title': 'Chief Product Officer', 'team': None},
    {'display_name': 'Favour', 'real_name': 'Favour Ikwan', 'title': 'Chief Operations Officer', 'team': 'OPS'},
    {'display_name': 'Suleman Tariq', 'real_name': 'Suleman Tariq', 'title': 'Tech Lead', 'team': 'ENG'},
    {'display_name': 'Zaiying Li', 'real_name': 'Zaiying Li', 'title': '', 'team': 'OPS'},
    {'display_name': 'Hemank', 'real_name': 'Hemank', 'title': '', 'team': 'RES'},
    {'display_name': 'Ben', 'real_name': 'Ben Diagi', 'title': 'Product Manager', 'team': 'PRO'},
    {'display_name': 'Chao', 'real_name': 'Chao Li', 'title': 'Quantitative Analyst', 'team': 'AI'},
    {'display_name': 'Abdullah', 'real_name': 'Abdullah', 'title': 'Head Of Investment', 'team': 'RES'},
    {'display_name': 'Manav', 'real_name': 'Manav Garg', 'title': 'Blockchain Engineer', 'team': 'RES'},
    {'display_name': 'Vasilis', 'real_name': 'Vasilis Kotopoulos', 'title': 'AI Team Lead', 'team': 'AI'},
    {'display_name': 'Olaitan Akintunde', 'real_name': 'Olaitan Akintunde', 'title': 'Video Editor and Motion Designer', 'team': 'MKT'},
    {'display_name': 'Chetan Kale', 'real_name': 'Chetan Kale', 'title': '', 'team': 'RES'},
    {'display_name': 'ayo', 'real_name': 'ayo', 'title': '', 'team': 'PRO'},
    {'display_name': 'Özcan İlhan', 'real_name': 'Özcan İlhan', 'title': '', 'team': 'ENG'},
    {'display_name': 'Faith Oladejo', 'real_name': 'Faith Oladejo', 'title': '', 'team': 'PRO'},
    {'display_name': 'Taf', 'real_name': 'Tafcir Majumder', 'title': 'Head Of Business Development', 'team': 'MKT'},
    {'display_name': 'Caleb N', 'real_name': 'Caleb', 'title': '', 'team': 'MKT'},
    {'display_name': 'divine', 'real_name': 'Divine Anthony', 'title': 'Devops', 'team': 'ENG'},
    {'display_name': 'Williams', 'real_name': 'Williams Williams', 'title': 'Senior Fullstack Engineer', 'team': 'ENG'},
    {'display_name': 'Anki Truong', 'real_name': 'Truong An (Anki)', 'title': '', 'team': 'ENG'},
    {'display_name': 'Ryan', 'real_name': 'Ryan Barcelona', 'title': 'Freelancer', 'team': 'MKT'},
    {'display_name': 'Phát -', 'real_name': 'Phát -', 'title': '', 'team': 'OPS'},
    {'display_name': 'AhmedHamdy', 'real_name': 'AhmedHamdy', 'title': 'Senior Data Scientist/ML Engineer', 'team': 'AI'},
    {'display_name': 'Grady', 'real_name': 'Grady', 'title': 'Data Scientist/AI Engineer', 'team': 'AI'},
    {'display_name': 'Khadijah', 'real_name': 'Khadijah Shogbuyi', 'title': '', 'team': 'OPS'},
    {'display_name': 'Talha Cagri', 'real_name': 'Talha Cagri Kotcioglu', 'title': 'Quantitative Analyst', 'team': 'AI'},
    {'display_name': 'Agustín Gamoneda', 'real_name': 'Agustín Gamoneda', 'title': '', 'team': 'MKT'},
    {'display_name': 'Peterson', 'real_name': 'Peterson Nwoko', 'title': 'Sr DevOps/SRE Engineer', 'team': 'ENG'}
]


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
    thread_ts: Optional[str] = None  # Thread timestamp for threading

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


def get_slack_users_list():
    """
    Get the list of all users in the Slack workspace.
    """
    try:
        # Apply rate limiting for Slack API
        if not slack_limiter.check_rate_limit():
            logger.warning("Slack API rate limit exceeded, waiting...")
            slack_limiter.wait_if_needed()
        
        response = slack_client.users_list()
        members = response.get("members")
        valid_users = []
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

# Function to load conversation history from database
def load_conversation_from_db(channel_id, thread_ts):
    """Load conversation history from database"""
    conversation_key = f"{channel_id}:{thread_ts}"
    
    # If found in memory cache
    if conversation_key in conversation_history:
        messages = conversation_history[conversation_key]
        
        # Apply history limiting (choose one or combine approaches)
        if len(messages) > 10:  # Only apply to longer conversations
            # Option 1: Simple truncation to last N messages
            messages = messages[-10:]
            
            # Option 2: Add a summary marker at the start
            messages.insert(0, {
                "role": "system", 
                "content": f"[Previous conversation history summarized: {len(messages)-10} earlier messages omitted]"
            })
        
        return messages
    
    try:
        with get_db() as db:
            # Check if conversation exists
            conversation = db.query(Conversation).filter(Conversation.id == conversation_key).first()
            
            if not conversation:
                logger.info(f"No conversation found in database for {conversation_key}")
                return []
                
            # Get all messages for this conversation
            messages = db.query(Message).filter(Message.conversation_id == conversation_key).order_by(Message.timestamp).all()
            
            if not messages:
                logger.info(f"No messages found in database for conversation {conversation_key}")
                return []
                
            # Convert to the format used by conversation_history
            history = []
            for msg in messages:
                history.append({
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "message_ts": msg.message_ts,
                    "metadata": msg.meta_data or {}
                })
                
            logger.info(f"Loaded {len(history)} messages from database for {conversation_key}")
            return history
    except Exception as e:
        logger.error(f"Error loading conversation from database: {str(e)}")
        return []

# Function to save conversation to database
def save_conversation_to_db(channel_id, thread_ts, messages):
    """Save conversation history to database"""
    if not messages:
        return
        
    conversation_key = f"{channel_id}:{thread_ts}"
    
    try:
        with get_db() as db:
            # Check if conversation exists
            conversation = db.query(Conversation).filter(Conversation.id == conversation_key).first()
            
            # Create conversation if it doesn't exist
            if not conversation:
                conversation = Conversation(
                    id=conversation_key,
                    channel_id=channel_id,
                    thread_ts=thread_ts
                )
                db.add(conversation)
                db.flush()
            
            # Get existing message timestamps to avoid duplicates
            existing_timestamps = {m.timestamp for m in db.query(Message.timestamp).filter(Message.conversation_id == conversation_key).all()}
            
            # Add new messages
            for msg in messages:
                if msg["timestamp"] in existing_timestamps:
                    continue
                    
                db_message = Message(
                    conversation_id=conversation_key,
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=msg["timestamp"],
                    message_ts=msg.get("message_ts"),
                    meta_data=msg.get("metadata", {})
                )
                db.add(db_message)
            
            db.commit()
            logger.info(f"Saved conversation to database: {conversation_key}")
    except Exception as e:
        logger.error(f"Error saving conversation to database: {str(e)}")

def add_message_to_conversation(conversation_key, role, content, message_ts=None, metadata=None):
    """Add a message to the conversation history."""
    # Store only the raw message content without any formatting
    if conversation_key not in conversation_history:
        conversation_history[conversation_key] = []
    
    # Create the message object with raw content
    message = {
        "role": role,
        "content": content,
        "timestamp": time.time(),
        "message_ts": message_ts,
        "metadata": metadata
    }
    
    conversation_history[conversation_key].append(message)
    
    # Save to database
    try:
        parts = conversation_key.split(":")
        if len(parts) == 2:
            channel_id, thread_ts = parts
            save_conversation_to_db(channel_id, thread_ts, conversation_history[conversation_key])
    except Exception as e:
        logger.error(f"Error saving conversation to database: {str(e)}")

def parse_user_mentions(text: str, use_cache: bool = True) -> str:
    """
    Convert Slack user mentions (<@U123ABC>) to their actual display names.
    
    Args:
        text: The text containing user mentions
        use_cache: Whether to use cached user info (default: True)
    
    Returns:
        Text with user mentions replaced by display names
    """
    # Cache for user information
    if not hasattr(parse_user_mentions, '_user_cache'):
        parse_user_mentions._user_cache = {}

    # Find all user mentions (<@U123ABC>)
    mention_pattern = r'<@([A-Z0-9]+)>'
    mentions = re.findall(mention_pattern, text)
    
    try:
        for user_id in mentions:
            display_name = None
            
            # Check cache first if enabled
            if use_cache and user_id in parse_user_mentions._user_cache:
                display_name = parse_user_mentions._user_cache[user_id]
            else:
                try:
                    # Apply rate limiting for Slack API
                    if not slack_limiter.check_rate_limit():
                        logger.warning("Slack API rate limit exceeded, waiting...")
                        slack_limiter.wait_if_needed()
                        
                    user_info = slack_client.users_info(user=user_id)
                    if user_info["ok"]:
                        display_name = (
                            user_info["user"]["profile"].get("display_name") or
                            user_info["user"]["profile"].get("real_name") or
                            f"<@{user_id}>"  # Fallback to original mention
                        )
                        
                        # Cache the result if caching is enabled
                        if use_cache:
                            parse_user_mentions._user_cache[user_id] = display_name
                    
                except SlackApiError as e:
                    logger.error(f"Error getting user info for {user_id}: {str(e)}")
                    display_name = f"<@{user_id}>"  # Keep original mention on error
            
            if display_name:
                text = text.replace(f"<@{user_id}>", display_name)
    
    except Exception as e:
        logger.error(f"Error in parse_user_mentions: {str(e)}")
        # Return original text if parsing fails
        return text
    
    return text

def safe_append(context_list, content):
    """
    Safely append content to the context list, ensuring it's always a string.
    
    Args:
        context_list: The list to append to (usually context_parts)
        content: The content to append (could be string, list, or other)
    """
    if isinstance(content, list):
        # If it's a list, join it with newlines
        context_list.append("\n".join(str(item) for item in content))
    elif content is not None:
        # For any other type, convert to string
        context_list.append(str(content))
    # If None, don't append anything

@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    logger.info("Slack events endpoint called")
    
    payload = await request.json()

    # Normal event processing continues...
    if payload.get("type") == "event_callback":
        event = payload.get("event", {})
        event_type = event.get("type")
        event_id = payload.get("event_id", "unknown")  # Get unique event ID
        channel_type = event.get("channel_type", "")
        
        # Log full event details for debugging
        logger.info(f"Processing event ID: {event_id}")
        logger.info(f"Event type: {event_type}")
        logger.info("Event: {event}")
        if event.get("subtype"):
            return {}

        # Handle direct messages to the bot
        if event_type == "message" and channel_type == "im":
            # Skip if no text content or if it's a bot message or processing message
            text = event.get("text", "")
            #skip update message
            if not text.strip() or event.get("bot_id") or "TMAI's neuron firing..." in text:
                return {}            

            # Extract user ID, channel ID, and text
            user_id = event.get("user", "")
            channel_id = event.get("channel", "")
            message_ts = event.get("ts", "")
            
            # Check if this message is part of a thread
            # If thread_ts exists, use that. Otherwise use the current message ts
            thread_ts = event.get("thread_ts", message_ts)
            
            logger.info(f"Message details - User: {user_id}, Channel: {channel_id}, Thread: {thread_ts}, Text: {text[:100]}...")
            
            # Create AI request object
            ai_request = AIRequest(
                text=parse_user_mentions(text),
                user_id=user_id,
                channel_id=channel_id,
                message_ts=message_ts,
                thread_ts=thread_ts  # Add thread_ts to track threading
            )
            
            # Send initial "processing" message
            try:
                processing_msg = slack_client.chat_postMessage(
                    channel=channel_id,
                    text="TMAI's neuron firing......",
                    thread_ts=thread_ts,  # Add thread_ts here
                    blocks=[
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"\n*Status:*\n```\n➤ Analyzing your query...\n```"
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
                )
                
                # Update the AI request with the processing message timestamp
                ai_request.message_ts = processing_msg.get("ts")

            except SlackApiError as e:
                logger.error(f"Error posting initial DM message: {e.response['error']}")
            
            # Start processing in the background
            background_tasks.add_task(process_direct_message, ai_request)
            
            return {}
            
        # Handle bot mentions
        elif event_type == "app_mention":
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
            
            # Remove the bot mention from the text (matches <@BOTID>)
            text = text.replace(f"<@U08GNQ8F2RH>", "")

            # Create AI request object
            ai_request = AIRequest(
                text=parse_user_mentions(text),
                user_id=user_id,
                channel_id=channel_id,
                message_ts=message_ts,
                thread_ts=thread_ts  # Add thread_ts to track threading
            )
            
            # Send initial "processing" message
            try:
                initial_response = slack_client.chat_postMessage(
                    channel=channel_id,
                    text="TMAI's neuron firing......",
                    thread_ts=thread_ts,
                    blocks=[
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"\n*Status:*\n```\n➤ Analyzing your query...\n```"
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
                                    "text": "Harnessing the power of a trillion artificial neurons... and slightly fewer from my creator ◓"
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
    
    

async def process_direct_message(ai_request: AIRequest):
    """
    Process a direct message to the bot and generate a response.
    
    Args:
        ai_request: The AI request object containing user, channel, text, and thread info
    """
    start_time = time.time()
    logger.info(f"Processing direct message from user {ai_request.user_id} in channel {ai_request.channel_id}")
    
    try:
        # Check rate limiting
        if not check_rate_limit():
            try:
                slack_client.chat_update(
                    channel=ai_request.channel_id,
                    ts=ai_request.message_ts,
                    text="⚠️ Rate limit exceeded. Please try again in a minute."
                )
            except SlackApiError as e:
                logger.error(f"Error updating message with rate limit notice: {e.response['error']}")
            return
        
        # Use thread_ts from the request for threading
        thread_ts = ai_request.thread_ts
        conversation_key = f"{ai_request.channel_id}:{thread_ts}"

        # Parse response to return the correct user id
        ai_request.text = parse_user_mentions(ai_request.text)

        # Obtain the user's display name
        sender_name = "User"  # Default fallback
        try:
            user_info = slack_client.users_info(user=ai_request.user_id)
            if user_info.get("ok") and user_info.get("user"):
                sender_name = user_info["user"].get("real_name") or user_info["user"].get("name") or "User"
        except SlackApiError as e:
            logger.warning(f"Could not get user name, using default: {e.response['error']}")

        # Try loading from in-memory cache first
        if conversation_key in conversation_history and conversation_history[conversation_key]:
            logger.info(f"Using in-memory conversation history with {len(conversation_history[conversation_key])} messages")
        else:
            # If not in memory, try to load from database or rebuild from Slack
            logger.info(f"DM conversation not in memory, attempting to load from database")
            db_messages = load_conversation_from_db(ai_request.channel_id, thread_ts)
            
            if db_messages:
                # If found in database, update in-memory cache
                conversation_history[conversation_key] = db_messages
                logger.info(f"Loaded DM conversation history from database with {len(db_messages)} messages")
            else:
                # For DM, try to rebuild history from recent messages
                logger.info(f"Rebuilding conversation history from Slack for DM channel {ai_request.channel_id} in thread {thread_ts}")
                
                try:
                    # Get recent messages in the DM channel
                    response = slack_client.conversations_replies(
                        channel=ai_request.channel_id,
                        ts=thread_ts
                    )
                    
                    if response.get("ok") and response.get("messages"):
                        # Process the messages
                        messages = response.get("messages", [])
                        conversation_history[conversation_key] = []
                        
                        # Reverse the messages to get them in chronological order
                        messages.reverse()
                        
                        for msg in messages:
                            # Determine if it's a user or bot message
                            is_bot = msg.get("bot_id") is not None
                            text = msg.get("text", "")
                            
                            if not text:
                                continue

                            # For bot messages, remove "TMAI's neuron firing......" and other status indicators
                            if is_bot:
                                if "TMAI's neuron firing..." in text or "is thinking" in text:
                                    continue
                            
                            #skip default thread signal
                            if "New Assistant Thread" in text:
                                continue
                            
                            # Add to conversation history
                            add_message_to_conversation(
                                conversation_key,
                                "assistant" if is_bot else "user",
                                text,
                                msg.get("ts")
                            )
                        
                        logger.info(f"Rebuilt DM conversation history with {len(conversation_history[conversation_key])} messages")
                    else:
                        logger.warning(f"Failed to get DM history: {response.get('error', 'Unknown error')}")
                except SlackApiError as e:
                    logger.error(f"Error getting DM history: {e.response['error']}")
        
        # Build context for AI - this is just formatting, NOT storage
        context_parts = []
        context_parts.append(f"Current {sender_name} query: {parse_user_mentions(ai_request.text)}\n")

        # Format conversation history for context
        history_context = []
        
        # Add conversation history if it exists
        if conversation_key in conversation_history and conversation_history[conversation_key]:
            # Limit to 20 most recent messages
            messages = conversation_history[conversation_key]
            if len(messages) > 20:
                messages = messages[-20:]  # Keep only the last 20 messages
                history_context = [f"**Conversation History:** Showing only the 20 most recent messages."]
            else:
                history_context = [f"**Here is the conversation history between you and {sender_name} so far:**"]
            
            # Format the messages for context (this is just for display, not storage)
            for i, msg in enumerate(messages):
                if msg["role"] == "user":
                    # Format user messages with sender name
                    msg_content = parse_user_mentions(msg['content'])
                    history_context.append(f"**{sender_name} (#{i} turn):** {msg_content}")
                else:
                    # Format assistant messages
                    content = msg["content"]
                    if len(content) > 500:
                        content = content[:500] + "... (content truncated)"
                    history_context.append(f"**Assistant (#{i - 1} turn):** {content}")

        logger.info(f"History context throughout the conversation: {' '.join(history_context)}")

        # Add the raw message to conversation history (only once!)
        add_message_to_conversation(conversation_key, "user", parse_user_mentions(ai_request.text), ai_request.message_ts)
        
        # Step 1: Analyze the content to determine what we need to do
        try:
            content_analysis = await analyze_content(ai_request.text, "\n".join(history_context), sender_name=sender_name)
            logger.info(f"Content analysis: {content_analysis}")
            
            # Manual check for Linear issue creation intent
            if not content_analysis.create_linear_issue:
                # Keywords that strongly indicate issue creation intent
                issue_creation_keywords = [
                    "create issue", "create a issue", "create an issue", 
                    "make issue", "make a issue", "make an issue",
                    "new issue", "add issue", "add a issue", "add an issue",
                    "create task", "create a task", "new task", "add task",
                    "create ticket", "make ticket", "new ticket",
                    "title", "title is", "title:", "titled"
                ]
                
                # Check for title/description patterns
                title_desc_pattern = re.search(r'title[:\s]+(.*?)(?:\s*description[:\s]+|\s*$)', ai_request.text, re.IGNORECASE)
                
                # Check for keywords
                has_creation_keyword = any(keyword.lower() in ai_request.text.lower() for keyword in issue_creation_keywords)
                
                # Set create_linear_issue to True if any pattern matches
                if has_creation_keyword or title_desc_pattern:
                    logger.info("Manual detection found Linear issue creation intent")
                    content_analysis.create_linear_issue = True
                    content_analysis.content_type = "prompt_requires_tool_use"
            
        except Exception as e:
            error_msg = f"Error analyzing content: {str(e)}"
            logger.error(error_msg)
            content_analysis = ContentAnalysisResult()
            content_analysis.text = ai_request.text
            content_analysis.content_type = "simple_prompt"  # Default to simple prompt
            
        # Extract URLs if present in the text
        urls = extract_urls(ai_request.text)
        if urls:
            content_analysis.urls = urls
            ai_request.urls = urls
            
        # Step 2: Update the processing message with analysis results
        response_stage_text = "*Status:*\n```\n✓ Query analyzed\n"
        
        # This shows the additional actions we'll take
        if content_analysis.content_type == "prompt_requires_tool_use":            
            if content_analysis.urls:
                response_stage_text += "  ➤ Processing URLs\n"
                
            if content_analysis.requires_slack_channel_history:
                response_stage_text += "  ➤ Searching channel history\n"
                
            if content_analysis.perform_RAG_on_linear_issues:
                response_stage_text += "  ➤ Searching Linear\n"
                
            if content_analysis.create_linear_issue:
                response_stage_text += "  ➤ Creating Linear issue\n"
        else:
            response_stage_text += "➤ Simple query processing\n"

        response_stage_text += "```"
        
        # Add all URLs that are being processed
        if content_analysis.urls:
            for url in content_analysis.urls[:3]:  # Show max 3 URLs to avoid clutter
                shortened_url = url[:50] + "..." if len(url) > 50 else url
                response_stage_text += f"  - URL: {shortened_url}\n"
            if len(content_analysis.urls) > 3:
                response_stage_text += f"  - ...and {len(content_analysis.urls) - 3} more URLs\n"
                
        try:
            # Update the initial processing message
            slack_client.chat_update(
                channel=ai_request.channel_id,
                ts=ai_request.message_ts,
                text="TMAI's neuron firing",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"\n{response_stage_text}"
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
                                "text": "Harnessing the power of a trillion artificial neurons... and slightly fewer from my creator ◑"
                            }
                        ]
                    }
                ]
            )
        except SlackApiError as e:
            logger.warning(f"Could not update processing message: {e.response.get('error', '')}")
            
        # Step 3: Process any URLs
        url_results = []
        if content_analysis.urls:
            # Process each URL
            for url in content_analysis.urls:
                try:
                    result = await fetch_url_content(url)
                    url_results.append(result)
                    logger.info(f"Processed URL {url}: got {len(result.get('text', ''))} chars of content")
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {str(e)}")
                    url_results.append({
                        "url": url,
                        "text": f"Error processing URL: {str(e)}",
                        "error": str(e)
                    })
            
            # Add URL content to context
            for result in url_results:
                if "text" in result and result["text"]:
                    url_text = result["text"]
                    # Limit text to 2000 chars to avoid context overflow
                    if len(url_text) > 2000:
                        url_text = url_text[:2000] + "... (content truncated)"
                    
                    safe_append(context_parts, f"Content from URL {result['url']}:\n{url_text}\n")

            response_stage_text = response_stage_text.replace("  ➤ Processing URLs\n", "✓ Processing URLs\n")
            # Update message to show we're processing URLs
            try:
                slack_client.chat_update(
                    channel=ai_request.channel_id,
                    text="TMAI's neuron firing",
                    ts=ai_request.message_ts,
                    blocks=[
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"\n{response_stage_text}"
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
                                    "text": "Harnessing the power of a trillion artificial neurons... and slightly fewer from my creator ◒"
                                }
                            ]
                        }
                    ]
                )
            except SlackApiError as e:
                logger.warning(f"Could not update processing message: {e.response.get('error', '')}")
                
        # Step 4: Search Slack channel history if needed
        if content_analysis.requires_slack_channel_history:                
            # Search channel history (limited for DMs)
            search_results = await search_channel_history(
                ai_request.channel_id,
                {"limit": 100},  # Limit to 100 messages
                query=content_analysis.text,
                history_context=history_context
            )

            response_stage_text = response_stage_text.replace("  ➤ Searching channel history\n", "✓ Searching channel history\n")
            try:
                slack_client.chat_update(
                    channel=ai_request.channel_id,
                    text="TMAI's neuron firing",
                    ts=ai_request.message_ts,
                    blocks=[
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"\n{response_stage_text}"
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
                )
            except SlackApiError as e:
                logger.warning(f"Could not update processing message: {e.response.get('error', '')}")
            
            if search_results and search_results.get("messages"):
                search_content = "Channel history search results:\n"
                for msg in search_results.get("messages", [])[:5]:  # Limit to 5 most relevant
                    search_content += f"- {msg.get('text', '')}\n"
                
                safe_append(context_parts, search_content)
                
        # Step 5: Perform RAG on Linear issues if needed
        if content_analysis.perform_RAG_on_linear_issues:         
            # Search Linear issues
            linear_results = await perform_linear_rag_search(content_analysis.text, limit=5, history_context=history_context, sender_name=sender_name)
            
            if linear_results and linear_results.get("results"):
                formatted_issues = format_linear_search_results(linear_results)
                safe_append(context_parts, "\n".join(formatted_issues))

            
            response_stage_text = response_stage_text.replace("  ➤ Searching Linear\n", "✓ Searching Linear\n")

            try:
                slack_client.chat_update(
                    channel=ai_request.channel_id,
                    text="TMAI's neuron firing",
                    ts=ai_request.message_ts,
                    blocks=[
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"\n{response_stage_text}"
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
                                    "text": "Harnessing the power of a trillion artificial neurons... and slightly fewer from my creator ◓"
                                }
                            ]
                        }
                    ]
                )
            except SlackApiError as e:
                logger.warning(f"Could not update processing message: {e.response.get('error', '')}")
                
        # Step 6: Create Linear issue if needed
        if content_analysis.create_linear_issue:
                
            # Create Linear issue
            result = await generate_linear_issue(content_analysis.text, "\n".join(history_context), sender_name)
            
            response_stage_text = response_stage_text.replace("  ➤ Creating Linear issue\n", "✓ Creating Linear issue\n")
            try:
                slack_client.chat_update(
                    channel=ai_request.channel_id,
                    text="TMAI's neuron firing",
                    ts=ai_request.message_ts,
                    blocks=[
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"\n{response_stage_text}"
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
                                    "text": "Harnessing the power of a trillion artificial neurons... and slightly fewer from my creator ◑"
                                }
                            ]
                        }
                    ]
                )
            except SlackApiError as e:
                logger.warning(f"Could not update processing message: {e.response.get('error', '')}")
            
            if result and result.get("success"):
                issue_url = result.get("issue_url", "")
                issue_title = result.get("issue_title", "")
                response_stage_text += f"✓ Created Linear issue: {issue_title}\n"
                safe_append(context_parts, f"Created Linear issue: {issue_title}. URL: {issue_url}")
            else:
                response_stage_text += f"❌ Error creating Linear issue: {result.get('error', 'Unknown error')}\n"
        
        response_stage_text += " ```➤ Generating response...\n```"
        
        try:
            slack_client.chat_update(
                channel=ai_request.channel_id,
                text="TMAI's neuron firing",
                ts=ai_request.message_ts,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"\n{response_stage_text}"
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
                                "text": "Harnessing the power of a trillion artificial neurons... and slightly fewer from my creator ◒"
                            }
                        ]
                    }
                ]
            )
        except SlackApiError as e:
            logger.warning(f"Could not update processing message: {e.response.get('error', '')}")
        
        # Step 7: Call AI with the full context including conversation history
        logger.info("Building final context for AI call with conversation history")
        # Ensure all context parts are strings before joining
        for i in range(len(context_parts)):
            if not isinstance(context_parts[i], str):
                logger.warning(f"Non-string found in context_parts at index {i}: {type(context_parts[i])}")
                context_parts[i] = str(context_parts[i])
                
        final_context = "\n".join(context_parts)
        
        # Call AI to generate response
        final_response = await stream_ai_response(final_context, "\n".join(history_context), ai_request, thread_ts, sender_name)
        
        # Log the time it took to process the request
        elapsed_time = time.time() - start_time
        logger.info(f"Processed direct message in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        error_msg = f"Error processing direct message request: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        try:
            slack_client.chat_update(
                channel=ai_request.channel_id,
                ts=ai_request.message_ts,
                text=f"⚠️ Error: {error_msg}\n\nPlease try again later."
            )
        except SlackApiError as slack_err:
            logger.error(f"Error sending error message: {slack_err.response.get('error', '')}")

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
    Fetch content from a URL and return it as structured data.
    Uses specialized libraries for Twitter (Tweepy) and GitHub (PyGithub).
    
    Args:
        url: The URL to fetch content from
        
    Returns:
        Dict with structured content data
    """
    try:
        result = {
            "url": url,
            "text": "",
            "title": "",
            "is_public_url": False,
            "is_tweet": False,
            "tweet_id": None,
            "user_id": None,
            "thread_tweets": None,
            "is_github": False,
            "repo_info": None,
            "error": None
        }
        
        # Extract Twitter URLs
        twitter_pattern = r"(?:twitter\.com|x\.com)/([^/]+)/status/(\d+)"
        twitter_match = re.search(twitter_pattern, url)
        
        # Extract GitHub repository URLs
        github_repo_pattern = r"github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+))?"
        github_repo_match = re.search(github_repo_pattern, url)
        
        # Extract GitHub issue or PR URLs
        github_issue_pattern = r"github\.com/([^/]+)/([^/]+)/(?:issues|pull)/(\d+)"
        github_issue_match = re.search(github_issue_pattern, url)
        
        # Handle Twitter URLs with Tweepy
        if twitter_match and TWITTER_BEARER_TOKEN:
            try:
                result["is_tweet"] = True
                user_screen_name = twitter_match.group(1)
                tweet_id = twitter_match.group(2)
                result["user_id"] = user_screen_name
                result["tweet_id"] = tweet_id
                
                # Initialize Tweepy client
                client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
                
                # Get tweet using Tweepy
                tweet = client.get_tweet(
                    id=tweet_id,
                    expansions=["author_id", "referenced_tweets.id", "attachments.media_keys"],
                    tweet_fields=["created_at", "text", "conversation_id"],
                    user_fields=["name", "username"]
                )
                
                if tweet.data:
                    # Extract tweet text
                    tweet_text = tweet.data.text
                    
                    # Get author information
                    author = None
                    if tweet.includes and "users" in tweet.includes:
                        for user in tweet.includes["users"]:
                            if user.id == tweet.data.author_id:
                                author = user
                                break
                    
                    # Format the tweet content
                    result["text"] = "Tweet Content:\n\n"
                    if author:
                        result["text"] += f"Author: {author.name} (@{author.username})\n"
                    result["text"] += f"Text: {tweet_text}\n"
                    result["text"] += f"Posted at: {tweet.data.created_at}\n"
                    
                    # Check if it's part of a thread
                    if hasattr(tweet.data, "conversation_id") and tweet.data.conversation_id != tweet_id:
                        # This is part of a conversation/thread
                        result["text"] += "\nThis tweet is part of a conversation thread."
                        
                        # Try to get more tweets from the thread
                        # Note: This is limited by the Twitter API
                        try:
                            # Get conversation thread
                            thread_tweets = client.search_recent_tweets(
                                query=f"conversation_id:{tweet.data.conversation_id}",
                                max_results=10
                            )
                            
                            if thread_tweets.data:
                                result["thread_tweets"] = []
                                result["text"] += "\n\nThread context (up to 10 tweets):\n"
                                
                                for thread_tweet in thread_tweets.data:
                                    result["thread_tweets"].append({
                                        "id": thread_tweet.id,
                                        "text": thread_tweet.text
                                    })
                                    result["text"] += f"\n- {thread_tweet.text}"
                        except Exception as thread_error:
                            logger.warning(f"Could not fetch thread tweets: {str(thread_error)}")
                else:
                    result["text"] = f"Tweet by @{user_screen_name} (ID: {tweet_id}) could not be retrieved."
                    result["error"] = "Tweet data not available"
                    
            except Exception as e:
                logger.error(f"Error fetching tweet with Tweepy: {str(e)}")
                result["error"] = f"Tweepy error: {str(e)}"
                # Fall back to web scraping
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as response:
                            if response.status == 200:
                                html = await response.text()
                                soup = BeautifulSoup(html, 'html.parser')
                                
                                # Extract tweet content through web scraping
                                tweet_texts = []
                                
                                # Find article elements (tweets)
                                articles = soup.find_all('article')
                                if articles:
                                    for article in articles:
                                        paragraphs = article.find_all(['p', 'span'])
                                        for p in paragraphs:
                                            if p.text and len(p.text.strip()) > 10:
                                                tweet_texts.append(p.text.strip())
                                
                                # Try additional selectors if needed
                                if not tweet_texts:
                                    main_elements = soup.select('[data-testid="tweetText"]')
                                    if main_elements:
                                        for element in main_elements:
                                            tweet_texts.append(element.get_text(strip=True))
                                
                                result["text"] = f"Tweet by @{user_screen_name} (ID: {tweet_id}):\n\n"
                                if tweet_texts:
                                    result["text"] += "\n\n".join(tweet_texts[:5])
                                else:
                                    result["text"] += "Content could not be extracted."
                except Exception as scrape_error:
                    logger.error(f"Fallback web scraping for tweet failed: {str(scrape_error)}")
                    result["text"] = f"Tweet by @{user_screen_name} (ID: {tweet_id}).\nError extracting content."
        
        # Handle GitHub URLs with PyGithub
        elif (github_repo_match or github_issue_match) and GITHUB_TOKEN:
            result["is_github"] = True
            try:
                # Initialize GitHub client
                g = Github(GITHUB_TOKEN)
                
                if github_repo_match:
                    # Handle GitHub repository
                    owner = github_repo_match.group(1)
                    repo_name = github_repo_match.group(2)
                    branch = github_repo_match.group(3) if github_repo_match.group(3) else None
                    
                    result["repo_info"] = {
                        "owner": owner,
                        "repo": repo_name,
                        "branch": branch,
                        "type": "repository"
                    }
                    
                    # Get repository
                    repo = g.get_repo(f"{owner}/{repo_name}")
                    
                    # Build repository information
                    result["text"] = f"GitHub Repository: {owner}/{repo_name}\n\n"
                    result["text"] += f"Description: {repo.description or 'No description'}\n"
                    result["text"] += f"Stars: {repo.stargazers_count} | Forks: {repo.forks_count}\n"
                    result["text"] += f"Language: {repo.language or 'Not specified'}\n"
                    result["text"] += f"Created: {repo.created_at}\n"
                    result["text"] += f"Last updated: {repo.updated_at}\n\n"
                    
                    # Try to get README content
                    try:
                        readme = repo.get_readme()
                        if readme:
                            # Decode README content
                            readme_content = base64.b64decode(readme.content).decode('utf-8')
                            
                            # Limit README content to a reasonable size
                            if len(readme_content) > 500:
                                readme_content = readme_content[:500] + "... (content truncated)"
                                
                            result["text"] += f"README:\n{readme_content}\n\n"
                    except Exception as readme_error:
                        logger.warning(f"Could not fetch README: {str(readme_error)}")
                        
                    # List top-level files
                    try:
                        contents = repo.get_contents("")
                        result["text"] += "Repository contents:\n"
                        
                        for content in contents[:15]:  # Limit to 15 items
                            result["text"] += f"- {content.name} ({content.type})\n"
                            
                        if len(contents) > 15:
                            result["text"] += f"... and {len(contents) - 15} more files/directories"
                    except Exception as contents_error:
                        logger.warning(f"Could not fetch repository contents: {str(contents_error)}")
                
                elif github_issue_match:
                    # Handle GitHub issue or PR
                    owner = github_issue_match.group(1)
                    repo_name = github_issue_match.group(2)
                    issue_number = int(github_issue_match.group(3))
                    
                    # Determine if it's an issue or PR
                    is_pr = "pull" in url
                    item_type = "Pull Request" if is_pr else "Issue"
                    
                    result["repo_info"] = {
                        "owner": owner,
                        "repo": repo_name,
                        "issue_number": issue_number,
                        "type": item_type.lower()
                    }
                    
                    # Get repository
                    repo = g.get_repo(f"{owner}/{repo_name}")
                    
                    # Get issue or PR
                    issue = repo.get_issue(issue_number)
                    
                    # Build issue/PR information
                    result["text"] = f"GitHub {item_type}: {owner}/{repo_name}#{issue_number}\n\n"
                    result["text"] += f"Title: {issue.title}\n"
                    result["text"] += f"State: {issue.state}\n"
                    result["text"] += f"Author: {issue.user.login}\n"
                    result["text"] += f"Created: {issue.created_at}\n"
                    result["text"] += f"Last updated: {issue.updated_at}\n\n"
                    
                    # Add body content
                    if issue.body:
                        body = issue.body
                        if len(body) > 500:
                            body = body[:500] + "... (content truncated)"
                        result["text"] += f"Description:\n{body}\n\n"
                    else:
                        result["text"] += "No description provided.\n\n"
                    
                    # If it's a PR, add PR-specific information
                    if is_pr:
                        try:
                            pr = repo.get_pull(issue_number)
                            result["text"] += f"Source Branch: {pr.head.ref}\n"
                            result["text"] += f"Target Branch: {pr.base.ref}\n"
                            result["text"] += f"Mergeable: {pr.mergeable}\n"
                            result["text"] += f"Additions: {pr.additions} | Deletions: {pr.deletions}\n"
                            result["text"] += f"Changed Files: {pr.changed_files}\n\n"
                        except Exception as pr_error:
                            logger.warning(f"Could not fetch PR details: {str(pr_error)}")
                    
                    # Try to get comments
                    try:
                        comments = issue.get_comments()
                        comment_count = comments.totalCount
                        
                        if comment_count > 0:
                            result["text"] += f"Comments: {comment_count}\n\n"
                            result["text"] += "Recent comments:\n"
                            
                            # Get up to 5 most recent comments
                            recent_comments = [comment for comment in comments.reversed][:5]
                            
                            for comment in recent_comments:
                                comment_text = comment.body
                                if len(comment_text) > 200:
                                    comment_text = comment_text[:200] + "..."
                                    
                                result["text"] += f"- {comment.user.login}: {comment_text}\n"
                    except Exception as comments_error:
                        logger.warning(f"Could not fetch comments: {str(comments_error)}")
            
            except Exception as e:
                logger.error(f"Error fetching GitHub content with PyGithub: {str(e)}")
                result["error"] = f"PyGithub error: {str(e)}"
                
                # Fall back to GitHub API directly
                try:
                    if github_repo_match:
                        owner = github_repo_match.group(1)
                        repo = github_repo_match.group(2)
                        
                        async with aiohttp.ClientSession() as session:
                            repo_url = f"https://api.github.com/repos/{owner}/{repo}"
                            async with session.get(repo_url, headers={"Accept": "application/vnd.github.v3+json"}) as response:
                                if response.status == 200:
                                    repo_data = await response.json()
                                    result["text"] = f"GitHub Repository: {owner}/{repo}\n\n"
                                    result["text"] += f"Description: {repo_data.get('description', 'No description')}\n"
                                    result["text"] += f"Stars: {repo_data.get('stargazers_count', 0)} | Forks: {repo_data.get('forks_count', 0)}\n"
                                else:
                                    result["text"] = f"GitHub Repository: {owner}/{repo}\n\nCould not fetch repository details."
                    
                    elif github_issue_match:
                        owner = github_issue_match.group(1)
                        repo = github_issue_match.group(2)
                        issue_number = github_issue_match.group(3)
                        is_pr = "pull" in url
                        item_type = "Pull Request" if is_pr else "Issue"
                        
                        async with aiohttp.ClientSession() as session:
                            issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
                            async with session.get(issue_url, headers={"Accept": "application/vnd.github.v3+json"}) as response:
                                if response.status == 200:
                                    issue_data = await response.json()
                                    result["text"] = f"GitHub {item_type}: {owner}/{repo}#{issue_number}\n\n"
                                    result["text"] += f"Title: {issue_data.get('title', f'{item_type} #{issue_number}')}\n"
                                    result["text"] += f"State: {issue_data.get('state', 'unknown')}\n\n"
                                    
                                    body = issue_data.get('body', 'No description provided')
                                    if len(body) > 500:
                                        body = body[:500] + "... (content truncated)"
                                    result["text"] += f"Description:\n{body}"
                                else:
                                    result["text"] = f"GitHub {item_type}: {owner}/{repo}#{issue_number}\n\nCould not fetch details."
                
                except Exception as api_error:
                    logger.error(f"GitHub API fallback failed: {str(api_error)}")
                    result["text"] = f"GitHub content for {url}.\nError extracting content."
        
        # Handle regular URLs
        else:
            result["is_public_url"] = False
            # Non-specialized URL or missing tokens, use web scraping
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Get page title
                        title_tag = soup.find('title')
                        if title_tag:
                            result["title"] = title_tag.get_text(strip=True)
                        
                        # Try to get main content
                        main_content = soup.find('main') or soup.find(id='content') or soup.find(class_='content') or soup.find('article')
                        
                        if main_content:
                            # Get text from main content
                            content_text = main_content.get_text(separator=' ', strip=True)
                            content_text = re.sub(r'\s+', ' ', content_text)
                            result["text"] = content_text[:1000]  # Limit to 500 chars
                        else:
                            # Get text from body if main content not found
                            body_text = soup.get_text(separator=' ', strip=True)
                            body_text = re.sub(r'\s+', ' ', body_text)
                            result["text"] = body_text[:1000]  # Limit to 5000 chars
                            
                        # Add title as context if available
                        if result["title"]:
                            result["text"] = f"Title: {result['title']}\n\n{result['text']}"
                    else:
                        result["error"] = f"HTTP error: {response.status}"
                        
        return result
                    
    except Exception as e:
        logger.error(f"Error in fetch_url_content: {str(e)}")
        return {
            "url": url,
            "text": f"Error fetching content: {str(e)}",
            "error": f"Error in fetch_url_content: {str(e)}",
            "is_tweet": False,
            "is_github": False,
            "is_public_url": False
        }

async def process_mention_request(ai_request: AIRequest, thread_ts: str):
    """
    Process a bot mention from Slack and generate a response.
    
    Args:
        ai_request: The AI request object containing user, channel, text
        thread_ts: The thread timestamp to respond in
    """
    start_time = time.time()
    logger.info(f"Processing mention request from user {ai_request.user_id} in channel {ai_request.channel_id}")
    
    try:
        # Check rate limiting
        if not check_rate_limit():
            try:
                slack_client.chat_update(
                    channel=ai_request.channel_id,
                    ts=ai_request.message_ts,
                    text="⚠️ Rate limit exceeded. Please try again in a minute."
                )
            except SlackApiError as e:
                logger.error(f"Error updating message with rate limit notice: {e.response['error']}")
            return
        
        # Get conversation history
        conversation_key = f"{ai_request.channel_id}:{thread_ts}"

        #parse response to return the correct user id
        ai_request.text = parse_user_mentions(ai_request.text)

        # Obtain the user's display name
        sender_name = "User"  # Default fallback
        try:
            user_info = slack_client.users_info(user=ai_request.user_id)
            if user_info.get("ok") and user_info.get("user"):
                sender_name = user_info["user"].get("real_name") or user_info["user"].get("name") or "User"
        except SlackApiError as e:
            logger.warning(f"Could not get user name, using default: {e.response['error']}")

        # Try loading from in-memory cache first
        if conversation_key in conversation_history and conversation_history[conversation_key]:
            logger.info(f"Using in-memory conversation history with {len(conversation_history[conversation_key])} messages")
        else:
            # If not in memory, try to load from database or rebuild from Slack
            logger.info(f"Conversation not in memory, attempting to load from database")
            db_messages = load_conversation_from_db(ai_request.channel_id, thread_ts)
            
            if db_messages:
                # If found in database, update in-memory cache
                conversation_history[conversation_key] = db_messages
                logger.info(f"Loaded conversation history from database with {len(db_messages)} messages")
            else:
                # If this is part of a thread, try to rebuild the history from Slack API
                if thread_ts and (conversation_key not in conversation_history or not conversation_history[conversation_key]):
                    logger.info(f"Rebuilding conversation history from Slack for thread {thread_ts}")
                    
                    try:
                        # Get the thread messages
                        response = slack_client.conversations_replies(
                            channel=ai_request.channel_id,
                            ts=thread_ts
                        )
                        
                        if response.get("ok") and response.get("messages"):
                            # Process the messages
                            messages = response.get("messages", [])
                            conversation_history[conversation_key] = []

                            #no idea why the process method sends the first message 2 times
                            messages = messages[1:]
                            for msg in messages:
                                # Determine if it's a user or bot message
                                is_bot = msg.get("bot_id") is not None
                                text = msg.get("text", "")
                                
                                # Skip empty messages
                                if not text:
                                    continue
                                    
                                # For bot messages, remove "TMAI's neuron firing......" and other status indicators
                                if is_bot:
                                    if "TMAI's neuron firing..." in text or "is thinking" in text:
                                        continue                                
                                # Add to conversation history
                                add_message_to_conversation(
                                    conversation_key,
                                    "assistant" if is_bot else "user",
                                    text,
                                    msg.get("ts")
                                )
                            
                            logger.info(f"Rebuilt conversation history with {len(conversation_history[conversation_key])} messages")
                        else:
                            logger.warning(f"Failed to get thread replies: {response.get('error', 'Unknown error')}")
                    except SlackApiError as e:
                        logger.error(f"Error getting thread replies: {e.response['error']}")
        
        # Build context for AI - this is just formatting, NOT storage
        context_parts = []
        context_parts.append(f"Current {sender_name} query: {parse_user_mentions(ai_request.text)}\n")

        # Format conversation history for context
        history_context = []
        
        # Add conversation history if it exists
        if conversation_key in conversation_history and conversation_history[conversation_key]:
            # Limit to 5 most recent messages
            messages = conversation_history[conversation_key]
            if len(messages) > 20:
                messages = messages[-20:]  # Keep only the last 10 messages
                history_context = [f"**Conversation History:** Showing only the 10 most recent messages."]
            else:
                history_context = [f"**Here is the conversation history between you and {sender_name} so far:**"]
            
            # Format the messages for context (this is just for display, not storage)
            for i, msg in enumerate(messages):
                if msg["role"] == "user":
                    # Format user messages with sender name
                    msg_content = parse_user_mentions(msg['content'])
                    history_context.append(f"**{sender_name} (#{i} turn):** {msg_content}")
                else:
                    # Format assistant messages
                    content = msg["content"]
                    if len(content) > 500:
                        content = content[:500] + "... (content truncated)"
                    history_context.append(f"**Assistant (#{i - 1} turn):** {content}")

        logger.info(f"History context throughout the conversation: {' '.join(history_context)}")

        # Add the raw message to conversation history (only once!) Add here to prevent the current message from being added to the history
        add_message_to_conversation(conversation_key, "user", parse_user_mentions(ai_request.text), ai_request.message_ts)
        
        # Step 1: Analyze the content to determine what we need to do
        try:
            content_analysis = await analyze_content(ai_request.text, "\n".join(history_context), sender_name=sender_name)
            logger.info(f"Content analysis: {content_analysis}")
            
            # Manual check for Linear issue creation intent
            if not content_analysis.create_linear_issue:
                # Keywords that strongly indicate issue creation intent
                issue_creation_keywords = [
                    "create issue", "create a issue", "create an issue", 
                    "make issue", "make a issue", "make an issue",
                    "new issue", "add issue", "add a issue", "add an issue",
                    "create task", "create a task", "new task", "add task",
                    "create ticket", "make ticket", "new ticket",
                    "title", "title is", "title:", "titled"
                ]
                
                # Check for title/description patterns
                title_desc_pattern = re.search(r'title[:\s]+(.*?)(?:\s*description[:\s]+|\s*$)', ai_request.text, re.IGNORECASE)
                
                # Check for keywords
                has_creation_keyword = any(keyword.lower() in ai_request.text.lower() for keyword in issue_creation_keywords)
                
                # Set create_linear_issue to True if any pattern matches
                if has_creation_keyword or title_desc_pattern:
                    logger.info("Manual detection found Linear issue creation intent")
                    content_analysis.create_linear_issue = True
                    content_analysis.content_type = "prompt_requires_tool_use"
            
        except Exception as e:
            error_msg = f"Error analyzing content: {str(e)}"
            logger.error(error_msg)
            context_parts.append(f"\n❌ {error_msg}")
            # Set default content analysis
            content_analysis = ContentAnalysisResult()
            content_analysis.content_type = "simple_prompt"
        
        # Step 2: Process the request based on content analysis
        
        # If we need to search Slack channel history
        if content_analysis.requires_slack_channel_history:
            try:
                # Update message to show slack search stage
                try:
                    slack_client.chat_update(
                        channel=ai_request.channel_id,
                        text="TMAI's neuron firing",
                        ts=ai_request.message_ts,
                        blocks=[
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"\n*Status:*\n```\n✓ Query analyzed\n➤ Searching Slack history...\n```"
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
                
                # Initialize search parameters with defaults
                search_params = {
                    "username": None,
                    "time_range": "days",
                    "time_value": 7,
                    "message_count": 50
                }
                
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
                    # Fix: Add the joined string, not the list itself
                    context_parts.append("\n".join(search_context))
                    
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
                    
                    # After retrieving relevant messages from Slack search
                    if len(relevant_messages) > 15:  # If too many messages
                        # Keep first 5 and last 10 most relevant messages
                        summary = f"Found {len(relevant_messages)} messages, showing most relevant examples."
                        relevant_messages = relevant_messages[:5] + ["..."] + relevant_messages[-10:]
                        context_parts.append(summary)
                        # Fix: Join the list of messages into a string before appending
                        context_parts.append("Relevant channel messages:\n" + "\n".join(relevant_messages))
                    else:
                        # Fix: Join the list of messages into a string before appending
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
                    linear_stage_text += "➤ Searching Linear...\n```"
                    
                    slack_client.chat_update(
                        channel=ai_request.channel_id,
                        text="TMAI's neuron firing",
                        ts=ai_request.message_ts,
                        blocks=[
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"\n{linear_stage_text}"
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
                logger.info(f"URL results: {url_results}")
                
                # Add any errors first
                if url_errors:
                    context_parts.append("\n❌ URL Processing Errors:")
                    for error in url_errors:
                        context_parts.append(f"- {error}")
                    context_parts.append("")  # Empty line after errors

                for result in url_results:
                    context_parts.append(f"\nURL: {result['url']}")
                    context_parts.append(f"Title: {result['title']}")
                    context_parts.append(f"Text: {result['text'][:500]}...(content truncated)")

        # if content_analysis.create_linear_issue == False:
        #     content_analysis.create_linear_issue = True
        if content_analysis.create_linear_issue == False:
            # Check if the query contains keywords that suggest creating a Linear issue
            issue_creation_keywords = [
                "create issue", "create a issue", "create an issue", 
                "make issue", "make a issue", "make an issue",
                "new issue", "add issue", "add a issue", "add an issue",
                "create task", "create a task", "new task", "add task",
                "create ticket", "make ticket", "new ticket",
                "title", "title is", "title:", "titled"
            ]
            
            # Convert to lowercase for case-insensitive matching
            text_lower = content_analysis.text.lower()
            
            # Check if any of the keywords are in the text
            if any(keyword in text_lower for keyword in issue_creation_keywords):
                content_analysis.create_linear_issue = True
                logger.info(f"Manually overriding create_linear_issue to True based on keyword detection")

        if content_analysis.create_linear_issue:
            try:
                create_issue_text = "*Status:*\n```\n✓ Query analyzed\n"
                create_issue_text += "➤Crafting your Linear issue...\n```"
                try:
                    slack_client.chat_update(
                        channel=ai_request.channel_id,
                        text="TMAI's neuron firing",
                        ts=ai_request.message_ts,
                        blocks=[
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"\n{create_issue_text}"
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
                response_stage_text += "✓ Linear searched\n"
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
                text="TMAI's neuron firing",
                ts=ai_request.message_ts,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"\n{response_stage_text}"
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
            )
        except SlackApiError as e:
            logger.warning(f"Could not update processing message: {e.response.get('error', '')}")
        
        # Step 3: Call AI with the full context including conversation history
        logger.info("Building final context for AI call with conversation history")
        # Ensure all context parts are strings before joining
        for i in range(len(context_parts)):
            if not isinstance(context_parts[i], str):
                logger.warning(f"Non-string found in context_parts at index {i}: {type(context_parts[i])}")
                context_parts[i] = str(context_parts[i])
        
        full_context = "\n".join(context_parts)
        
        # Get the message with streaming for a better user experience
        message_ts = await stream_ai_response(full_context, history_context, ai_request, thread_ts, sender_name=sender_name)
        
            
        
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
    # First check global rate limit
    if not global_limiter.check_rate_limit():
        logger.warning("Global rate limit exceeded")
        return False
    
    # Request is allowed
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
    
    # Use AI classification with our prompt from YAML
    try:
        # Apply rate limiting for OpenAI
        if not openai_limiter.check_rate_limit():
            logger.warning("OpenAI rate limit exceeded, waiting...")
            openai_limiter.wait_if_needed()
        
        # Get the prompt templates from YAML
        system_prompt = PROMPTS.get("analyze_content", {}).get("system_template", "")
        user_prompt_template = PROMPTS.get("analyze_content", {}).get("user_template", "")
    
        # Format the user prompt with properly joined history context
        # history_text = "\n".join(history_context) if history_context else "No previous conversation"
        user_prompt = user_prompt_template.format(
            slack_users=slack_users,
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
                response_format={"type": "json_object"},
                temperature=0.7
            )
        elif AI_MODEL.startswith("o"):
            prompt = system_prompt + "\n\n" + user_prompt
            logger.info(f"User prompt at analyze_content: {prompt}")
            response = client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
            )
        
        # Parse the response
        try:
            raw_analysis = response.choices[0].message.content
            
            # Parse the raw response into a JSON object
            analysis = json.loads(raw_analysis)
            
            # Check if required fields exist in the analysis
            if "content_type" not in analysis:
                logger.error(f"Missing required field 'content_type' in AI response: {analysis}")
                raise KeyError("Missing required field 'content_type' in AI response")
            
            # Update the result with the AI analysis
            result.content_type = analysis.get("content_type", "simple_prompt")
            result.requires_slack_channel_history = analysis.get("requires_slack_channel_history", False)
            result.perform_RAG_on_linear_issues = analysis.get("perform_RAG_on_linear_issues", False)
            
            # If URLs were identified in the analysis, update our list
            if "urls" in analysis and analysis["urls"]:
                result.urls = analysis["urls"]
            
            return result
        except json.JSONDecodeError as je:
            logger.error(f"JSON parsing error: {str(je)}")
            logger.error(f"Problematic content: {raw_analysis}")
            # Fallback to simple_prompt for any error
            result.content_type = "simple_prompt"
            result.requires_slack_channel_history = False
            result.perform_RAG_on_linear_issues = False
            logger.info("Falling back to simple_prompt type due to JSON parsing error")
            return result
        except KeyError as ke:
            logger.error(f"Key error in AI response: {str(ke)}")
            # Fallback to simple_prompt for any error
            result.content_type = "simple_prompt"
            result.requires_slack_channel_history = False
            result.perform_RAG_on_linear_issues = False
            logger.info("Falling back to simple_prompt type due to missing key in response")
            return result
        
    except Exception as e:
        logger.error(f"Error in AI content analysis: {str(e)}", exc_info=True)
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
            system_prompt = PROMPTS.get("slack_search_operator", {}).get("system_template", "")
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
                        response_format={"type": "json_object"},
                        temperature=0.7
                    )
                elif AI_MODEL.startswith("o"):
                    prompt = system_prompt + "\n\n" + user_prompt
                    response = client.chat.completions.create(
                        model=AI_MODEL,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                    )
        
                # Parse the response
                ai_params = json.loads(response.choices[0].message.content)
                
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
    Perform semantic search over Linear issues and comments.
    """
    # Apply rate limiting for Linear API
    if not linear_limiter.check_rate_limit():
        logger.warning("Linear API rate limit exceeded, waiting...")
        linear_limiter.wait_if_needed()
        
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
            system_prompt = PROMPTS.get("linear_search_operator", {}).get("system_template", "")
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
                
                logger.info(f"Sending this user prompt to OpenAI: {user_prompt}")

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
                            response_format={"type": "json_object"},
                            temperature=0.7
                        )
                    elif AI_MODEL.startswith("o"):
                        prompt = system_prompt + "\n\n" + user_prompt
                        response = client.chat.completions.create(
                            model=AI_MODEL,
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
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


async def generate_linear_issue(query: str, history_context: str, sender_name: str) -> Dict[str, Any]:
    """
    Generate Linear issue based on the user's query.
    """
    try:
        # Apply OpenAI rate limiting
        if not openai_limiter.check_rate_limit():
            logger.warning("OpenAI rate limit exceeded for issue generation, waiting...")
            openai_limiter.wait_if_needed()
            
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
                response_format={"type": "json_object"},
                temperature=0.7
            )
        elif AI_MODEL.startswith("o"):
            prompt = create_issue_system_prompt + "\n\n" + create_issue_user_prompt
            response = client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
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

        # Apply Linear rate limiting before creating issue
        if not linear_limiter.check_rate_limit():
            logger.warning("Linear API rate limit exceeded for issue creation, waiting...")
            linear_limiter.wait_if_needed()
            
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

async def stream_ai_response(context: str, history_context: str, ai_request: AIRequest, thread_ts: str, sender_name: str) -> Optional[str]:
    try:
        # Use existing message_ts from the AI request instead of creating a new message
        message_ts = ai_request.message_ts

        # Get system prompts
        draft_system_prompt = PROMPTS.get("draft_response", {}).get("system_template", "").format(
            slack_users=slack_users,
            sender_name=sender_name  # Add sender_name parameter to the format() call
        )
        if not draft_system_prompt:
            draft_system_prompt = "You are an AI assistant for Token Metrics."

        draft_user_prompt = PROMPTS.get("draft_response", {}).get("user_template", "").format(
            sender_name=sender_name,
            conversation_history=history_context,
            context=context
        )
        if not draft_user_prompt:
            draft_user_prompt = "Here's the history so far between you and {sender_name}: {conversation_history}"

        final_system_prompt = PROMPTS.get("final_response", {}).get("system_template", "")
        if not final_system_prompt:
            final_system_prompt = "Refine the draft response for Slack formatting."

        # Get conversation history for this thread
        conversation_key = f"{ai_request.channel_id}:{thread_ts}"
        messages = []
        
        # Add system message for draft
        messages.append({"role": "system", "content": draft_system_prompt})  
        messages.append({"role": "user", "content": draft_user_prompt})

        # STEP 1: Generate draft response with OpenAI
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Apply OpenAI rate limiting
        if not openai_limiter.check_rate_limit():
            logger.warning("OpenAI rate limit exceeded for draft generation, waiting...")
            openai_limiter.wait_if_needed()
        
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
        
        # Apply OpenAI rate limiting again for the refinement
        if not openai_limiter.check_rate_limit():
            logger.warning("OpenAI rate limit exceeded for response refinement, waiting...")
            openai_limiter.wait_if_needed()
        
        # Stream the final response
        full_response = ""
        
        # Stream the refined response
        stream = client.chat.completions.create(
            model="gpt-4o-mini", #this is easy, should be fast
            messages=final_messages,
            stream=True
        )
        
        # Apply Slack rate limiting before posting message
        if not slack_limiter.check_rate_limit():
            logger.warning("Slack API rate limit exceeded for message posting, waiting...")
            slack_limiter.wait_if_needed()
        
        # Create initial message
        initial_response = slack_client.chat_postMessage(
            channel=ai_request.channel_id,
            thread_ts=thread_ts,
            text="..."  # Will be updated with content
        )
        current_message_ts = initial_response["ts"]
        
        buffer = ""
        last_update_time = time.time()
        first_chunk = True
        current_part = ""
        part_number = 1
        
        for chunk in stream:
            current_time = time.time()
            
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                current_part += content
                buffer += content
                
                # Delete the acknowledge message as soon as we receive first chunk
                if first_chunk:
                    first_chunk = False
                    try:
                        if not slack_limiter.check_rate_limit():
                            logger.warning("Slack API rate limit exceeded for message deletion, waiting...")
                            slack_limiter.wait_if_needed()
                            
                        # Try to delete the processing message
                        slack_client.chat_delete(
                            channel=ai_request.channel_id,
                            ts=message_ts
                        )
                        logger.info(f"Deleted processing message with ts: {message_ts}")
                    except SlackApiError as e:
                        logger.warning(f"Could not delete processing message: {e.response.get('error', '')}")
                
                # Check if current part is approaching Slack's limit (3500 chars to be safe)
                if len(current_part) > 3500:
                    # Start a new part
                    part_number += 1
                    try:
                        if not slack_limiter.check_rate_limit():
                            logger.warning("Slack API rate limit exceeded for new message creation, waiting...")
                            slack_limiter.wait_if_needed()
                        
                        # Create a new message for the next part
                        new_message = slack_client.chat_postMessage(
                            channel=ai_request.channel_id,
                            thread_ts=thread_ts,
                            text=f"{part_number}"
                        )
                        current_message_ts = new_message["ts"]
                        current_part = ""  # Reset for new part
                    except SlackApiError as e:
                        logger.warning(f"Failed to create new message part: {e.response.get('error', '')}")
                
                # Update message when buffer is full or time has elapsed
                if len(buffer) > 1000 or current_time - last_update_time > 2.0:
                    try:
                        if not slack_limiter.check_rate_limit():
                            logger.warning("Slack API rate limit exceeded for message update, waiting...")
                            await asyncio.sleep(1)
                        
                        # Update the current message with accumulated part
                        formatted_text = format_for_slack(current_part)
                        if part_number > 1:
                            formatted_text = f"\n{formatted_text}"
                            
                        slack_client.chat_update(
                            channel=ai_request.channel_id,
                            ts=current_message_ts,
                            text=formatted_text
                        )
                        
                        buffer = ""
                        last_update_time = current_time
                    except SlackApiError as e:
                        logger.warning(f"Failed to update streaming message: {e.response.get('error', '')}")
        
        # Send final update if there's anything in the buffer
        if buffer:
            try:
                if not slack_limiter.check_rate_limit():
                    logger.warning("Slack API rate limit exceeded for final update, waiting...")
                    await asyncio.sleep(1)
                
                # Final update with any remaining content
                formatted_text = format_for_slack(current_part)
                if part_number > 1:
                    formatted_text = f"\n{formatted_text}"
                    
                slack_client.chat_update(
                    channel=ai_request.channel_id,
                    ts=current_message_ts,
                    text=formatted_text
                )
            except SlackApiError as e:
                logger.warning(f"Failed to send final update: {e.response.get('error', '')}")
        
        # Store the complete response in conversation history
        add_message_to_conversation(conversation_key, "assistant", full_response, current_message_ts)
        
        return full_response
        
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
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
            issue_id = issue.get('id', 'No ID')
            context_parts.append(f"{i}. {issue.get('title', 'Untitled')} ({issue_id})")
            context_parts.append(f"Description: {issue.get('description', '')}")
            
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


# Add this function
def check_app_health():
    """Check critical components of the app at startup"""
    # Check database connection
    db_ok = check_db_connection()
    logger.info(f"Database connection check: {'OK' if db_ok else 'FAILED'}")
    
    if not db_ok:
        logger.warning("WARNING: Database connection failed! Conversations will only be stored in memory.")
    
    # Add other health checks as needed
    
@app.on_event("startup")
async def startup_event():
    """Run tasks when the app starts up"""
    # Initialize the database if needed
    init_db()
    
    # Schedule periodic database cleanup
    async def run_db_cleanup():
        while True:
            try:
                # Wait 6 hours between cleanup runs
                await asyncio.sleep(6 * 60 * 60)  # 6 hours in seconds
                
                # Clean up conversations older than 24 hours
                removed_count = cleanup_old_conversations(hours=24)
                logger.info(f"Scheduled cleanup completed: removed {removed_count} conversations")
                
                # Also clean in-memory cache
                clean_expired_conversations()
            except Exception as e:
                logger.error(f"Error in scheduled cleanup: {str(e)}")
    
    # Start the background task
    asyncio.create_task(run_db_cleanup())
    logger.info("Scheduled database cleanup task started")

# Call this function before starting the app
if __name__ == "__main__":
    # Run health checks
    check_app_health()
    
    # Start the app
    uvicorn.run(app, host="0.0.0.0", port=8000)