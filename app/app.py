"""
TMAI Slack Bot - FastAPI Application
Handles Slack events, command endpoints, and manages the agent lifecycle.
"""

import os
import sys
import logging
import yaml
import asyncio
import json
from typing import Dict, List, Any, Optional

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from slack_sdk.errors import SlackApiError

from conversation_db import init_db, check_db_connection, cleanup_old_conversations
from rate_limiter import global_limiter
from TMAI_slack_agent import TMAISlackAgent, AIRequest

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            os.path.join(logs_dir, "TMAI_slack_agent.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5, 
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("slack_ai_app")
logging.getLogger("openai_client").setLevel(logging.DEBUG)
logging.getLogger("tmai_agent").setLevel(logging.DEBUG)
# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Get required environment variables
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_USER_TOKEN = os.environ.get("SLACK_USER_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
AI_MODEL = os.environ.get("AI_MODEL", "o3-mini")

if not SLACK_BOT_TOKEN:
    logger.error("SLACK_BOT_TOKEN not found in environment variables")
    raise ValueError("SLACK_BOT_TOKEN is required")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY is required")

# Load prompts from YAML
try:
    with open('prompts.yaml', 'r', encoding='utf-8') as file:
        PROMPTS = yaml.safe_load(file)
    logger.info("Successfully loaded prompts from prompts.yaml")
except Exception as e:
    logger.error(f"Error loading prompts: {str(e)}")
    PROMPTS = {}

# Initialize the TMaiSlackAgent
agent = TMAISlackAgent(
    slack_bot_token=SLACK_BOT_TOKEN,
    slack_user_token=SLACK_USER_TOKEN,
    openai_api_key=OPENAI_API_KEY,
    ai_model=AI_MODEL,
    prompts=PROMPTS
)

# Create FastAPI app
app = FastAPI(title="TMAI Slack Assistant")

@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    """Handle Slack events (mentions, direct messages, etc.)"""
    logger.info("Slack events endpoint called")
    
    try:
        payload = await request.json()
        
        # Handle URL verification
        if payload.get("type") == "url_verification":
            return {"challenge": payload["challenge"]}
        
        # Process normal events
        if payload.get("type") == "event_callback":
            event = payload.get("event", {})
            event_type = event.get("type")
            event_id = payload.get("event_id", "unknown")
            channel_type = event.get("channel_type", "")
            
            logger.info(f"Processing event ID: {event_id}, Type: {event_type}")
            
            # Skip message subtypes (like bot messages, updates, etc.)
            if event.get("subtype"):
                return {}
            
            # Handle direct messages to the bot
            if event_type == "message" and channel_type == "im":
                # Skip if no text content or if it's a bot message or processing message
                text = event.get("text", "")
                if not text.strip() or event.get("bot_id") or "TMAI's neuron firing..." in text:
                    return {}
                
                # Extract required data
                user_id = event.get("user", "")
                channel_id = event.get("channel", "")
                message_ts = event.get("ts", "")
                thread_ts = event.get("thread_ts", message_ts)
                
                # Get sender name
                sender_name = "User"  # Default fallback
                try:
                    user_info = agent.slack_client.users_info(user=user_id)
                    if user_info.get("ok") and user_info.get("user"):
                        sender_name = user_info["user"].get("real_name") or user_info["user"].get("name") or "User"
                except SlackApiError as e:
                    logger.warning(f"Could not get user name, using default: {e.response['error']}")
                
                logger.info(f"Direct message from {sender_name}: {text}")
                
                # Create AI request
                ai_request = AIRequest(
                    text=text,
                    user_id=user_id,
                    channel_id=channel_id,
                    message_ts=message_ts,
                    thread_ts=thread_ts,
                    sender_name=sender_name
                )
                
                # Send initial processing message
                try:
                    processing_msg = await asyncio.to_thread(
                        agent.slack_client.chat_postMessage,
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
                                        "text": "Harnessing the power of a trillion artificial neurons... and slightly fewer from my creator ◐"
                                    }
                                ]
                            }
                        ]
                    )
                    
                    # Update the AI request with processing message timestamp
                    ai_request.message_ts = processing_msg.get("ts")
                except SlackApiError as e:
                    logger.error(f"Error posting initial DM message: {e.response['error']}")
                
                # Process in background
                background_tasks.add_task(
                    agent.process_slack_message,
                    ai_request,
                    is_direct_message=True
                )
                
            # Handle bot mentions
            elif event_type == "app_mention":
                logger.info(f"Bot was mentioned in channel {event.get('channel')}")
                
                # Extract required data
                user_id = event.get("user", "")
                channel_id = event.get("channel", "")
                text = event.get("text", "")
                message_ts = event.get("ts", "")
                thread_ts = event.get("thread_ts", message_ts)
                
                # Remove bot mention from text (matches <@BOTID>)
                text = text.replace(f"<@{os.environ.get('SLACK_BOT_ID', 'U08GNQ8F2RH')}>", "").strip()
                
                # Get sender name
                sender_name = "User"  # Default fallback
                try:
                    user_info = agent.slack_client.users_info(user=user_id)
                    if user_info.get("ok") and user_info.get("user"):
                        sender_name = user_info["user"].get("real_name") or user_info["user"].get("name") or "User"
                except SlackApiError as e:
                    logger.warning(f"Could not get user name, using default: {e.response['error']}")
                
                # Create AI request
                ai_request = AIRequest(
                    text=text,
                    user_id=user_id,
                    channel_id=channel_id,
                    message_ts=message_ts,
                    sender_name=sender_name
                )
                
                # Send initial processing message
                try:
                    initial_response = await asyncio.to_thread(
                        agent.slack_client.chat_postMessage,
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
                    
                    # Update the AI request with processing message timestamp
                    ai_request.message_ts = initial_response.get("ts")
                except SlackApiError as e:
                    logger.error(f"Error posting initial message: {e.response['error']}")
                
                # Process in background
                background_tasks.add_task(
                    agent.process_slack_message,
                    ai_request,
                    thread_ts=thread_ts,
                    is_direct_message=False
                )
        
        # Return an empty 200 response to acknowledge receipt
        return {}
        
    except Exception as e:
        logger.error(f"Error processing Slack event: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/slack/ai_command")
async def ai_command(request: Request):
    """Handle Slack slash commands."""
    return {"message": "AI command received"}

@app.on_event("startup")
async def startup_event():
    """Run tasks when the app starts up."""
    # Initialize the database
    init_db()
    
    # Check database connection
    db_ok = check_db_connection()
    logger.info(f"Database connection check: {'OK' if db_ok else 'FAILED'}")
    
    if not db_ok:
        logger.warning("WARNING: Database connection failed! Conversations will only be stored in memory.")
    
    # Schedule periodic database cleanup
    async def run_db_cleanup():
        while True:
            try:
                # Wait 6 hours between cleanup runs
                await asyncio.sleep(6 * 60 * 60)  # 6 hours in seconds
                
                # Clean up conversations older than 24 hours
                removed_count = cleanup_old_conversations(hours=24)
                logger.info(f"Scheduled cleanup completed: removed {removed_count} conversations")
            except Exception as e:
                logger.error(f"Error in scheduled cleanup: {str(e)}")
    
    # Start the background task
    asyncio.create_task(run_db_cleanup())
    logger.info("Scheduled database cleanup task started")

def check_rate_limit():
    """Check if the request is within rate limits."""
    return global_limiter.check_rate_limit()

if __name__ == "__main__":
    # Start the server
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 