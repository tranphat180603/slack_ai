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

from ops_conversation_db import init_db, check_db_connection, cleanup_old_conversations
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

# Set all module loggers to DEBUG level for comprehensive logging
logging.getLogger("openai_client").setLevel(logging.DEBUG)
logging.getLogger("tmai_agent").setLevel(logging.DEBUG)
logging.getLogger("agent").setLevel(logging.DEBUG)
logging.getLogger("context_manager").setLevel(logging.INFO)
logging.getLogger("slack_tools").setLevel(logging.DEBUG)
logging.getLogger("linear_client").setLevel(logging.DEBUG)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Get required environment variables
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_USER_TOKEN = os.environ.get("SLACK_USER_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Get model configuration (allowing different models for different components)
DEFAULT_AI_MODEL = os.environ.get("AI_MODEL", "o3-mini")
COMMANDER_MODEL = os.environ.get("COMMANDER_MODEL", DEFAULT_AI_MODEL)
CAPTAIN_MODEL = os.environ.get("CAPTAIN_MODEL", DEFAULT_AI_MODEL)
SOLDIER_MODEL = os.environ.get("SOLDIER_MODEL", DEFAULT_AI_MODEL)

# Create model configuration
MODEL_CONFIG = {
    "commander": COMMANDER_MODEL,
    "captain": CAPTAIN_MODEL,
    "soldier": SOLDIER_MODEL
}

# Log model configuration
logger.info(f"Using models: Commander={COMMANDER_MODEL}, Captain={CAPTAIN_MODEL}, Soldier={SOLDIER_MODEL}")

if not SLACK_BOT_TOKEN:
    logger.error("SLACK_BOT_TOKEN not found in environment variables")
    raise ValueError("SLACK_BOT_TOKEN is required")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY is required")

# Load prompts from YAML
try:
    PROMPTS = {}
    prompt_files = [
        'command_prompts.yaml', 
        'linear_prompts.yaml', 
        'slack_prompt.yaml', 
        'github_prompt.yaml', 
        'website_prompt.yaml',
        'gdrive_prompts.yaml'
    ]
    
    logger.info(f"Loading prompts from {len(prompt_files)} files")
    
    for file_name in prompt_files:
        file_path = os.path.join('prompts', file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = yaml.safe_load(file)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Loaded {file_path} with categories: {list(file_content.keys())}")
                
                # Merge the content into the main PROMPTS dictionary
                for key, value in file_content.items():
                    if key in PROMPTS:
                        # If key already exists, merge the sub-dictionaries
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Merging category '{key}' from {file_path}")
                        PROMPTS[key].update(value)
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Adding new category '{key}' from {file_path}")
                        PROMPTS[key] = value
            logger.info(f"Successfully loaded prompts from {file_path}")
        except Exception as e:
            logger.warning(f"Error loading prompt file {file_path}: {str(e)}")
            
    if PROMPTS:
        logger.info(f"Loaded {len(PROMPTS)} prompt categories")
        if logger.isEnabledFor(logging.DEBUG):
            for category, content in PROMPTS.items():
                if isinstance(content, dict):
                    logger.debug(f"Category '{category}' contains: {list(content.keys())}")
    else:
        logger.warning("No prompts were successfully loaded")
except Exception as e:
    logger.error(f"Error in prompt loading process: {str(e)}")
    PROMPTS = {}

# Initialize the TMaiSlackAgent
agent = TMAISlackAgent(
    slack_bot_token=SLACK_BOT_TOKEN,
    slack_user_token=SLACK_USER_TOKEN,
    openai_api_key=OPENAI_API_KEY,
    ai_model=DEFAULT_AI_MODEL,
    model_config=MODEL_CONFIG,
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
            
            # For file_shared events, just log it and return - we'll handle the file when processing the message
            if event_type == "file_shared":
                logger.info(f"Received file_shared event, will process with related message")
                return {}
            
            # Skip message subtypes (like bot messages, updates, etc.)
            # EXCEPT for file_share subtype which we need to process for images
            subtype = event.get("subtype")
            if subtype and subtype != "file_share":
                logger.info(f"Skipping message subtype: {subtype}")
                return {}
            
            # Special handling for file_share subtype
            if subtype == "file_share":
                logger.info(f"Processing file_share message subtype")
            
            # Handle direct messages to the bot
            if event_type == "message" and channel_type == "im":
                # Skip if no text content or if it's a bot message or processing message
                # BUT don't skip if it's a file_share subtype
                text = event.get("text", "")
                if (not text.strip() and subtype != "file_share") or event.get("bot_id") or "TMAI's neuron firing..." in text:
                    logger.info(f"Skipping message: empty text={not text.strip()}, bot_id={event.get('bot_id') is not None}")
                    return {}
                
                # Extract required data
                user_id = event.get("user", "")
                channel_id = event.get("channel", "")
                message_ts = event.get("ts", "")
                thread_ts = event.get("thread_ts", message_ts)
                
                # Get files if any
                files = event.get("files", [])
                
                # Get sender name
                sender_name = "User"  # Default fallback
                try:
                    user_info = agent.slack_client.users_info(user=user_id)
                    if user_info.get("ok") and user_info.get("user"):
                        sender_name = "@" + user_info["user"].get("profile").get("display_name") or user_info["user"].get("name") or "User"
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
                    files=files,
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
                                    "text": f"\n```\n➤ Analyzing your query...\n```"
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
                    thread_ts=thread_ts,
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
                
                # Get files if any
                files = event.get("files", [])
                
                # Remove bot mention from text (matches <@BOTID>)
                text = text.replace(f"<@{os.environ.get('SLACK_BOT_ID', 'U08GNQ8F2RH')}>", "").strip()
                
                # Get sender name
                sender_name = "User"  # Default fallback
                try:
                    user_info = agent.slack_client.users_info(user=user_id)
                    if user_info.get("ok") and user_info.get("user"):
                        sender_name = "@" + user_info["user"].get("profile").get("display_name") or user_info["user"].get("name") or "User"
                except SlackApiError as e:
                    logger.warning(f"Could not get user name, using default: {e.response['error']}")
                
                # Create AI request
                ai_request = AIRequest(
                    text=text,
                    user_id=user_id,
                    channel_id=channel_id,
                    message_ts=message_ts,
                    thread_ts=thread_ts,
                    files=files,
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
                                    "text": f"\n```\n➤ Analyzing your query...\n```"
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

@app.get("/debug/contexts")
async def debug_contexts():
    """Debug endpoint to view active contexts (ONLY for development/debugging)"""
    # Only enable in development mode
    if os.environ.get("ENVIRONMENT", "").lower() != "development":
        return {"error": "Debug endpoints only available in development mode"}
    
    from context_manager import context_manager
    
    # Get a sanitized view of all contexts
    contexts_info = {}
    for ctx_id, ctx in context_manager.contexts.items():
        contexts_info[ctx_id] = {
            "created_at": ctx.get("created_at", 0),
            "channel_id": ctx.get("channel_id", "unknown"),
            "user_id": ctx.get("user_id", "unknown"),
            "stop_requested": ctx.get("stop_requested", False),
            "user_query": ctx.get("user_query", "")[:50] + "..." if len(ctx.get("user_query", "")) > 50 else ctx.get("user_query", ""),
        }
    
    return {
        "active_contexts": len(context_manager.contexts),
        "contexts": contexts_info
    }

@app.post("/slack/interactivity")
async def slack_interactivity(request: Request, background_tasks: BackgroundTasks):
    """Handle Slack interactive components (buttons, menus, etc.)"""
    logger.info("Slack interactivity endpoint called")
    
    try:
        # Slack sends form-encoded payload
        form_data = await request.form()
        payload_str = form_data.get("payload")
        
        if not payload_str:
            logger.warning("No payload found in interactive request")
            return {}
        
        # Parse the payload JSON
        payload = json.loads(payload_str)
        logger.debug(f"Received interaction payload type: {payload.get('type')}")
        
        # Handle view submissions (from modals)
        if payload.get("type") == "view_submission":
            # Initialize SlackModals
            from ops_slack.slack_modals import SlackModals
            slack_modals = SlackModals(agent.slack_client)
            
            # Extract metadata and callback_id before processing
            view = payload.get("view", {})
            callback_id = view.get("callback_id", "")
            metadata_str = view.get("private_metadata", "{}")
            metadata = {}
            channel_id = None
            
            try:
                metadata = json.loads(metadata_str)
                conversation_id = metadata.get("conversation_id")
                if conversation_id and ":" in conversation_id:
                    channel_id = conversation_id.split(":")[0]
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Error extracting metadata: {str(e)}")
            
            # For Linear issue creation/update, immediately acknowledge to prevent timeout
            if callback_id in ["linear_create_issue_modal", "linear_update_issue_modal"]:
                logger.info(f"Immediately acknowledging {callback_id} to prevent timeout")
                
                # Extract form state for background processing
                form_state = view.get("state", {}).get("values", {})
                user_id = payload.get("user", {}).get("id")
                
                # Define background task to process the Linear operation
                async def process_linear_operation():
                    try:
                        # Process the submission in the background
                        result = await slack_modals.handle_view_submission(payload)
                        logger.info(f"Background view submission result: {result.get('success')}, message: {result.get('message')}")
                        
                        # Send confirmation message
                        if result.get("success") and channel_id:
                            try:
                                await asyncio.sleep(0.5)  # Small delay
                                agent.slack_client.chat_postMessage(
                                    channel=channel_id,
                                    text=f":white_check_mark: {result.get('message')}"
                                )
                            except Exception as e:
                                logger.error(f"Error sending confirmation: {str(e)}")
                        elif not result.get("success") and channel_id:
                            # Send error message
                            try:
                                await asyncio.sleep(0.5)  # Small delay
                                agent.slack_client.chat_postMessage(
                                    channel=channel_id,
                                    text=f":x: Error: {result.get('message')}"
                                )
                            except Exception as e:
                                logger.error(f"Error sending error message: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error in background Linear operation: {str(e)}")
                        if channel_id:
                            try:
                                agent.slack_client.chat_postMessage(
                                    channel=channel_id,
                                    text=f":x: An error occurred while processing your request: {str(e)}"
                                )
                            except Exception:
                                pass
                
                # Schedule the background task
                background_tasks.add_task(process_linear_operation)
                
                # Return immediate success to close the modal
                return {"response_action": "clear"}
            
            # For all other modal types, process normally
            # Process the submission
            result = await slack_modals.handle_view_submission(payload)
            
            logger.info(f"View submission result: {result.get('success')}, message: {result.get('message')}")
            # Add more detailed debug logging
            logger.debug(f"Full result from handle_view_submission: {result}")
            
            # If successful, acknowledge with an empty response to close the modal
            if result.get("success"):
                # Send a message to the user about the completed action
                user_id = payload.get("user", {}).get("id")
                
                # If we found a channel, send a message
                if channel_id:
                    try:
                        # Define a background function to send the message
                        async def send_success_message():
                            try:
                                await asyncio.sleep(0.5)  # Small delay to ensure modal closes first
                                agent.slack_client.chat_postMessage(
                                    channel=channel_id,
                                    text=f":white_check_mark: {result.get('message')}"
                                )
                            except Exception as e:
                                logger.error(f"Error in background message task: {str(e)}")
                        
                        # Add the task to background tasks
                        background_tasks.add_task(send_success_message)
                    except Exception as e:
                        logger.error(f"Error scheduling success message: {str(e)}")
                    
                # Return empty response to close the modal - must be {} not None
                return {
                    "response_action": "clear"
                }
            else:
                # Return errors to display in the modal
                # For simplicity, show a general error message for now
                error_message = result.get("message", "An error occurred")
                logger.warning(f"Returning error to Slack modal: {error_message}")
                
                # Format error response according to Slack's requirements
                # https://api.slack.com/surfaces/modals/using#handling_submissions
                errors = {}
                
                # Try to show error in the most appropriate block
                if "title" in error_message.lower():
                    errors["title_block"] = error_message
                elif "description" in error_message.lower():
                    errors["description_block"] = error_message
                else:
                    # Default to showing error in title block
                    errors["title_block"] = error_message
                
                return {
                    "response_action": "errors",
                    "errors": errors
                }
        
        # CRITICAL: Return a 200 OK immediately if this is a stop action
        # This ensures we stay within Slack's 3-second timeout window
        if payload.get("type") == "block_actions":
            actions = payload.get("actions", [])
            if actions and actions[0].get("action_id") == "stop_processing":
                # Process stop request in background
                background_tasks.add_task(
                    agent.handle_interaction_payload,
                    payload
                )
                logger.info("Stop button clicked, returning immediate acknowledgment")
                # Immediately acknowledge with a simple message
                # The actual UI will be updated by the background task
                return {"response_action": "clear"}
            else:
                # Other button clicks
                background_tasks.add_task(
                    agent.handle_interaction_payload,
                    payload
                )
        
        # Return a 200 OK immediately to acknowledge receipt
        return {}
    
    except Exception as e:
        logger.error(f"Error processing interaction: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

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