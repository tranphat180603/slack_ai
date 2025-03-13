import fastapi
import uvicorn
import requests
from fastapi import Request, Form, BackgroundTasks
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

# Set stdout and stderr to handle unicode properly on Windows
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

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
AI_MODEL = os.environ.get("AI_MODEL", "o1-mini")

# Initialize Slack client
slack_client = WebClient(token=SLACK_BOT_TOKEN)
logger.info("Slack client initialized")

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

# Rate limiting tracker
rate_limit_tracker = {
    "last_reset": time.time(),
    "requests": 0
}

app = fastapi.FastAPI()

class ProcessedContent(BaseModel):
    url: str
    text_content: str
    is_tweet: bool
    tweet_id: Optional[str] = None
    user_id: Optional[str] = None
    thread_tweets: Optional[List[dict]] = None

class AIRequest(BaseModel):
    text: str
    user_id: str
    channel_id: str
    response_url: str
    files: Optional[List[Dict[str, Any]]] = None
    urls: Optional[List[str]] = None
    message_ts: Optional[str] = None

@dataclasses.dataclass
class ContentAnalysisResult:
    """Result of content analysis, with categorization and extracted entities."""
    content_type: str = ""  # Main content type
    requires_history: bool = False  # Whether we need channel history
    channel_query_description: str = ""  # Description for channel-based query
    urls: List[str] = None  # General URLs
    linear_issue: Optional[str] = None  # Linear issue ID if present
    github_repo: Optional[str] = None  # GitHub repo if present
    twitter_urls: List[str] = None  # Twitter URLs if present
    linear_working_hours_query: bool = False  # Flag for working hours query
    linear_query: bool = False  # Flag for general Linear query
    text: str = ""  # Original query text

@app.post("/slack/ai_command")
async def ai_command(request: Request, background_tasks: BackgroundTasks):
    """
    Entry point for Slack slash commands.
    Uses the Slack Web API to better manage messages.
    """
    try:
        # Parse the form data from Slack
        form_data = await request.form()
        logger.info(f"Received form data: {form_data.keys()}")
        
        # Extract the necessary fields from form data
        text = form_data.get("text", "")
        user_id = form_data.get("user_id", "")
        channel_id = form_data.get("channel_id", "")
        response_url = form_data.get("response_url", "")
        
        logger.info(f"Processing AI request: {text[:50]}...")
        
        # Check for files
        files = []
        if "files" in form_data:
            try:
                files_json = json.loads(form_data["files"])
                files = files_json if isinstance(files_json, list) else []
            except:
                logger.error("Failed to parse files JSON")
        
        # Create AI request object
        ai_request = AIRequest(
            text=text,
            user_id=user_id,
            channel_id=channel_id,
            response_url=response_url,
            files=files,
            urls=extract_urls(text)
        )
        
        # Send initial message and get the timestamp
        try:
            # Post initial "processing" message and capture its timestamp
            initial_response = slack_client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text="AI is processing...",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Input:*\n```\n{text[:50]}{('...' if len(text) > 50 else '')}\n```\n*Status:*\nProcessing..."
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
            logger.info(f"Posted initial message with ts: {initial_response.get('message_ts')}")
            
            # Include the message timestamp in the AI request for later reference
            ai_request.message_ts = initial_response.get("message_ts")
            
        except SlackApiError as e:
            logger.error(f"Error posting initial message: {e.response['error']}")
        
        # Start processing in the background
        background_tasks.add_task(process_ai_request, ai_request)
        
        # Return an empty 200 response since we already sent the message
        return {}
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        if "channel_id" in locals() and "user_id" in locals():
            try:
                slack_client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text=f"Sorry, I encountered an error: {str(e)}"
                )
            except:
                pass
        return {"status": "error", "message": str(e)}

async def process_ai_request(ai_request: AIRequest):
    """Process an AI request with direct message handling using the Web API."""
    try:
        logger.info(f"Processing AI request from user {ai_request.user_id}: {ai_request.text[:50]}...")
        
        # Check rate limits
        if not check_rate_limit():
            await send_slack_response(
                ai_request.response_url,
                "Rate limit exceeded. Please try again in a minute.",
                error=True,
                replace_original=True  # Replace the loading message
            )
            return
        
        # Analyze the content to determine what we're looking for
        content_analysis = await analyze_content(ai_request.text)
        logger.info(f"Content analysis: {content_analysis}")
        
        # Determine search parameters based on the content analysis
        search_params = await evaluate_search_parameters(ai_request.text, content_analysis)
        logger.info(f"Search parameters: {search_params}")
        
        # Get relevant messages from channel history if needed
        search_results = []
        if content_analysis.requires_history:
            search_results = await search_channel_history(ai_request.channel_id, search_params)
            logger.info(f"Retrieved {len(search_results)} messages from history")
        
        # Determine which content sources we need to fetch
        required_sources = await determine_required_content_sources(
            ai_request.text, 
            search_results, 
            content_analysis
        )
        logger.info(f"Required content sources: {required_sources}")
        
        # Evaluate the search results to identify the most relevant content
        evaluation_result = None
        if search_results:
            evaluation_result = await evaluate_search_results(
                ai_request.text,
                search_results,
                content_analysis
            )
            logger.info(f"Evaluation result contains {len(evaluation_result.get('relevant_messages', []))} relevant messages")
        
        # Check for recent images if the query is image-related
        recent_images = None
        if detect_image_related_query(ai_request.text):
            recent_messages = await get_recent_messages(ai_request.channel_id)
            recent_images = extract_images_from_messages(recent_messages)
            logger.info(f"Retrieved {len(recent_images)} recent images")
        
        # Selectively fetch only the content we need
        content = await fetch_content_selectively(
            ai_request.text,
            required_sources,
            content_analysis,
            ai_request.channel_id
        )
        logger.info("Content fetched selectively")
        
        # Build the final context for AI processing
        context, image_urls = await build_response_context(
            ai_request.text,
            content,
            search_results,
            evaluation_result,
            recent_images
        )
        logger.info(f"Built context with length: {len(context)}")
        
        # Call AI with the built context
        response_text = await call_ai(context, image_urls)
        logger.info(f"Received AI response with length: {len(response_text)}")
        
        # Send a new message with the AI response and delete the old one
        try:
            # Post new message with AI response
            slack_client.chat_postMessage(
                channel=ai_request.channel_id,
                text=response_text,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": response_text,
                            "verbatim": True
                        }
                    }
                ]
            )
            logger.info("Posted AI response as new message")
            
            # Delete the ephemeral processing message if possible
            if hasattr(ai_request, 'message_ts') and ai_request.message_ts:
                try:
                    slack_client.chat_delete(
                        channel=ai_request.channel_id,
                        ts=ai_request.message_ts
                    )
                    logger.info(f"Deleted processing message with ts: {ai_request.message_ts}")
                except SlackApiError as e:
                    # This may fail for ephemeral messages, which is okay
                    logger.warning(f"Could not delete processing message: {e.response['error']}")
            
        except SlackApiError as e:
            logger.error(f"Error sending AI response: {e.response['error']}")
            # Fall back to response_url if Web API fails
            await send_slack_response(
                ai_request.response_url,
                response_text
            )
        
    except Exception as e:
        logger.error(f"Error in process_ai_request: {str(e)}")
        
        # Send error as a new message
        try:
            slack_client.chat_postEphemeral(
                channel=ai_request.channel_id,
                user=ai_request.user_id,
                text=f"Sorry, I encountered an error: {str(e)}"
            )
        except Exception as post_error:
            logger.error(f"Error sending error message: {str(post_error)}")
            # Fall back to response_url
            await send_slack_response(
                ai_request.response_url,
                f"Sorry, I encountered an error: {str(e)}",
                error=True
            )

async def send_slack_response(response_url: str, text: str, error: bool = False, replace_original: bool = True, channel_id: str = None, is_ephemeral: bool = False):
    """Send a response back to Slack with improved error handling for used URLs."""
    logger.info(f"Sending response to Slack: {text[:50]}...")
    
    try:
        headers = {"Content-Type": "application/json"}
        
        # Prepare blocks for better formatting
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": text,
                    "verbatim": True  # Add this to prevent Slack from auto-formatting
                }
            }
        ]
        
        # Add error styling if needed
        if error:
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "⚠️ _This response contains an error_"
                    }
                ]
            })
        
        # Send the response
        payload = {
            "text": text,  # Fallback text
            "blocks": blocks,
            "response_type": "in_channel" if not is_ephemeral else "ephemeral",
            "replace_original": replace_original,
            "delete_original": False  # Don't delete the original - just replace it
        }
        
        # Use aiohttp for async request
        async with aiohttp.ClientSession() as session:
            async with session.post(response_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error sending Slack response: {response.status} - {error_text}")
                    
                    # If we get a used_url error, try to send a message directly to the channel instead
                    if "used_url" in error_text and channel_id:
                        try:
                            # Fall back to posting directly to the channel
                            slack_client.chat_postMessage(
                                channel=channel_id,
                                text=text,
                                blocks=blocks
                            )
                            logger.info("Successfully sent message directly to channel after used_url error")
                        except Exception as e:
                            logger.error(f"Failed to send fallback message: {str(e)}")
                else:
                    logger.info("Successfully sent response to Slack")
    except Exception as e:
        logger.error(f"Exception sending Slack response: {str(e)}")

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

async def analyze_content(text: str) -> ContentAnalysisResult:
    """
    Analyze the content of a message to determine its type and extract relevant information.
    
    Args:
        text: The text of the message
        
    Returns:
        ContentAnalysisResult object with analysis results
    """
    logger.info("Analyzing message content")
    
    # Initialize with default values
    result = ContentAnalysisResult()
    result.text = text  # Store the original text
    
    # Extract URLs from the text
    urls = extract_urls(text)
    result.urls = urls if urls else None
    
    # Set placeholder for twitter URLs
    result.twitter_urls = []
    
    # Extract specific content patterns first
    
    # Extract Linear issue ID if present
    linear_issue_match = re.search(r'\b([A-Z]+-\d+)\b', text)
    if linear_issue_match:
        result.linear_issue = linear_issue_match.group(1)
        logger.info(f"Extracted Linear issue ID: {result.linear_issue}")
        result.content_type = "linear_issue"
        return result  # Return early if we found a Linear issue
    
    # Extract GitHub repo if present in text or URLs
    github_repo = None
    github_pattern = r'github\.com/([^/\s]+/[^/\s]+)'
    
    # First check in the text
    github_match = re.search(github_pattern, text)
    if github_match:
        github_repo = github_match.group(1)
        result.github_repo = github_repo
        result.content_type = "github"
        logger.info(f"Extracted GitHub repo from text: {github_repo}")
        return result  # Return early if we found a GitHub repo
    
    # If not found in text, check in URLs
    if not github_repo and urls:
        for url in urls:
            github_match = re.search(github_pattern, url)
            if github_match:
                github_repo = github_match.group(1)
                # Remove any trailing parts like /blob, /tree, etc.
                github_repo = re.sub(r'/(blob|tree|pull|issues).*$', '', github_repo)
                result.github_repo = github_repo
                result.content_type = "github"
                logger.info(f"Extracted GitHub repo from URL: {github_repo}")
                return result  # Return early if we found a GitHub repo
    
    # Check for Twitter URLs
    twitter_urls = []
    twitter_pattern = r'(https?://(twitter\.com|x\.com)/[^\s]+)'
    
    # First check in the text
    for match in re.finditer(twitter_pattern, text):
        twitter_url = match.group(1)
        twitter_urls.append(twitter_url)
        result.content_type = "twitter"
        logger.info(f"Extracted Twitter URL from text: {twitter_url}")
    
    # Then check in the extracted URLs
    if not twitter_urls and urls:
        for url in urls:
            if re.match(twitter_pattern, url):
                twitter_urls.append(url)
                result.content_type = "twitter"
                logger.info(f"Extracted Twitter URL: {url}")
    
    if twitter_urls:
        result.twitter_urls = twitter_urls
        return result  # Return early if we found Twitter URLs
    
    # If we have URLs but no specific pattern matched, set as url_content
    if not result.content_type and urls:
        result.content_type = "url_content"
        logger.info("Setting content type to URL content")
        return result  # Return early if we have URLs
    
    # For all other content, use AI classification with our improved prompts
    try:
        # Get the prompt templates from YAML
        system_prompt = PROMPTS.get("analyze_content", {}).get("system", "")
        user_prompt_template = PROMPTS.get("analyze_content", {}).get("user_template", "")
        
        # Format the user prompt
        user_prompt = user_prompt_template.format(
            text=text,
            urls=urls if urls else []
        )
        
        # Call OpenAI
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        analysis = json.loads(response.choices[0].message.content)
        logger.info(f"AI content analysis: {analysis}")
        
        # Update the result with the AI analysis
        result.content_type = analysis.get("content_type", "simple_query")  # Default to simple_query if not specified
        result.requires_history = analysis.get("requires_history", False)
        result.channel_query_description = analysis.get("channel_query_description", "")
        
        # Update additional fields if provided
        if "linear_issue" in analysis and analysis["linear_issue"]:
            result.linear_issue = analysis["linear_issue"]
        
        if "github_repo" in analysis and analysis["github_repo"]:
            result.github_repo = analysis["github_repo"]
        
        if "linear_working_hours_query" in analysis:
            result.linear_working_hours_query = analysis["linear_working_hours_query"]
        
        if "linear_query" in analysis:
            result.linear_query = analysis["linear_query"]
        
        # Update content type based on specific flags
        if result.linear_query and result.content_type != "linear_issue":
            result.content_type = "linear_data"
        
        if result.linear_working_hours_query:
            result.content_type = "linear_working_hours"

        
        #manually checking if the query is linear related
        if_linear = False
        if_linear = check_if_linear_query(text)
        if if_linear:
            result.linear_query = True
        
        logger.info(f"AI determined content type: {result.content_type}, requires_history: {result.requires_history}")
        return result
        
    except Exception as e:
        logger.error(f"Error in AI content analysis: {str(e)}")
        # Fallback to simple_query for any error
        result.content_type = "simple_query"
        result.requires_history = False
        logger.info("Falling back to simple_query type due to error")
        return result

async def check_if_linear_query(text: str) -> bool:
    """
    Check if the query is likely related to Linear project management data.
    
    Parameters:
    - text: The text to analyze
    
    Returns:
    - True if the query is likely about Linear, False otherwise
    """
    logger.info(f"Checking if query is Linear-related: {text}")
    
    # Check for obvious Linear keywords
    linear_keywords = [
        "linear", "project", "issue", "task", "cycle", "sprint", "roadmap",
        "team", "milestone", "assignee", "priority", "estimate", "points",
        "backlog", "ticket", "track", "progress", "status", "todo", "in progress",
        "done", "complete", "blocked", "timeline", "due date", "deadline",
        "working hours", "time tracking", "logged time", "work item"
    ]
    
    # Check for any of the keywords in the text
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in linear_keywords):
        logger.info("Text contains Linear keywords")
        return True
    
    # For less obvious cases, ask AI if this is a Linear-related query
    system_prompt = """
    You are an expert at determining if a query is related to project management in Linear.
    You'll receive a user query and should determine if it's asking about:
    - Team information in Linear
    - Sprint/cycle planning or status
    - Issues, tasks, or work items
    - User assignments or workload
    - Project timelines or roadmaps
    - Project or team progress
    - Working hours or time tracking
    - Any other Linear project management data
    
    Answer with ONLY "yes" or "no" - is this query asking about Linear project management data?
    """
    
    user_prompt = f"Is this query related to Linear project management data? Query: '{text}'"
    
    try:
        # Call OpenAI to determine if this is a Linear query
        response = await call_openai(system_prompt, user_prompt)
        
        # Check if the response indicates this is a Linear query
        if response and response.lower().strip() in ["yes", "true", "y"]:
            logger.info("AI determined this is a Linear query")
            return True
        else:
            logger.info("AI determined this is NOT a Linear query")
            return False
            
    except Exception as e:
        logger.error(f"Error determining if query is Linear-related: {str(e)}")
        
        # Fall back to keyword matching if AI call fails
        return any(keyword in text_lower for keyword in linear_keywords)

async def search_channel_history(channel_id: str, search_params: Dict[str, Any]):
    """
    Search channel history with enhanced parameters for relevance.
    Now includes better handling of recently posted images and files.
    """
    logger.info(f"Searching channel history with custom parameters: {search_params}")
    
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
        
        # Fetch recent history first - this ensures we get the most recent messages including images
        result = slack_client.conversations_history(
            channel=channel_id,
            limit=20,  # Start with most recent 20 messages to catch newly posted images
            inclusive=True
        )
        
        messages = result["messages"]
        
        # If we need more messages and there are more to fetch
        if max_messages > 20 and result.get("has_more", False):
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
        
        # Enhanced processing of messages - especially focused on images
        processed_messages = []
        for msg in messages:
            # Check for images in files
            has_files = False
            has_images = False
            
            if "files" in msg:
                has_files = True
                for file in msg["files"]:
                    if file.get("mimetype", "").startswith("image/"):
                        has_images = True
                        # Add image-specific metadata to make it easier to find
                        file["is_image"] = True
                        
            # Add these flags to make filtering easier
            msg["has_files"] = has_files
            msg["has_images"] = has_images
            
            # Also mark messages that have URLs or code blocks
            msg["has_urls"] = "http://" in msg.get("text", "") or "https://" in msg.get("text", "")
            msg["has_code"] = "```" in msg.get("text", "")
            
            processed_messages.append(msg)
        
        logger.info(f"Found {len(processed_messages)} relevant messages in channel history")
        
        # If we're specifically looking for images, make sure those are prioritized
        if search_params.get("resource_types") and "images" in search_params.get("resource_types"):
            # Move messages with images to the front of the list
            image_messages = [msg for msg in processed_messages if msg.get("has_images")]
            non_image_messages = [msg for msg in processed_messages if not msg.get("has_images")]
            processed_messages = image_messages + non_image_messages
            logger.info(f"Prioritized {len(image_messages)} messages with images")
            
        return processed_messages
        
    except SlackApiError as e:
        logger.error(f"Error searching channel history: {e.response['error']}")
        return []

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

async def fetch_content_selectively(query: str, required_sources: Dict[str, bool], content_analysis: ContentAnalysisResult, channel_id: str = None, timestamp: str = None, thread_ts: str = None, slack_client=None):
    """Fetch content from required sources in parallel."""
    logger.info(f"Fetching content from required sources: {required_sources}")
    
    content = {}
    
    # Prepare tasks based on required sources
    tasks = []
    
    # Linear issue - FIXED: Check for both 'linear' and 'linear_issue' in required_sources
    if (required_sources.get("linear_issue", False) or required_sources.get("linear", False)) and content_analysis.linear_issue:
        logger.info(f"Adding task to fetch Linear issue: {content_analysis.linear_issue}")
        tasks.append(("linear_issue", fetch_linear_issue(content_analysis.linear_issue)))
    
    # Linear working hours
    if required_sources.get("linear_working_hours", False):
        logger.info("Fetching Linear working hours data")
        # If we have a specific team mentioned in the query, fetch for that team
        team_id = None
        
        # TODO: Extract team name from query if possible
        
        tasks.append(("linear_working_hours", fetch_team_working_hours(team_id)))
    
    # General Linear query using the new query planner
    if required_sources.get("linear_query", False):
        logger.info("Fetching general Linear data using query planner")
        tasks.append(("linear_data", process_linear_query(query)))
    
    # GitHub repo
    if required_sources.get("github_repo", False) and content_analysis.github_repo:
        tasks.append(("github_repo", fetch_github_repo(content_analysis.github_repo)))
    
    # URL content
    if required_sources.get("url_content", False) and content_analysis.urls:
        for url in content_analysis.urls:
            tasks.append((f"url_{url}", fetch_url_content(url)))
    
    # Twitter content
    if required_sources.get("twitter", False) and content_analysis.twitter_urls:
        for url in content_analysis.twitter_urls:
            tasks.append((f"twitter_{url}", fetch_twitter_content(url)))
    
    # Channel info (for channel-specific queries)
    if required_sources.get("channel_info", False) and channel_id:
        tasks.append(("channel_info", get_channel_info(channel_id)))
    
    # Execute all tasks in parallel
    results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
    
    # Process results
    for i, (key, _) in enumerate(tasks):
        result = results[i]
        if isinstance(result, Exception):
            logger.error(f"Error fetching {key}: {str(result)}")
            content[key] = {"error": str(result)}
        else:
            logger.info(f"Successfully fetched {key}")
            
            # Special handling for Twitter URLs to ensure they're added with the right key
            if key.startswith("twitter_"):
                url = key[len("twitter_"):]
                content[f"twitter_{url}"] = result
            # Special handling for regular URLs to ensure they're added with the right key
            elif key.startswith("url_"):
                url = key[len("url_"):]
                content[f"url_{url}"] = result
            else:
                content[key] = result
    
    logger.info(f"Content fetched from {len(content)} sources")
    return content

async def fetch_url_content(url: str):
    """Fetch content from a general URL."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove scripts, styles, etc.
                    for element in soup(["script", "style", "footer", "nav"]):
                        element.extract()
                    
                    # Get text
                    text = soup.get_text(separator=' ', strip=True)
                    
                    # Simple text cleaning
                    text = re.sub(r'\s+', ' ', text)
                    
                    return text[:5000]  # Limit to 5000 chars
                else:
                    return f"Error fetching URL: HTTP {response.status}"
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

async def fetch_github_repo(repo_url: str):
    """Fetch GitHub repository information."""
    if not github_client:
        logger.error("GitHub client not initialized")
        return {"error": "GitHub API access not configured"}
    
    try:
        # Extract repo owner and name from URL
        parsed_url = urlparse(repo_url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        if len(path_parts) < 2:
            return {"error": "Invalid GitHub repository URL format"}
        
        owner, repo_name = path_parts[0], path_parts[1]
        
        logger.info(f"Fetching GitHub repository: {owner}/{repo_name}")
        
        # Try to fetch the repository
        try:
            repo = github_client.get_repo(f"{owner}/{repo_name}")
        except Exception as repo_error:
            logger.error(f"Error accessing GitHub repository {owner}/{repo_name}: {str(repo_error)}")
            return {
                "name": f"{owner}/{repo_name}",
                "description": "⚠️ This repository couldn't be accessed. It may be private or not exist.",
                "url": repo_url,
                "error": str(repo_error)
            }
        
        # Get basic repo information
        repo_info = {
            "name": repo.full_name,
            "description": repo.description or "No description available",
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "url": repo.html_url,
            "language": repo.language
        }
        
        # Try to get README content
        try:
            readme = repo.get_readme()
            readme_content = base64.b64decode(readme.content).decode('utf-8')
            repo_info["readme"] = readme_content
        except Exception as readme_error:
            logger.warning(f"Could not fetch README for {owner}/{repo_name}: {str(readme_error)}")
            repo_info["readme"] = "README not available"
        
        return repo_info
    
    except Exception as e:
        logger.error(f"Error fetching GitHub repository: {str(e)}")
        return {
            "name": repo_url,
            "description": f"Error fetching repository information: {str(e)}",
            "url": repo_url,
            "error": str(e)
        }

async def fetch_linear_issue(issue_id: str):
    """Fetch details of a Linear issue by its ID."""
    logger.info(f"Fetching Linear issue: {issue_id}")
    
    if not linear_client:
        logger.error("Linear client not initialized")
        return None
        
    try:
        issue = linear_client.get_issue(issue_id)
        if not issue:
            logger.warning(f"Linear issue not found: {issue_id}")
            return None
            
        issue_data = {
            "id": issue.id,
            "title": issue.title,
            "description": issue.description,
            "status": issue.state.get("name") if issue.state else "Unknown",
            "assignee": issue.assignee.get("name") if issue.assignee else "Unassigned",
            "labels": [label.get("name") for label in issue.labels] if hasattr(issue, "labels") and issue.labels else [],
            "priority": getattr(issue, "priority", None),
            "created_at": issue.created_at
        }
        
        logger.info(f"Retrieved Linear issue: {issue.title}")
        return issue_data
        
    except Exception as e:
        logger.error(f"Error fetching Linear issue {issue_id}: {str(e)}")
        return None

async def fetch_team_working_hours(team_id: Optional[str] = None, required_hours: int = 40):
    """
    Fetch working hours information for a specific team or all teams.
    Can also identify users not meeting their required hours.
    
    Parameters:
    - team_id: Optional team ID. If None, will fetch for all teams.
    - required_hours: Number of required working hours (default: 40)
    """
    logger.info(f"Fetching team working hours. Team ID: {team_id if team_id else 'All teams'}")
    
    if not linear_client:
        logger.error("Linear client not initialized")
        return {
            "error": "Linear client not initialized. Please check your API configuration."
        }
    
    try:
        # If we have a specific team ID, get data for just that team
        if team_id:
            team_data = linear_client.get_team_working_hours(team_id)
            
            # Find users not meeting hours
            users_below_target = []
            for member in team_data.get("members", []):
                if member.get("total_hours", 0) < required_hours:
                    users_below_target.append({
                        "name": member.get("name", "Unknown"),
                        "email": member.get("email", ""),
                        "hours_logged": member.get("total_hours", 0),
                        "missing_hours": member.get("missing_hours", 0),
                        "team_id": team_id
                    })
            
            return {
                "team_data": team_data,
                "users_below_target": users_below_target,
                "week_start_date": team_data.get("week_start_date", ""),
                "week_end_date": team_data.get("week_end_date", "")
            }
        else:
            # Get all users not meeting required hours across all teams
            users_below_target = linear_client.find_users_not_meeting_required_hours(required_hours)
            
            # Get summary of all teams
            teams_summary = linear_client.get_all_teams_working_hours()
            
            return {
                "users_below_target": users_below_target,
                "teams_summary": teams_summary,
                "required_hours": required_hours
            }
            
    except Exception as e:
        error_message = f"Error fetching working hours information: {str(e)}"
        logger.error(error_message)
        return {"error": error_message}

async def fetch_twitter_content(url: str):
    """Fetch content from a Twitter/X URL."""
    # This will reuse your existing Twitter/X functionality
    processed_content = process_tweet_url(url)
    
    # Extract just what we need
    twitter_content = {
        "tweet_id": processed_content.tweet_id,
        "user_id": processed_content.user_id,
        "text": processed_content.text_content,
        "is_thread": processed_content.thread_tweets is not None,
        "thread": processed_content.thread_tweets
    }
    
    return twitter_content

async def build_response_context(query: str, content: Dict[str, Any], channel_history: List[Dict[str, Any]], evaluation_result: Dict[str, Any] = None, recent_images: List[Dict[str, Any]] = None):
    """Build the context for sending to the AI, based on the content sources that were fetched."""
    logger.info("Building final response context")
    
    parts = []
    image_urls = []
    
    # Format channel history if present
    formatted_messages = []
    if channel_history:
        if evaluation_result and "relevant_message_indices" in evaluation_result:
            relevant_indices = evaluation_result.get("relevant_message_indices", [])
            relevant_messages = [channel_history[i] for i in relevant_indices if i < len(channel_history)]
            
            # Format the relevant messages
            for message in relevant_messages:
                user_info = message.get("user_profile", {})
                username = user_info.get("real_name", "Unknown User")
                text = message.get("text", "")
                
                # Add attachment info if available
                attachments = message.get("attachments", [])
                for attachment in attachments:
                    if attachment.get("text"):
                        text += f"\n[Attachment: {attachment.get('text')}]"
                
                formatted = f"{username}: {text}"
                formatted_messages.append(formatted)
            
            if formatted_messages:
                parts.append("Relevant channel history:")
                parts.extend(formatted_messages)
                parts.append("")  # Add blank line
    
    # Add Linear issue data if present
    if "linear_issue" in content:
        issue_data = content["linear_issue"]
        parts.append("Linear Issue Information:")
        
        if isinstance(issue_data, dict):
            if "error" in issue_data:
                parts.append(f"Error retrieving Linear issue: {issue_data['error']}")
            else:
                # Access dictionary keys instead of attributes
                parts.append(f"Title: {issue_data.get('title', 'Unknown')}")
                parts.append(f"ID: {issue_data.get('id', 'Unknown')}")
                
                if 'description' in issue_data:
                    parts.append(f"Description: {issue_data['description']}")
                
                if 'status' in issue_data:
                    parts.append(f"State: {issue_data['status']}")
                elif 'state' in issue_data and isinstance(issue_data['state'], dict):
                    parts.append(f"State: {issue_data['state'].get('name', 'Unknown')}")
                
                if 'assignee' in issue_data:
                    if isinstance(issue_data['assignee'], dict):
                        parts.append(f"Assignee: {issue_data['assignee'].get('name', 'Unassigned')}")
                    else:
                        parts.append(f"Assignee: {issue_data['assignee']}")
                
                if 'labels' in issue_data and issue_data['labels']:
                    if isinstance(issue_data['labels'], list):
                        labels = [label for label in issue_data['labels']]
                        parts.append(f"Labels: {', '.join(labels)}")
                    elif isinstance(issue_data['labels'], dict) and 'nodes' in issue_data['labels']:
                        # Handle the format seen in the logs
                        labels = [label.get('name', '') for label in issue_data['labels']['nodes']]
                        parts.append(f"Labels: {', '.join(labels) or 'None'}")
                
                if 'priority' in issue_data:
                    priority = issue_data['priority']
                    priority_label = {0: "No priority", 1: "Urgent", 2: "High", 3: "Medium", 4: "Low"}.get(priority, f"Priority {priority}")
                    parts.append(f"Priority: {priority_label}")
                
                if 'created_at' in issue_data:
                    parts.append(f"Created: {issue_data['created_at']}")
                elif 'createdAt' in issue_data:
                    parts.append(f"Created: {issue_data['createdAt']}")
        else:
            # Handle the case where issue_data might be something else
            parts.append(f"Issue data: {str(issue_data)}")
        
        parts.append("")  # Add blank line
    
    # Add Linear data from the query planner if present
    if "linear_data" in content:
        linear_data = content["linear_data"]
        parts.append("Linear Data Information:")
        
        if isinstance(linear_data, dict):
            if "success" in linear_data and linear_data["success"]:
                # Add the formatted results from the planner
                parts.append(linear_data.get("formatted_results", "No formatted results available"))
            elif "error" in linear_data:
                parts.append(f"Error retrieving Linear data: {linear_data['error']}")
                
                # Add available partial results if present
                if "partial_results" in linear_data:
                    parts.append("Partial results available:")
                    if "teams" in linear_data["partial_results"]:
                        teams = linear_data["partial_results"]["teams"]
                        parts.append(f"Available teams: {', '.join([team.get('name', 'Unknown') for team in teams])}")
                    if "cycles" in linear_data["partial_results"]:
                        cycles = linear_data["partial_results"]["cycles"]
                        cycle_names = []
                        for cycle in cycles:
                            cycle_number = cycle.get('number', 'Unknown')
                            cycle_names.append(f"#{cycle_number}")
                        parts.append(f"Available cycles: {', '.join(cycle_names)}")
                
                # Add clarification suggestions if present
                if "clarification_needed" in linear_data:
                    for clarification in linear_data["clarification_needed"]:
                        parts.append(f"Clarification needed: {clarification.get('message', 'Unknown issue')}")
        else:
            parts.append("Error: Linear data is not in the expected format")
        
        parts.append("")  # Add blank line
    
    # Add Linear query results if present (from the old approach)
    if "linear_query" in content:
        linear_query_data = content["linear_query"]
        parts.append("Linear Query Results:")
        
        if isinstance(linear_query_data, dict):
            if "formatted_results" in linear_query_data:
                parts.append(linear_query_data["formatted_results"])
            elif "error" in linear_query_data:
                parts.append(f"Error executing Linear query: {linear_query_data['error']}")
            else:
                parts.append("No formatted results available from Linear query")
        else:
            parts.append("Error: Linear query data is not in the expected format")
        
        parts.append("")  # Add blank line
    
    # Add Linear working hours data if present
    if "linear_working_hours" in content:
        working_hours_data = content["linear_working_hours"]
        parts.append("Linear Working Hours Information:")
        
        if isinstance(working_hours_data, dict) and "error" in working_hours_data:
            parts.append(f"Error retrieving working hours: {working_hours_data['error']}")
        elif isinstance(working_hours_data, dict) and "team_id" in working_hours_data:
            # Single team data format
            team_id = working_hours_data.get("team_id", "Unknown")
            members = working_hours_data.get("members", [])
            week_start = working_hours_data.get("week_start_date", "Unknown")
            week_end = working_hours_data.get("week_end_date", "Unknown")
            
            parts.append(f"Period: {week_start} to {week_end}")
            parts.append(f"Team members with logged hours: {len(members)}")
            
            # Find users not meeting required hours
            required_hours = 40  # Default
            users_below_threshold = [m for m in members if m.get("total_hours", 0) < required_hours]
            
            if users_below_threshold:
                parts.append(f"Users not meeting {required_hours} hours requirement:")
                for user in users_below_threshold:
                    parts.append(f"- {user.get('name', 'Unknown')}: {user.get('total_hours', 0)} hours (missing {user.get('missing_hours', 0)} hours)")
            else:
                parts.append(f"All users have met the {required_hours} hours requirement.")
            
            # Add total and average team hours
            total_hours = working_hours_data.get("total_team_hours", 0)
            avg_hours = working_hours_data.get("avg_team_hours", 0)
            
            parts.append(f"Total team hours: {total_hours}")
            parts.append(f"Average hours per member: {avg_hours}")
            
        elif isinstance(working_hours_data, list):
            # All teams data format
            parts.append(f"Teams with working hours data: {len(working_hours_data)}")
            
            for team_data in working_hours_data:
                team_name = team_data.get("team_name", "Unknown")
                team_key = team_data.get("team_key", "")
                week_data = team_data.get("week_data", {})
                
                members = week_data.get("members", [])
                week_start = week_data.get("week_start_date", "Unknown")
                week_end = week_data.get("week_end_date", "Unknown")
                
                parts.append(f"\nTeam: {team_name} ({team_key})")
                parts.append(f"Period: {week_start} to {week_end}")
                parts.append(f"Team members with logged hours: {len(members)}")
                
                # Find users not meeting required hours
                required_hours = 40  # Default
                users_below_threshold = [m for m in members if m.get("total_hours", 0) < required_hours]
                
                if users_below_threshold:
                    parts.append(f"Users not meeting {required_hours} hours requirement:")
                    for user in users_below_threshold[:5]:  # Limit to 5 users to avoid long responses
                        parts.append(f"- {user.get('name', 'Unknown')}: {user.get('total_hours', 0)} hours (missing {user.get('missing_hours', 0)} hours)")
                    
                    if len(users_below_threshold) > 5:
                        parts.append(f"... and {len(users_below_threshold) - 5} more users")
                else:
                    parts.append(f"All users have met the {required_hours} hours requirement.")
                
                # Add total and average team hours
                total_hours = week_data.get("total_team_hours", 0)
                avg_hours = week_data.get("avg_team_hours", 0)
                
                parts.append(f"Total team hours: {total_hours}")
                parts.append(f"Average hours per member: {avg_hours}")
        
        parts.append("")  # Add blank line
    
    # Add GitHub repo data if present
    if "github_repo" in content:
        repo_data = content["github_repo"]
        parts.append("GitHub Repository Information:")
        
        if isinstance(repo_data, dict) and "error" in repo_data:
            parts.append(f"Error retrieving GitHub repository: {repo_data['error']}")
        else:
            parts.append(f"Repository: {repo_data.get('full_name', 'Unknown')}")
            parts.append(f"Description: {repo_data.get('description', 'No description')}")
            parts.append(f"Stars: {repo_data.get('stargazers_count', 0)}")
            parts.append(f"Forks: {repo_data.get('forks_count', 0)}")
            parts.append(f"Language: {repo_data.get('language', 'Not specified')}")
            
            if "readme" in repo_data:
                parts.append(f"\nReadme:")
                parts.append(repo_data["readme"])
        
        parts.append("")  # Add blank line
    
    # Add URL content if present
    for key, value in content.items():
        if key.startswith("url_"):
            url = key[len("url_"):]
            parts.append(f"Content from URL: {url}")
            
            if isinstance(value, dict) and "error" in value:
                parts.append(f"Error retrieving URL content: {value['error']}")
            else:
                # Limit content length
                content_text = str(value)
                if len(content_text) > 2000:
                    content_text = content_text[:2000] + "... (content truncated)"
                
                parts.append(content_text)
            
            parts.append("")  # Add blank line
    
    # Add Twitter content if present
    for key, value in content.items():
        if key.startswith("twitter_"):
            url = key[len("twitter_"):]
            parts.append(f"Twitter Content from URL: {url}")
            
            if isinstance(value, dict) and "error" in value:
                parts.append(f"Error retrieving Twitter content: {value['error']}")
            elif isinstance(value, dict):
                tweet_text = value.get("text", "No tweet text available")
                parts.append(f"Tweet: {tweet_text}")
                
                user_name = value.get("user", {}).get("name", "Unknown")
                username = value.get("user", {}).get("username", "Unknown")
                parts.append(f"Posted by: {user_name} (@{username})")
                
                if "thread_tweets" in value and value["thread_tweets"]:
                    parts.append("\nThread:")
                    for tweet in value["thread_tweets"]:
                        parts.append(f"- {tweet.get('text', 'No text')}")
            
            parts.append("")  # Add blank line
    
    # Add channel info if present
    if "channel_info" in content:
        channel_data = content["channel_info"]
        parts.append("Channel Information:")
        
        if isinstance(channel_data, dict) and "error" in channel_data:
            parts.append(f"Error retrieving channel info: {channel_data['error']}")
        else:
            parts.append(f"Channel: {channel_data.get('name', 'Unknown')}")
            parts.append(f"Topic: {channel_data.get('topic', {}).get('value', 'No topic')}")
            parts.append(f"Purpose: {channel_data.get('purpose', {}).get('value', 'No purpose')}")
            parts.append(f"Members: {channel_data.get('num_members', 0)}")
        
        parts.append("")  # Add blank line
    
    # Add image information if present
    if recent_images and len(recent_images) > 0:
        parts.append("Recent Images in Channel:")
        
        for i, image in enumerate(recent_images):
            parts.append(f"Image {i+1}:")
            parts.append(f"Posted by: {image.get('user', 'Unknown')}")
            parts.append(f"URL: {image.get('url', 'No URL')}")
            
            # Add the image for vision analysis
            image_urls.append({
                "url": image.get('url'),
                "description": f"Image {i+1} from the channel"
            })
        
        parts.append("")  # Add blank line
    
    # Build the final context
    context_parts = "\n".join(parts)
    
    # Use prompt template from YAML config
    context_template = PROMPTS.get("build_response_context", {}).get("context_template", "")
    
    # Fallback if not in YAML
    if not context_template:
        context_template = """
        User query: {query}
        
        {content_parts}
        
        Based on the above information, please provide a helpful, concise, and accurate response to the user's query.
        """
    
    # Format the context
    context = context_template.format(
        query=query,
        content_parts=context_parts
    )
    
    logger.info(f"Built context with {len(context)} characters")
    return context, image_urls

async def call_ai(context: str, image_urls: List[Dict[str, Any]] = None):
    """
    Call the AI model with the full context and any images that need to be analyzed.
    Properly handles Slack's private image URLs by downloading and base64 encoding them.
    """
    logger.info("Calling AI with context")
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Get system prompts from YAML
        system_text_prompt = PROMPTS.get("call_ai", {}).get("system_text", "")
        system_vision_prompt = PROMPTS.get("call_ai", {}).get("system_vision", "")
        
        # Check if we have images to include in the request
        if image_urls and len(image_urls) > 0:
            logger.info(f"Including {len(image_urls)} images in AI request")
            
            # Create a multimodal message array
            user_message_content = [
                {"type": "text", "text": context}
            ]
            
            # Process and add images (up to 4 to keep request size manageable)
            for img in image_urls[:4]:
                if img.get("url"):
                    logger.info(f"Processing image: {img.get('title', 'Image')}")
                    
                    # Download and prepare the image
                    image_content = await prepare_image_for_openai(img["url"])
                    
                    if image_content:
                        logger.info(f"Successfully prepared image for OpenAI")
                        user_message_content.append(image_content)
                    else:
                        logger.error(f"Failed to prepare image: {img.get('title', 'Image')}")
            
            # Only proceed with vision model if we successfully prepared at least one image
            if len(user_message_content) > 1:
                # Make multimodal API call
                response = client.chat.completions.create(
                    model=AI_MODEL,
                    messages=[
                        {"role": "system", "content": system_vision_prompt},
                        {"role": "user", "content": user_message_content}
                    ]
                )
            else:
                # If all images failed, fall back to text-only
                logger.warning("All images failed to prepare. Falling back to text-only request.")
                response = client.chat.completions.create(
                    model=AI_MODEL,
                    messages=[
                        {"role": "system", "content": system_text_prompt},
                        {"role": "user", "content": context + "\n\nNote: There were images in the conversation, but they couldn't be processed."}
                    ]
                )
        else:
            # Text-only request
            response = client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": system_text_prompt},
                    {"role": "user", "content": context}
                ]
            )
        
        ai_response = response.choices[0].message.content
        logger.info(f"AI response received: {ai_response}")
        return ai_response
    except Exception as e:
        error_message = f"Error calling AI: {str(e)}"
        logger.error(error_message)
        return f"Sorry, I encountered an error analyzing the content: {str(e)}"

async def evaluate_search_parameters(query: str, content_analysis: ContentAnalysisResult):
    """
    Evaluate how to search channel history based on the query and initial content analysis.
    Determines search depth, relevance criteria, and time range.
    """
    logger.info("Evaluating search parameters for channel history")
    
    # If channel history isn't required, return minimal parameters
    if not content_analysis.requires_history:
        logger.info("Channel history search not required - returning minimal parameters")
        return {
            "time_range": "days",
            "time_value": 1,
            "message_count": 0,  # Set to 0 to skip search
            "keywords": [],
            "users": [],
            "resource_types": []
        }
    
    try:
        # Get the prompt templates from YAML
        system_prompt = PROMPTS.get("evaluate_search_parameters", {}).get("system", "")
        user_prompt_template = PROMPTS.get("evaluate_search_parameters", {}).get("user_template", "")
        
        # Format the user prompt
        user_prompt = user_prompt_template.format(
            query=query,
            content_analysis=content_analysis.dict()
        )
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        search_params = json.loads(response.choices[0].message.content)
        logger.info(f"Search parameters determined: {search_params}")
        
        return search_params
    
    except Exception as e:
        logger.error(f"Error determining search parameters: {str(e)}")
        # Return default parameters
        return {
            "time_range": "days",
            "message_count": 50,
            "relevance_keywords": [word.lower() for word in query.split() if len(word) > 3],
            "user_focus": None,
            "resource_types": ["text"]
        }

async def evaluate_search_results(query: str, search_results: List[Dict[str, Any]], content_analysis: ContentAnalysisResult):
    """
    Evaluate search results to find the most relevant messages, URLs, and files.
    Now with enhanced image identification.
    """
    logger.info(f"Evaluating {len(search_results)} search results against query: {query[:50]}...")
    
    try:
        # Extract all URLs from the search results for reference
        all_urls = []
        for msg in search_results:
            urls = extract_urls(msg.get("text", ""))
            if urls:
                logger.info(f"Extracted {len(urls)} URLs: {urls}")
                all_urls.extend(urls)
                
        # Format messages for evaluation, including file information
        formatted_messages = []
        for i, msg in enumerate(search_results):
            message_info = {
                "index": i,
                "text": msg.get("text", ""),
                "has_files": msg.get("has_files", False),
                "has_images": msg.get("has_images", False),
                "has_urls": msg.get("has_urls", False),
                "has_code": msg.get("has_code", False),
            }
            
            # Include file details for images
            if message_info["has_images"] and "files" in msg:
                message_info["files"] = []
                for file_idx, file in enumerate(msg["files"]):
                    if file.get("mimetype", "").startswith("image/"):
                        file_info = {
                            "index": file_idx,
                            "type": "image",
                            "name": file.get("title", "Image"),
                            "url": file.get("url_private", "")
                        }
                        message_info["files"].append(file_info)
                        
            formatted_messages.append(message_info)
        
        # Get the prompt templates from YAML
        system_prompt = PROMPTS.get("evaluate_search_results", {}).get("system", "")
        user_prompt_template = PROMPTS.get("evaluate_search_results", {}).get("user_template", "")
        
        # Format the user prompt
        user_prompt = user_prompt_template.format(
            query=query,
            formatted_messages=json.dumps(formatted_messages, indent=2)
        )
        
        # Get AI to evaluate the relevance of each message
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        logger.info(f"Evaluation result: identified {len(result.get('relevant_message_indices', []))} relevant messages")
        logger.info(f"Evaluation result: {result}")
        
        # Build the structured evaluation result
        relevant_messages = []
        for idx in result.get("relevant_message_indices", []):
            if idx < len(search_results):
                relevant_messages.append(search_results[idx])
                
        # Extract and add image URLs
        image_urls = []
        for msg_idx in result.get("relevant_message_indices", []):
            if msg_idx < len(search_results) and "files" in search_results[msg_idx]:
                for file in search_results[msg_idx]["files"]:
                    if file.get("mimetype", "").startswith("image/"):
                        image_urls.append({
                            "url": file.get("url_private", ""),
                            "title": file.get("title", "Image"),
                            "from_user": search_results[msg_idx].get("user", "Unknown")
                        })
        
        evaluation_result = {
            "relevant_messages": relevant_messages,
            "image_urls": image_urls,  # Add image URLs directly to evaluation result
            "key_insights": result.get("key_insights", [])
        }
        
        return evaluation_result
        
    except Exception as e:
        logger.error(f"Error evaluating search results: {str(e)}")
        # Return a simplified result if evaluation fails
        return {"relevant_messages": search_results[:5]}

async def determine_required_content_sources(query: str, search_results: List[Dict[str, Any]], content_analysis: ContentAnalysisResult):
    """
    Analyze search results and determine which external content sources need to be fetched.
    This optimizes API calls by only fetching what's actually needed.
    
    Returns:
        Dict with boolean flags for each content source
    """
    logger.info("Determining required content sources based on search results")
    
    # Start fresh based on what we need for this specific query
    required_sources = {
        "linear": False,
        "github": False,
        "url_content": False,
        "twitter": False,
        "channel_info": False,
        "linear_working_hours": False,  # For working hours data
        "linear_query": False           # For generalized Linear queries
    }
    
    # Check for general Linear query directly
    if getattr(content_analysis, "linear_query", False):
        logger.info("General Linear query detected, will fetch Linear data")
        required_sources["linear_query"] = True
    
    # Check for working hours query directly
    if getattr(content_analysis, "linear_working_hours_query", False) or content_analysis.content_type == "linear_working_hours":
        logger.info("Linear working hours query detected, will fetch working hours data")
        required_sources["linear_working_hours"] = True
        
    # Working hours keywords that indicate we should check working hours
    working_hours_keywords = [
        "hours", "working", "time tracking", "timesheet", "logged",
        "completed", "tracking", "time", "week", "required"
    ]
    
    # If query contains "who hasn't" or similar with working hours keywords
    if (("who" in query.lower() and "not" in query.lower()) or 
        "who hasn't" in query.lower() or 
        "hasn't completed" in query.lower() or
        "didn't complete" in query.lower()) and any(keyword in query.lower() for keyword in working_hours_keywords):
        
        logger.info("Query asking about who hasn't completed working hours")
        required_sources["linear_working_hours"] = True
    
    # If no search results, check if we need content from the original query
    if not search_results:
        # Only include sources specifically mentioned in the user's query
        required_sources["linear"] = content_analysis.linear_issue is not None
        required_sources["github"] = content_analysis.github_repo is not None
        required_sources["url_content"] = content_analysis.urls is not None and len(content_analysis.urls) > 0
        required_sources["twitter"] = content_analysis.twitter_urls is not None and len(content_analysis.twitter_urls) > 0
        
        # Preserve the working hours and general Linear query flags we set earlier
        if getattr(content_analysis, "linear_working_hours_query", False):
            required_sources["linear_working_hours"] = True
        if getattr(content_analysis, "linear_query", False):
            required_sources["linear_query"] = True
            
        return required_sources
    
    try:
        # Extract URLs from the search results
        urls = []
        for msg in search_results:
            if msg.get("text"):
                extracted_urls = extract_urls(msg["text"])
                if extracted_urls:
                    urls.extend(extracted_urls)
        
        # Get the prompt templates from YAML
        system_prompt = PROMPTS.get("determine_required_content_sources", {}).get("system", "")
        user_prompt_template = PROMPTS.get("determine_required_content_sources", {}).get("user_template", "")
        
        # Create a summary of what the query seems to be about
        is_article = "article" in query.lower() or "link" in query.lower() or "url" in query.lower()
        is_code = "code" in query.lower() or "github" in query.lower() or "repo" in query.lower()
        is_issue = "issue" in query.lower() or "ticket" in query.lower() or "linear" in query.lower()
        is_tweet = "tweet" in query.lower() or "twitter" in query.lower() or "x.com" in query.lower()
        is_working_hours = "hours" in query.lower() or "time" in query.lower() or any(keyword in query.lower() for keyword in working_hours_keywords)
        is_linear_query = getattr(content_analysis, "linear_query", False) or "linear" in query.lower()
        
        # Format the user prompt
        user_prompt = user_prompt_template.format(
            query=query,
            urls=urls,
            is_article=is_article,
            is_code=is_code,
            is_issue=is_issue,
            is_tweet=is_tweet,
            is_working_hours=is_working_hours,
            is_linear_query=is_linear_query
        )
        
        # Get AI recommendation
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        ai_recommendations = json.loads(response.choices[0].message.content)
        logger.info(f"AI recommendations for content sources: {ai_recommendations}")
        
        # Use the AI recommendations directly instead of merging
        for source, needed in ai_recommendations.items():
            source_key = source
            if source == "linear_issue":
                source_key = "linear"
            if source_key in required_sources:
                required_sources[source_key] = needed
        
        # Always preserve the linear_query flag if it was set during analysis
        if getattr(content_analysis, "linear_query", False):
            required_sources["linear_query"] = True
        
        logger.info(f"Final required content sources: {required_sources}")
        return required_sources
        
    except Exception as e:
        logger.error(f"Error determining content sources: {str(e)}")
        # Fall back to minimal analysis if anything fails, but preserve working hours flag
        fallback_sources = {
            "linear_issue": False,
            "github_repo": False,
            "url_content": True,  # Default to fetching URLs since that's usually helpful
            "twitter": False,
            "channel_info": False,
            "linear_working_hours": required_sources.get("linear_working_hours", False),  # Preserve this value
            "linear_query": required_sources.get("linear_query", False)  # Preserve generalized Linear query flag
        }
        return fallback_sources

def detect_image_related_query(query: str) -> bool:
    """Detect if a query is related to images or visual content."""
    # Keywords that suggest the user is asking about an image
    image_keywords = [
        "image", "picture", "photo", "pic", "screenshot", "chart", "graph",
        "see", "look", "show", "display", "visual", "diagram", 
    ]
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Check for image keywords
    for keyword in image_keywords:
        if keyword in query_lower:
            return True

    
    return False

async def get_recent_messages(channel_id: str, limit: int = 10):
    """Fetch the most recent messages from a channel."""
    try:
        result = slack_client.conversations_history(
            channel=channel_id,
            limit=limit
        )
        
        if not result["ok"]:
            logger.error(f"Error fetching recent messages: {result.get('error')}")
            return []
            
        return result["messages"]
    except Exception as e:
        logger.error(f"Error fetching recent messages: {str(e)}")
        return []

def extract_images_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract image information from messages."""
    images = []
    
    for msg in messages:
        # Skip if message doesn't have files
        if "files" not in msg:
            continue
            
        # Look for image files
        for file in msg["files"]:
            # Check if file is an image
            if file.get("mimetype", "").startswith("image/"):
                # Add image with context
                images.append({
                    "url": file.get("url_private", ""),
                    "title": file.get("title", "Image"),
                    "from_user": msg.get("user", "Unknown"),
                    "text": msg.get("text", ""),
                    "ts": msg.get("ts", ""),
                    "permalink": file.get("permalink", ""),
                    "file_id": file.get("id", "")
                })
    
    return images

async def download_image_from_slack(image_url: str) -> Optional[bytes]:
    """
    Download an image from Slack with the proper authorization.
    Returns the image bytes that can be sent to OpenAI.
    """
    logger.info(f"Downloading image from Slack: {image_url[:50]}...")
    
    try:
        headers = {
            "Authorization": f"Bearer {SLACK_BOT_TOKEN}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, headers=headers) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Failed to download image from Slack: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error downloading image from Slack: {str(e)}")
        return None

async def prepare_image_for_openai(image_url: str) -> Optional[Dict[str, str]]:
    """
    Prepare an image from Slack for sending to OpenAI.
    Downloads the image and converts it to a base64 encoded string.
    """
    try:
        # Download the image
        image_bytes = await download_image_from_slack(image_url)
        if not image_bytes:
            return None
            
        # Encode as base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        # Return properly formatted for OpenAI
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            }
        }
    except Exception as e:
        logger.error(f"Error preparing image for OpenAI: {str(e)}")
        return None

async def delete_animation_messages(message_timestamps):
    """Delete animation messages from Slack."""
    logger.info(f"Deleting {len(message_timestamps)} animation messages")
    
    try:
        for channel_id, ts in message_timestamps:
            if channel_id and ts:
                try:
                    # Delete the message
                    slack_client.chat_delete(
                        channel=channel_id,
                        ts=ts
                    )
                    logger.info(f"Deleted message with ts: {ts}")
                except SlackApiError as e:
                    logger.error(f"Error deleting message: {e.response['error']}")
    except Exception as e:
        logger.error(f"Error in delete_animation_messages: {str(e)}")

# Load prompts from YAML file
try:
    with open('prompts.yaml', 'r', encoding='utf-8') as file:
        PROMPTS = yaml.safe_load(file)
    logger.info("Successfully loaded prompts from prompts.yaml")
except Exception as e:
    logger.error(f"Error loading prompts: {str(e)}")
    # Fallback to empty dict
    PROMPTS = {}

# Linear GraphQL Schema - Simplified for AI query generation
LINEAR_GRAPHQL_SCHEMA = """
type Query {
  # Teams
  teams: TeamConnection
  team(id: ID!): Team
  
  # Users
  users: UserConnection
  user(id: ID!): User
  me: User
  
  # Issues
  issues(filter: IssueFilter): IssueConnection
  issue(id: ID!): Issue
  
  # Cycles
  cycles(filter: CycleFilter): CycleConnection
  cycle(id: ID!): Cycle
}

type Team {
  id: ID!
  name: String!
  key: String!
  description: String
  members: UserConnection
  issues(filter: IssueFilter): IssueConnection
  cycles(filter: CycleFilter): CycleConnection
  states: WorkflowStateConnection
}

type User {
  id: ID!
  name: String!
  email: String!
  displayName: String
  teams: TeamConnection
  assignedIssues(filter: IssueFilter): IssueConnection
  createdIssues(filter: IssueFilter): IssueConnection
  timeEntries(filter: TimeEntryFilter): TimeEntryConnection
}

type Issue {
  id: ID!
  title: String!
  description: String
  state: WorkflowState
  creator: User
  assignee: User
  team: Team
  labels: IssueLabelConnection
  cycle: Cycle
  createdAt: DateTime!
  updatedAt: DateTime!
  completedAt: DateTime
  estimate: Float
  timeSpent: TimeEntry
  timeTracked: Int
}

type Cycle {
  id: ID!
  number: Int!
  name: String!
  team: Team!
  startsAt: DateTime!
  endsAt: DateTime!
  completedAt: DateTime
  issues: IssueConnection
  progress: Float
  scopeTarget: Int
}

type TimeEntry {
  id: ID!
  date: DateTime!
  seconds: Int!
  issue: Issue
  user: User
}

# Common filters and connections
type IssueFilter {
  team: TeamFilter
  assignee: UserFilter
  cycle: CycleFilter
  state: WorkflowStateFilter
  updatedAt: DateComparator
  createdAt: DateComparator
  completedAt: DateComparator
}

type CycleFilter {
  team: TeamFilter
  isActive: Boolean
  isCompleted: Boolean
  startDate: DateComparator
  endDate: DateComparator
}

type DateComparator {
  gt: DateTime
  gte: DateTime
  lt: DateTime
  lte: DateTime
  eq: DateTime
}

# Additional connection types
type TeamConnection {
  nodes: [Team!]!
  totalCount: Int!
}

type UserConnection {
  nodes: [User!]!
  totalCount: Int!
}

type IssueConnection {
  nodes: [Issue!]!
  totalCount: Int!
}

type CycleConnection {
  nodes: [Cycle!]!
  totalCount: Int!
}

type TimeEntryConnection {
  nodes: [TimeEntry!]!
  totalCount: Int!
}

type WorkflowStateConnection {
  nodes: [WorkflowState!]!
}

type WorkflowState {
  id: ID!
  name: String!
  type: String!
}

type IssueLabelConnection {
  nodes: [IssueLabel!]!
}

type IssueLabel {
  id: ID!
  name: String!
}
"""

def extract_query_and_variables(graphql_query_string: str) -> Dict[str, Any]:
    """
    Extract the query and variables from a GraphQL query string.
    
    Parameters:
    - graphql_query_string: The full GraphQL query string
    
    Returns:
    - Dict containing 'query' and 'variables'
    """
    logger.info("Extracting query and variables from GraphQL string")
    
    # Default empty variables dict
    variables = {}
    
    # Handle case when nothing is provided
    if not graphql_query_string or graphql_query_string.strip() == "":
        logger.error("Empty GraphQL query string provided")
        return {"query": "", "variables": {}}
    
    try:
        # Check if the string contains a variables block
        variables_match = re.search(r'variables\s*:\s*({[\s\S]*})', graphql_query_string, re.DOTALL)
        if variables_match:
            try:
                variables_str = variables_match.group(1).strip()
                variables = json.loads(variables_str)
                # Remove variables section from query
                graphql_query_string = graphql_query_string.replace(variables_match.group(0), "").strip()
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse variables from query: {variables_match.group(1)} - Error: {str(e)}")
        
        # Look for query definition - various formats supported
        query_match = re.search(r'(query\s*\w*\s*\(.*?\)\s*{[\s\S]*}|{[\s\S]*})', graphql_query_string, re.DOTALL)
        if query_match:
            query = query_match.group(1).strip()
        else:
            # If no query definition found, try to use the whole string as a query
            query = graphql_query_string.strip()
            
            # Add surrounding braces if they're missing
            if not query.startswith("{"):
                query = "{ " + query
            if not query.endswith("}"):
                query = query + " }"
                
            logger.info(f"Reformatted query: {query[:100]}...")
        
        # Clean up query - remove comments
        query = re.sub(r'#.*\n', '\n', query)
        
        # Final validation
        if not query or query.strip() == "":
            logger.error("Failed to extract a valid query")
            return {"query": "", "variables": {}}
            
        return {
            "query": query,
            "variables": variables
        }
    except Exception as e:
        logger.error(f"Error extracting query and variables: {str(e)}")
        return {"query": "", "variables": {}}

async def transform_query_to_graphql(user_query: str, previous_results: Optional[Dict] = None, previous_evaluation: Optional[Dict] = None) -> str:
    """
    Transform a natural language query into a GraphQL query for Linear.
            
    Parameters:
    - user_query: The natural language query from the user
    - previous_results: Results from previous attempts (if any)
    - previous_evaluation: Evaluation of previous results (if any)
    
    Returns:
    - String containing the GraphQL query
    """
    logger.info("Transforming user query to GraphQL")
    
    # Prepare context for the AI
    context = {
        "user_query": user_query,
        "linear_schema": LINEAR_GRAPHQL_SCHEMA,
        "previous_attempts": []
    }
    
    # Add information about previous attempts if available
    if previous_results and previous_evaluation:
        context["previous_attempts"].append({
            "graphql_query": previous_results.get("graphql_query"),
            "results": previous_results,
            "evaluation": previous_evaluation,
            "issues": previous_evaluation.get("issues", [])
        })
    
    # Get prompts from YAML config
    system_prompt = PROMPTS.get("transform_query_to_graphql", {}).get("system", "")
    user_prompt_template = PROMPTS.get("transform_query_to_graphql", {}).get("user_template", "")
    
    if not system_prompt or not user_prompt_template:
        logger.warning("Missing prompts for transform_query_to_graphql, using fallback prompts")
        system_prompt = "You are an expert in transforming natural language queries into GraphQL queries for the Linear API."
        user_prompt_template = "User's question: {query}\n\nLinear GraphQL Schema:\n{schema}\n\n{previous_attempts}\n\nGenerate a complete GraphQL query."
    
    # Format previous attempts text if any
    previous_attempts_text = ""
    if context["previous_attempts"]:
        previous_attempts_text = "Previous attempts and their issues:\n" + json.dumps(context["previous_attempts"], indent=2)
    
    # Format the user prompt
    user_prompt = user_prompt_template.format(
        query=user_query,
        schema=LINEAR_GRAPHQL_SCHEMA,
        previous_attempts=previous_attempts_text
    )
    
    try:
        # Call OpenAI
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Extract the query
        graphql_query = response.choices[0].message.content.strip()
        
        # Clean up the query - remove markdown code blocks if present
        graphql_query = re.sub(r'^```graphql\s*', '', graphql_query)
        graphql_query = re.sub(r'^```\s*', '', graphql_query)
        graphql_query = re.sub(r'\s*```$', '', graphql_query)
        
        # Log a sample of the generated query
        logger.info(f"Generated GraphQL query (first 100 chars): {graphql_query[:100]}...")
        
        if not graphql_query or graphql_query.strip() == "":
            logger.error("Empty GraphQL query generated")
            raise ValueError("Failed to generate a valid GraphQL query")
        
        return graphql_query
    
    except Exception as e:
        logger.error(f"Error transforming query to GraphQL: {str(e)}")
        # Return a very simple fallback query to avoid complete failure
        return "{ teams { nodes { id name } } }"

async def execute_graphql_query(graphql_query: str) -> Dict[str, Any]:
    """
    Execute a GraphQL query against the Linear API.
    
    Parameters:
    - graphql_query: The GraphQL query to execute
    
    Returns:
    - Dict containing the query results
    """
    logger.info("Executing GraphQL query")
    
    if not linear_client:
        raise Exception("Linear client not initialized")
    
    try:
        # Format the query and extract variables if present
        query_parts = extract_query_and_variables(graphql_query)
        query = query_parts["query"]
        variables = query_parts["variables"]
        
        logger.info(f"Extracted query: {query[:100]}...")
        logger.info(f"Extracted variables: {variables}")
        
        # Execute the query
        response = requests.post(
            linear_client.api_url,
            json={"query": query, "variables": variables},
            headers=linear_client.headers
        )
        
        if response.status_code != 200:
            logger.error(f"GraphQL query failed: {response.text}")
            raise Exception(f"GraphQL query failed: {response.text}")
            
        results = response.json()
        logger.info(f"Query executed successfully")
        
        # Check for errors in the response
        if "errors" in results:
            logger.error(f"GraphQL errors: {results['errors']}")
            raise Exception(f"GraphQL errors: {json.dumps(results['errors'])}")
        
        return results
    except Exception as e:
        logger.error(f"Error executing GraphQL query: {str(e)}")
        raise

async def evaluate_query_results(user_query: str, graphql_query: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate if the query results satisfy the user's original query.
    
    Parameters:
    - user_query: The original natural language query
    - graphql_query: The GraphQL query that was executed
    - results: The results from the GraphQL query
    
    Returns:
    - Dict containing evaluation results
    """
    logger.info("Evaluating query results")
    
    # Get prompts from YAML config
    system_prompt = PROMPTS.get("evaluate_query_results", {}).get("system", "")
    user_prompt_template = PROMPTS.get("evaluate_query_results", {}).get("user_template", "")
    
    if not system_prompt or not user_prompt_template:
        logger.warning("Missing prompts for evaluate_query_results, using fallback prompts")
        system_prompt = "You are an expert evaluator of API responses."
        user_prompt_template = "User's question: {user_query}\n\nGraphQL query:\n{graphql_query}\n\nResults:\n{results_json}\n\nEvaluate if these results answer the question."
    
    # Format the user prompt
    user_prompt = user_prompt_template.format(
        user_query=user_query,
        graphql_query=graphql_query,
        results_json=json.dumps(results, indent=2)
    )
    
    # Call OpenAI
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    # Extract the evaluation
    evaluation = json.loads(response.choices[0].message.content)
    logger.info(f"Evaluation result: {evaluation}")
    
    return evaluation

async def format_linear_results(user_query: str, results: Dict[str, Any]) -> str:
    """
    Format Linear results in a human-readable way.
    
    Parameters:
    - user_query: The original natural language query
    - results: The raw results from the GraphQL query
    
    Returns:
    - Formatted string for display
    """
    logger.info("Formatting Linear results")
    
    # Get prompts from YAML config
    system_prompt = PROMPTS.get("format_linear_results", {}).get("system", "")
    user_prompt_template = PROMPTS.get("format_linear_results", {}).get("user_template", "")
    
    if not system_prompt or not user_prompt_template:
        logger.warning("Missing prompts for format_linear_results, using fallback prompts")
        system_prompt = "You are an expert in data presentation."
        user_prompt_template = "User's question: {user_query}\n\nAPI results:\n{results_json}\n\nFormat these results clearly to answer the question."
    
    # Format the user prompt
    user_prompt = user_prompt_template.format(
        user_query=user_query,
        results_json=json.dumps(results, indent=2)
    )
    
    # Call OpenAI
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Return the formatted results
    formatted_results = response.choices[0].message.content
    logger.info(f"Formatted results: {formatted_results[:100]}...")
    return formatted_results

async def retrieve_linear_data(user_query: str, max_attempts: int = 3):
    """
    General-purpose function to retrieve data from Linear based on a natural language query.
    
    Uses AI to:
    1. Transform the user query into a GraphQL query
    2. Execute the GraphQL query against Linear API
    3. Evaluate results and retry if needed
    
    Parameters:
    - user_query: Natural language query about Linear data
    - max_attempts: Maximum number of retry attempts
    
    Returns:
    - Dict containing the query results and metadata
    """
    logger.info(f"Retrieving Linear data for query: {user_query}")
    
    # Track attempts and results
    attempts = 0
    results = None
    is_satisfied = False
    evaluation = None
    graphql_query = None
    error = None
    all_attempts = []
    
    while attempts < max_attempts and not is_satisfied:
        attempts += 1
        logger.info(f"Attempt {attempts}/{max_attempts}")
        
        try:
            # Step 1: Transform user query into GraphQL query
            try:
                graphql_query = await transform_query_to_graphql(
                    user_query, 
                    results, 
                    evaluation
                )
                logger.info(f"GraphQL query successfully generated, length: {len(graphql_query)}")
            except Exception as transform_error:
                logger.error(f"Error transforming query: {str(transform_error)}")
                error = f"Error generating GraphQL query: {str(transform_error)}"
                # Fall back to a simple query
                graphql_query = "{ teams { nodes { id name } } }"
                logger.info(f"Using fallback query: {graphql_query}")
            
            # Step 2: Execute the GraphQL query
            try:
                results = await execute_graphql_query(graphql_query)
                logger.info("GraphQL query executed successfully")
            except Exception as execute_error:
                logger.error(f"Error executing query: {str(execute_error)}")
                error = f"Error executing GraphQL query: {str(execute_error)}"
                # Provide minimal results to continue
                results = {"data": {}, "error": str(execute_error)}
            
            # Step 3: Evaluate if the results satisfy the original query
            try:
                evaluation = await evaluate_query_results(
                    user_query, 
                    graphql_query, 
                    results
                )
                logger.info(f"Query evaluation completed: {evaluation.get('is_satisfied', False)}")
            except Exception as eval_error:
                logger.error(f"Error evaluating results: {str(eval_error)}")
                error = f"Error evaluating query results: {str(eval_error)}"
                # Create a minimal evaluation to continue
                evaluation = {
                    "is_satisfied": False,
                    "reason": f"Error during evaluation: {str(eval_error)}",
                    "missing_data": ["Error occurred during evaluation"],
                    "suggestions": ["Try a more specific query"]
                }
            
            # Track this attempt
            all_attempts.append({
                "attempt": attempts,
                "graphql_query": graphql_query,
                "evaluation": evaluation,
                "error": error
            })
            
            # Check if we're satisfied with the results
            is_satisfied = evaluation.get("is_satisfied", False)
            
            if is_satisfied:
                logger.info("Query successfully satisfied")
            else:
                reason = evaluation.get("reason", "Unknown reason")
                logger.info(f"Query not satisfied: {reason}")
                
                # On the last attempt, accept what we have if there's any data
                if attempts == max_attempts and results and "data" in results and results["data"]:
                    logger.info("Last attempt - accepting results even though not fully satisfied")
                    is_satisfied = True
                
        except Exception as e:
            error = str(e)
            logger.error(f"Unexpected error retrieving Linear data: {error}")
            
            # Track this attempt
            all_attempts.append({
                "attempt": attempts,
                "graphql_query": graphql_query,
                "error": error
            })
            
            # Break on error unless it's the first attempt
            if attempts > 1:
                break
    
    # Format the final result
    formatted_results = None
    if results and (is_satisfied or attempts == max_attempts):
        try:
            formatted_results = await format_linear_results(user_query, results)
            logger.info("Results formatted successfully")
        except Exception as format_error:
            logger.error(f"Error formatting results: {str(format_error)}")
            formatted_results = f"Error formatting results: {str(format_error)}\n\nRaw data: {json.dumps(results.get('data', {}), indent=2)}"
    
    # Always return something, even if all attempts failed
    return {
        "original_query": user_query,
        "graphql_query": graphql_query,
        "results": results or {"data": {}},
        "evaluation": evaluation or {"is_satisfied": False, "reason": "Maximum attempts reached"},
        "formatted_results": formatted_results or f"Could not retrieve data after {attempts} attempts. Please try a more specific query.",
        "attempts": attempts,
        "all_attempts": all_attempts,
        "success": is_satisfied,
        "error": error
    }

# Helper functions for Linear GraphQL queries

async def list_teams() -> List[Dict[str, Any]]:
    """
    Get a list of all teams in Linear.
    
    Returns:
    - List of team objects with id and name
    """
    logger.info("Fetching list of teams from Linear")
    
    query = """
    query {
      teams {
        nodes {
          id
          name
          key
          description
        }
      }
    }
    """
    
    try:
        result = await universal_graphql_executor(query)
        teams = result.get("data", {}).get("teams", {}).get("nodes", [])
        logger.info(f"Retrieved {len(teams)} teams from Linear")
        return teams
    except Exception as e:
        logger.error(f"Error fetching teams: {str(e)}")
        return []

async def list_cycles_for_team(team_id: str, include_inactive: bool = False) -> List[Dict[str, Any]]:
    """
    Get a list of cycles for a specific team.
    
    Parameters:
    - team_id: The Linear team ID
    - include_inactive: Whether to include cycles that aren't active
    
    Returns:
    - List of cycle objects
    """
    logger.info(f"Fetching cycles for team {team_id}")
    
    # Build filter based on whether we want inactive cycles
    filter_clause = ""
    if not include_inactive:
        filter_clause = ", filter: { completedAt: null }"
    
    query = """
    query($teamId: String!) {
      team(id: $teamId) {
        cycles(first: 10""" + filter_clause + """, orderBy: createdAt) {
          nodes {
            id
            number
            name
            startsAt
            endsAt
            completedAt
            description
            progress
          }
        }
      }
    }
    """
    
    variables = {"teamId": team_id}
    
    try:
        result = await universal_graphql_executor(query, variables)
        cycles = result.get("data", {}).get("team", {}).get("cycles", {}).get("nodes", [])
        logger.info(f"Retrieved {len(cycles)} cycles for team {team_id}")
        return cycles
    except Exception as e:
        logger.error(f"Error fetching cycles for team {team_id}: {str(e)}")
        return []

async def get_active_cycle_for_team(team_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the currently active cycle for a team.
    
    Parameters:
    - team_id: The Linear team ID
    
    Returns:
    - The active cycle object or None if no active cycle
    """
    logger.info(f"Getting active cycle for team {team_id}")
    
    cycles = await list_cycles_for_team(team_id, include_inactive=False)
    
    # Find the active cycle (current date is between startsAt and endsAt)
    now = datetime.datetime.utcnow().isoformat()
    
    for cycle in cycles:
        starts_at = cycle.get("startsAt")
        ends_at = cycle.get("endsAt")
        
        if starts_at and ends_at:
            if starts_at <= now <= ends_at:
                logger.info(f"Found active cycle {cycle.get('number')} for team {team_id}")
                return cycle
    
    # If we couldn't find an active cycle, return the most recent one
    if cycles:
        logger.info(f"No active cycle found, returning most recent cycle {cycles[0].get('number')}")
        return cycles[0]
    
    logger.warning(f"No cycles found for team {team_id}")
    return None

async def list_issues_for_cycle(team_id: str, cycle_id: str) -> List[Dict[str, Any]]:
    """
    Get a list of issues for a specific cycle.
    
    Parameters:
    - team_id: The Linear team ID
    - cycle_id: The Linear cycle ID
    
    Returns:
    - List of issue objects
    """
    logger.info(f"Fetching issues for cycle {cycle_id} in team {team_id}")
    
    query = """
    query($teamId: ID!, $cycleId: ID!) {
      team(id: $teamId) {
        issues(filter: { cycle: { id: { eq: $cycleId } } }) {
          nodes {
            id
            title
            description
            state {
              name
              type
              color
            }
            assignee {
              id
              name
              email
            }
            priority
            estimate
            createdAt
            updatedAt
            completedAt
          }
        }
      }
    }
    """
    
    variables = {
        "teamId": team_id,
        "cycleId": cycle_id
    }
    
    try:
        result = await universal_graphql_executor(query, variables)
        issues = result.get("data", {}).get("team", {}).get("issues", {}).get("nodes", [])
        logger.info(f"Retrieved {len(issues)} issues for cycle {cycle_id}")
        return issues
    except Exception as e:
        logger.error(f"Error fetching issues for cycle {cycle_id}: {str(e)}")
        return []

async def search_teams_by_name(name: str) -> List[Dict[str, Any]]:
    """
    Search for teams by name.
    
    Parameters:
    - name: The team name to search for
    
    Returns:
    - List of matching team objects
    """
    logger.info(f"Searching for teams with name: {name}")
    
    # Get all teams and filter locally
    all_teams = await list_teams()
    
    # Case-insensitive search
    name_lower = name.lower()
    matching_teams = []
    
    for team in all_teams:
        team_name = team.get("name", "").lower()
        team_key = team.get("key", "").lower()
        
        # Check for exact match or contained match
        if team_name == name_lower or name_lower in team_name or name_lower == team_key:
            matching_teams.append(team)
    
    logger.info(f"Found {len(matching_teams)} teams matching '{name}'")
    return matching_teams

class LinearQueryPlanner:
    """
    Planner/Orchestrator for Linear queries that handles resolving ambiguities
    in natural language queries and building complete GraphQL queries.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("linear_query_planner")
    
    async def process_query(self, user_query: str, additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a natural language query about Linear data.
        
        Parameters:
        - user_query: The natural language query from the user
        - additional_context: Any additional context that might help resolve ambiguities
        
        Returns:
        - Dictionary with the query results and metadata
        """
        self.logger.info(f"Processing Linear query: {user_query}")
        
        # Step 1: Analyze the query to identify what information we need
        query_analysis = await self._analyze_query(user_query)
        
        # Step 2: Resolve ambiguities in the query
        resolved_params = await self._resolve_ambiguities(query_analysis, additional_context)
        
        # Store the original query and additional context in resolved_params
        resolved_params["original_query"] = user_query
        if additional_context:
            resolved_params["additional_context"] = additional_context
        
        # Step 3: Build and execute the final query
        if not resolved_params.get("error"):
            results = await self._execute_final_query(resolved_params)
        else:
            # Handle unresolvable ambiguities
            return {
                "success": False,
                "error": resolved_params.get("error"),
                "partial_results": resolved_params.get("partial_results"),
                "clarification_needed": resolved_params.get("clarification_needed")
            }
        
        # Step 4: Format the results
        formatted_results = await self._format_results(user_query, results, resolved_params)
        
        return {
            "success": True,
            "original_query": user_query,
            "resolved_params": resolved_params,
            "raw_results": results,
            "formatted_results": formatted_results
        }
    
    async def _analyze_query(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze the query to identify what entities and information we need to resolve.
        
        Parameters:
        - user_query: The natural language query from the user
        
        Returns:
        - Dictionary with analysis results
        """
        self.logger.info("Analyzing query to identify required information")
        
        # Get prompts from YAML config
        system_prompt = """
        You are an AI that analyzes user queries about Linear project management data.
        Your job is to identify what information we need to retrieve from Linear to answer the query.
        
        For any query, identify:
        1. What type of information is being requested (teams, cycles, issues, etc.)
        2. Any specific entities mentioned (team names, cycle numbers, user names, etc.)
        3. Any ambiguous or relative references that need resolution ("this cycle", "engineering team", etc.)
        
        Return your analysis as a JSON object with these fields without any additional text:
        - query_type: The main type of information being requested (team_info, cycle_info, issues_list, etc.)
        - entities: Object containing any specific entities mentioned (team_name, cycle_number, user_name, etc.)
        - ambiguities: Array of strings describing any ambiguous references that need resolution
        - parameters_needed: Array of parameter types needed to fulfill this query (team_id, cycle_id, etc.)
        """
        
        user_prompt = f"Analyze this Linear project management query: \"{user_query}\""
        
        try:
            # Call the AI to analyze the query
            response = await call_openai(system_prompt, user_prompt)
            
            # Parse the AI's response
            try:
                # Check if the response is wrapped in markdown code block
                if response.startswith("```json") and response.endswith("```"):
                    # Remove the markdown code block markers
                    json_text = response[len("```json"):-len("```")]
                else:
                    # Already raw JSON
                    json_text = response
                
                analysis = json.loads(json_text)
                self.logger.info(f"Query analysis results: {analysis}")
                return analysis
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse AI response as JSON: {response}")
                return {
                    "query_type": "unknown",
                    "entities": {},
                    "ambiguities": ["Failed to parse query"],
                    "parameters_needed": []
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing query: {str(e)}")
            return {
                "query_type": "unknown",
                "entities": {},
                "ambiguities": ["Error analyzing query"],
                "parameters_needed": []
            }
    
    async def _resolve_ambiguities(self, query_analysis: Dict[str, Any], additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Resolve ambiguities identified in the query analysis.
        
        Parameters:
        - query_analysis: The analysis of the query
        - additional_context: Any additional context that might help resolve ambiguities
        
        Returns:
        - Dictionary with resolved parameters or error information
        """
        self.logger.info("Resolving ambiguities in query")
        
        resolved_params = {}
        clarification_needed = []
        
        if not additional_context:
            additional_context = {}
        
        # Extract needed parameters and entities from analysis
        parameters_needed = query_analysis.get("parameters_needed", [])
        entities = query_analysis.get("entities", {})
        
        # Resolve team references
        if "team_id" in parameters_needed:
            team_name = entities.get("team_name")
            if team_name:
                matching_teams = await search_teams_by_name(team_name)
                
                if len(matching_teams) == 1:
                    # Found exactly one match
                    resolved_params["team_id"] = matching_teams[0]["id"]
                    resolved_params["team_name"] = matching_teams[0]["name"]
                    self.logger.info(f"Resolved team '{team_name}' to ID: {resolved_params['team_id']}")
                elif len(matching_teams) > 1:
                    # Multiple matches found, need clarification
                    clarification_needed.append({
                        "type": "team_selection",
                        "message": f"Multiple teams match '{team_name}'. Please specify which one you meant.",
                        "options": matching_teams
                    })
                    self.logger.info(f"Multiple teams match '{team_name}', clarification needed")
                else:
                    # No matches found
                    clarification_needed.append({
                        "type": "team_not_found",
                        "message": f"No teams found matching '{team_name}'. Please provide a valid team name."
                    })
                    self.logger.info(f"No teams match '{team_name}'")
            else:
                # No team specified, provide list of all teams
                all_teams = await list_teams()
                clarification_needed.append({
                    "type": "team_selection",
                    "message": "Please specify which team you're interested in.",
                    "options": all_teams
                })
                self.logger.info("No team specified, clarification needed")
        
        # Resolve cycle references
        if "cycle_id" in parameters_needed and "team_id" in resolved_params:
            cycle_number = entities.get("cycle_number")
            cycle_reference = entities.get("cycle_reference", "").lower()
            
            # Check if additional_context contains cycles for this team
            team_name = resolved_params.get("team_name")
            context_cycles = None
            
            if additional_context and "cycles" in additional_context and team_name in additional_context["cycles"]:
                context_cycles = additional_context["cycles"][team_name]
                self.logger.info(f"Found {len(context_cycles)} cycles for {team_name} in context")
            
            if cycle_number:
                # Specific cycle number mentioned
                cycles = await list_cycles_for_team(resolved_params["team_id"], include_inactive=True)
                matching_cycles = [c for c in cycles if str(c.get("number")) == str(cycle_number)]
                
                if matching_cycles:
                    resolved_params["cycle_id"] = matching_cycles[0]["id"]
                    resolved_params["cycle_number"] = matching_cycles[0]["number"]
                    self.logger.info(f"Resolved cycle #{cycle_number} to ID: {resolved_params['cycle_id']}")
                else:
                    # No need for explicit clarification - the AI can handle this with context
                    self.logger.info(f"No cycle #{cycle_number} found, but context will be provided to AI")
            elif cycle_reference in ["current", "active", "this"] or any("this cycle" in ambiguity.lower() for ambiguity in query_analysis.get("ambiguities", [])):
                # Get current cycle information, but don't worry if we can't find it exactly
                # The context added to the prompt will help the AI figure it out
                active_cycle = await get_active_cycle_for_team(resolved_params["team_id"])
                
                if active_cycle:
                    resolved_params["cycle_id"] = active_cycle["id"]
                    resolved_params["cycle_number"] = active_cycle["number"]
                    self.logger.info(f"Using active cycle #{active_cycle['number']} for context")
                else:
                    # The AI will use the context to interpret "this cycle"
                    self.logger.info("No active cycle found, but context will be provided to AI")
            
            # If we couldn't determine a specific cycle, don't worry
            # The AI with context can handle references like "this cycle" or "current cycle"
        
        # Check if we need team_id but couldn't resolve it
        if "team_id" in parameters_needed and "team_id" not in resolved_params:
            return {
                "error": "Could not resolve team reference",
                "clarification_needed": clarification_needed,
                "partial_results": {"teams": await list_teams()}
            }
        
        # Check if we need cycle_id but couldn't resolve it
        if "cycle_id" in parameters_needed and "cycle_id" not in resolved_params and "team_id" in resolved_params:
            return {
                "error": "Could not resolve cycle reference",
                "clarification_needed": clarification_needed,
                "partial_results": {"cycles": await list_cycles_for_team(resolved_params["team_id"], include_inactive=True)}
            }
        
        # Include query_type in resolved params
        resolved_params["query_type"] = query_analysis.get("query_type", "unknown")
        
        return resolved_params
    
    async def _execute_final_query(self, resolved_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build and execute the final GraphQL query based on resolved parameters.
        
        Parameters:
        - resolved_params: The resolved parameters for the query
        
        Returns:
        - Dictionary with query results
        """
        self.logger.info("Building and executing final query")
        
        query_type = resolved_params.get("query_type", "unknown")
        
        if query_type == "team_info":
            # Query for team information
            if "team_id" in resolved_params:
                team_id = resolved_params["team_id"]
                
                query = """
                query($teamId: String!) {
                  team(id: $teamId) {
                    id
                    name
                    key
                    description
                    members {
                      nodes {
                        id
                        name
                        email
                      }
                    }
                    activeCycle {
                      id
                      number
                      name
                      startsAt
                      endsAt
                      progress
                    }
                    issues(first: 10, orderBy: createdAt) {
                      nodes {
                        id
                        title
                        state {
                          name
                        }
                      }
                    }
                  }
                }
                """
                
                variables = {"teamId": team_id}
                
                try:
                    result = await universal_graphql_executor(query, variables)
                    self.logger.info("Successfully executed team_info query")
                    return result
                except Exception as e:
                    self.logger.error(f"Error executing team_info query: {str(e)}")
                    return {"error": str(e)}
        
        elif query_type == "cycle_info":
            # Query for cycle information
            if "team_id" in resolved_params and "cycle_id" in resolved_params:
                team_id = resolved_params["team_id"]
                cycle_id = resolved_params["cycle_id"]
                
                query = """
                query($teamId: String!, $cycleId: String!) {
                  team(id: $teamId) {
                    id
                    name
                    key
                    cycle(id: $cycleId) {
                      id
                      number
                      name
                      startsAt
                      endsAt
                      progress
                      completedAt
                      description
                      issues {
                        nodes {
                          id
                          title
                          description
                          state {
                            name
                            type
                          }
                          assignee {
                            name
                          }
                          priority
                          estimate
                        }
                      }
                    }
                  }
                }
                """
                
                variables = {
                    "teamId": team_id,
                    "cycleId": cycle_id
                }
                
                try:
                    result = await universal_graphql_executor(query, variables)
                    self.logger.info("Successfully executed cycle_info query")
                    return result
                except Exception as e:
                    self.logger.error(f"Error executing cycle_info query: {str(e)}")
                    return {"error": str(e)}
        
        elif query_type == "issues_list":
            # Query for a list of issues
            if "team_id" in resolved_params:
                team_id = resolved_params["team_id"]
                
                # Create variables dictionary
                variables = {"teamId": team_id}
                
                # Build filter clause
                filter_clauses = []
                
                # Add cycle filter if present
                if "cycle_id" in resolved_params:
                    filter_clauses.append("cycle: { id: { eq: $cycleId } }")
                    variables["cycleId"] = resolved_params["cycle_id"]
                
                # Add date filter for "this week" if present in the query
                query_has_this_week = False
                for ambiguity in resolved_params.get("ambiguities", []):
                    if "this week" in ambiguity.lower():
                        query_has_this_week = True
                        break
                
                original_query = resolved_params.get("original_query", "")
                if "this week" in original_query.lower() or query_has_this_week:
                    self.logger.info("Adding 'this week' date filter to the query")
                    
                    # For "this week" we can use a simpler approach with a predefined duration
                    filter_clauses.append("updatedAt: { gte: \"P0W\", lte: \"P1W\" }")
                
                # Also filter to only show non-completed issues by default
                filter_clauses.append("state: { type: { neq: \"completed\" } }")
                
                # Combine filter clauses if any
                filter_string = ""
                if filter_clauses:
                    filter_string = f", filter: {{ {', '.join(filter_clauses)} }}"
                
                # Build the query with all necessary variables
                variable_defs = ["$teamId: String!"]
                if "cycleId" in variables:
                    variable_defs.append("$cycleId: String!")
                
                query = f"""
                query({', '.join(variable_defs)}) {{
                  team(id: $teamId) {{
                    id
                    name
                    key
                    issues(first: 50{filter_string}) {{
                      nodes {{
                        id
                        title
                        description
                        state {{
                          name
                          type
                        }}
                        assignee {{
                          name
                        }}
                        priority
                        estimate
                        createdAt
                        updatedAt
                        completedAt
                      }}
                    }}
                  }}
                }}
                """
                
                try:
                    result = await universal_graphql_executor(query, variables)
                    self.logger.info("Successfully executed issues_list query")
                    self.logger.info(f"Query results: {result}")
                    
                    # Log the number of issues returned
                    if "data" in result and "team" in result["data"] and "issues" in result["data"]["team"]:
                        issues = result["data"]["team"]["issues"]["nodes"]
                        self.logger.info(f"Retrieved {len(issues)} issues for team {team_id}")
                    else:
                        self.logger.info(f"No issues found for team {team_id}")
                    
                    return result
                except Exception as e:
                    self.logger.error(f"Error executing issues_list query: {str(e)}")
                    return {"error": str(e)}
        
        # Default case - couldn't determine the appropriate query
        self.logger.warning(f"Unknown query type: {query_type}")
        return {"error": f"Unknown query type: {query_type}"}
    
    async def _format_results(self, user_query: str, results: Dict[str, Any], resolved_params: Dict[str, Any]) -> str:
        """
        Format the query results into a human-readable response.
        
        Parameters:
        - user_query: The original user query
        - results: The raw query results
        - resolved_params: The resolved parameters used for the query
        
        Returns:
        - Formatted response string
        """
        self.logger.info("Formatting query results")
        
        if "error" in results:
            return f"Sorry, I encountered an error: {results['error']}"
        
        # Get the query type to determine how to format the results
        query_type = resolved_params.get("query_type", "unknown")
        original_query = resolved_params.get("original_query", user_query)
        additional_context = resolved_params.get("additional_context", {})
        
        try:
            # Get prompts from YAML config
            system_prompt = """
            You are an AI assistant that formats Linear project management data in a clear, readable way.
            
            Your task is to:
            1. Analyze the raw data returned by a GraphQL query
            2. Format it in a way that directly answers the user's original question
            3. Highlight the most important information
            4. Organize the data logically with appropriate structure
            5. Keep your response concise while including all relevant details
            
            Remember: Do not explain the data structure; focus on presenting the information in a useful way.
            """
            
            # Add context information to the prompt
            context_str = ""
            if additional_context and "cycles" in additional_context:
                context_str += "Available cycles information:\n"
                for team_name, cycles in additional_context["cycles"].items():
                    context_str += f"\n{team_name} cycles:\n"
                    for cycle in cycles:
                        progress = f"{int(cycle['progress'] * 100)}%" if cycle.get('progress') is not None else "Unknown"
                        context_str += f"- Cycle #{cycle['number']}: {progress} complete, {cycle.get('start_date', 'Unknown')} to {cycle.get('end_date', 'Unknown')}\n"
            
            user_prompt = f"""
            Format these Linear project management data results to answer the user's question.
            
            Original question: "{original_query}"
            
            {context_str}
            
            Raw data:
            {json.dumps(results.get('data', {}), indent=2)}
            
            Format the data in a clear, readable way that directly answers the user's question.
            Use appropriate organization but avoid Markdown formatting as this will be displayed in Slack.
            """
            
            # Call OpenAI
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract the formatted results
            formatted_results = response.choices[0].message.content.strip()
            logger.info(f"Formatted results: {formatted_results[:100]}...")
            return formatted_results
        
        except Exception as e:
            self.logger.error(f"Error formatting results: {str(e)}")
            return "Error formatting results"

async def call_openai(system_prompt: str, user_prompt: str) -> str:
    """
    Call the OpenAI API to get a response.
    
    Parameters:
    - system_prompt: The system prompt for the AI
    - user_prompt: The user prompt for the AI
    
    Returns:
    - The AI's response text
    """
    logger.info("Calling OpenAI API")
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        raise

async def process_linear_query(query: str) -> Dict[str, Any]:
    """
    Process a Linear query using the new LinearQueryPlanner.

    Parameters:
    - query: The user's natural language query about Linear data

    Returns:
    - Dictionary with the query results and various metadata
    """
    logger.info(f"Processing Linear query with LinearQueryPlanner: {query}")
    
    try:
        # Step 1: Create an instance of the query planner
        planner = LinearQueryPlanner()
        
        # Step 2: Get basic information about all teams and active cycles for context
        try:
            teams = await list_teams()
            logger.info(f"Retrieved {len(teams)} teams for context")
            
            # Get active cycles for the top teams
            cycles_context = {}
            for team in teams[:5]:  # Limit to top 5 teams to avoid too many API calls
                team_id = team["id"]
                team_name = team["name"]
                cycles = await list_cycles_for_team(team_id, include_inactive=False)
                
                # Filter to just active and recent cycles
                active_cycles = []
                for cycle in cycles[:3]:  # Just get the 3 most recent cycles
                    active_cycles.append({
                        "id": cycle.get("id"),
                        "number": cycle.get("number"),
                        "progress": cycle.get("progress", 0),
                        "start_date": cycle.get("startsAt"),
                        "end_date": cycle.get("endsAt")
                    })
                
                if active_cycles:
                    cycles_context[team_name] = active_cycles
            
            logger.info(f"Retrieved cycle information for {len(cycles_context)} teams")
            
            # Add context to additional_context
            additional_context = {
                "teams": teams,
                "cycles": cycles_context
            }
        except Exception as e:
            logger.warning(f"Error getting additional context for Linear query: {str(e)}")
            additional_context = {}
        
        # Step 3: Process the query with context
        result = await planner.process_query(query, additional_context)
        
        logger.info(f"LinearQueryPlanner processed query successfully: {result.get('success', False)}")
        return result
    except Exception as e:
        logger.error(f"Error processing Linear query with LinearQueryPlanner: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "formatted_results": f"Error processing Linear query: {str(e)}"
        }

async def universal_graphql_executor(query: str, variables: dict = None) -> Dict[str, Any]:
    """
    Universal function to execute any GraphQL query against the Linear API.
    
    Parameters:
    - query: The GraphQL query string
    - variables: Optional variables to pass with the query
    
    Returns:
    - Dictionary containing the response data or error information
    """
    logger.info(f"Executing GraphQL query with universal executor")
    logger.info(f"Query length: {len(query)}")
    if variables:
        logger.info(f"Variables: {json.dumps(variables)}")
    
    if not LINEAR_API_KEY:
        raise Exception("LINEAR_API_KEY is not configured")
    
    # Ensure we have valid variables
    if variables is None:
        variables = {}
    
    try:
        headers = {
            "Authorization": LINEAR_API_KEY,
            "Content-Type": "application/json"
        }
        
        # Prepare the request payload
        payload = {
            "query": query,
            "variables": variables
        }
        
        # Make the request
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.linear.app/graphql", json=payload, headers=headers) as response:
                # Parse the response
                response_data = await response.json()
                
                # Check response status
                if response.status != 200:
                    logger.error(f"GraphQL request failed with status {response.status}: {response_data}")
                    raise Exception(f"GraphQL request failed with status {response.status}: {json.dumps(response_data)}")
                
                # Check for errors in the response
                if "errors" in response_data:
                    logger.error(f"GraphQL errors: {response_data['errors']}")
                    raise Exception(f"GraphQL errors: {json.dumps(response_data['errors'])}")
                
                logger.info("GraphQL query executed successfully")
                
                # Add detailed logging for debugging
                if "data" in response_data and "team" in response_data["data"] and "issues" in response_data["data"]["team"]:
                    issues = response_data["data"]["team"]["issues"]["nodes"]
                    logger.info(f"Query returned {len(issues)} issues")
                    
                    # Log a sample of the first few issues for debugging
                    if issues:
                        sample_size = min(3, len(issues))
                        sample_issues = issues[:sample_size]
                        sample_data = []
                        for issue in sample_issues:
                            sample_data.append({
                                'title': issue.get('title', 'No title'),
                                'state': issue.get('state', {}).get('name', 'Unknown'),
                                'assignee': issue.get('assignee', {}).get('name', 'Unassigned')
                            })
                        logger.info(f"Sample issues: {json.dumps(sample_data, indent=2)}")
                    
                logger.info(f"Linear API response: {response_data}")
                return response_data
                
    except Exception as e:
        logger.error(f"Error executing GraphQL query: {str(e)}")
        # Re-raise the exception to be handled by the caller
        raise

async def get_channel_info(channel_id: str) -> Dict[str, Any]:
    """
    Get information about a Slack channel.
    
    Parameters:
    - channel_id: The Slack channel ID
    
    Returns:
    - Dictionary with channel information
    """
    logger.info(f"Getting info for channel {channel_id}")
    
    try:
        # Call the Slack API to get channel info
        response = slack_client.conversations_info(channel=channel_id)
        
        if response and response.get("ok"):
            channel_data = response.get("channel", {})
            
            # Extract relevant information
            channel_info = {
                "id": channel_data.get("id"),
                "name": channel_data.get("name"),
                "is_private": channel_data.get("is_private", False),
                "topic": channel_data.get("topic"),
                "purpose": channel_data.get("purpose"),
                "created": channel_data.get("created"),
                "creator": channel_data.get("creator"),
                "num_members": channel_data.get("num_members")
            }
            
            logger.info(f"Successfully got info for channel {channel_id}")
            return channel_info
        else:
            error_message = response.get("error", "Unknown error")
            logger.error(f"Error getting channel info: {error_message}")
            return {"error": f"Failed to get channel info: {error_message}"}
            
    except SlackApiError as e:
        logger.error(f"Slack API error getting channel info: {str(e)}")
        return {"error": f"Slack API error: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error getting channel info: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}

def process_tweet_url(url: str) -> ProcessedContent:
    """
    Process a Twitter/X URL to extract tweet information with improved error handling.
    
    Parameters:
    - url: The Twitter/X URL
    
    Returns:
    - ProcessedContent object with tweet information
    """
    logger.info(f"Processing tweet URL: {url}")
    
    try:
        # Extract tweet ID and user ID from the URL
        # For Twitter URLs like: https://twitter.com/username/status/1234567890
        match = re.search(r"twitter\.com/([^/]+)/status/(\d+)", url)
        if not match:
            # Try X URLs: https://x.com/username/status/1234567890
            match = re.search(r"x\.com/([^/]+)/status/(\d+)", url)
            
        if not match:
            logger.error(f"Could not extract tweet ID from URL: {url}")
            return ProcessedContent(
                url=url,
                text_content=f"Could not extract tweet information from URL: {url}",
                is_tweet=False
            )
        
        username = match.group(1)
        tweet_id = match.group(2)
        
        # If we don't have Twitter credentials, just return basic info
        if not TWITTER_BEARER_TOKEN:
            logger.warning("No Twitter credentials available. Returning basic info.")
            return ProcessedContent(
                url=url,
                text_content=f"Tweet by @{username} (ID: {tweet_id}). Cannot fetch content without Twitter API credentials.",
                is_tweet=True,
                tweet_id=tweet_id,
                user_id=username
            )
        
        # Initialize Twitter client
        client = tweepy.Client(TWITTER_BEARER_TOKEN)
        
        try:
            # Fetch the tweet with timeout and retry logic
            for attempt in range(3):  # Try up to 3 times
                try:
                    response = client.get_tweet(
                        id=tweet_id,
                        expansions=["author_id", "attachments.media_keys", "referenced_tweets.id"],
                        tweet_fields=["created_at", "public_metrics", "text"],
                        user_fields=["name", "username", "profile_image_url"],
                        timeout=5  # 5 second timeout
                    )
                    break  # Exit the retry loop if successful
                except tweepy.TweepyException as e:
                    if attempt < 2:  # Don't sleep on the last attempt
                        time.sleep(1)  # Wait 1 second before retrying
                    else:
                        raise  # Re-raise the exception on the last attempt
            
            if not response or not response.data:
                logger.error(f"Failed to fetch tweet {tweet_id} - empty response")
                return ProcessedContent(
                    url=url,
                    text_content=f"Failed to fetch tweet {tweet_id}. The tweet may be private or deleted.",
                    is_tweet=True,
                    tweet_id=tweet_id,
                    user_id=username
                )
            
            # Process the tweet data
            tweet = response.data
            tweet_text = tweet.text if hasattr(tweet, "text") else "No text available"
            
            # Extract user information
            user = None
            if response.includes and "users" in response.includes:
                user = response.includes["users"][0]
            
            # Create a simplified version with essential info
            return ProcessedContent(
                url=url,
                text_content=tweet_text,
                is_tweet=True,
                tweet_id=tweet_id,
                user_id=user.id if user else username,
                metadata={
                    "author": user.username if user else username,
                    "created_at": tweet.created_at.isoformat() if hasattr(tweet, "created_at") else None,
                    "likes": tweet.public_metrics.get("like_count", 0) if hasattr(tweet, "public_metrics") else 0,
                    "retweets": tweet.public_metrics.get("retweet_count", 0) if hasattr(tweet, "public_metrics") else 0
                }
            )
            
        except tweepy.TweepyException as e:
            logger.error(f"Tweepy error processing tweet {tweet_id}: {str(e)}")
            # Return basic info with error context
            return ProcessedContent(
                url=url,
                text_content=f"Tweet by @{username} (ID: {tweet_id}). Error fetching content: {str(e)}",
                is_tweet=True,
                tweet_id=tweet_id,
                user_id=username
            )
            
    except Exception as e:
        logger.error(f"Error processing tweet URL {url}: {str(e)}")
        return ProcessedContent(
            url=url,
            text_content=f"Error processing tweet URL: {str(e)}",
            is_tweet=False
        )

def get_channel_info(client, channel_id):
    """
    Get information about a Slack channel.
    
    Parameters:
    - client: Slack client instance
    - channel_id: The ID of the channel to get information for
    
    Returns:
    - Dictionary containing channel information
    """
    logger.info(f"Getting channel info for {channel_id}")
    
    try:
        # For public and private channels
        try:
            response = client.conversations_info(channel=channel_id)
            if response["ok"]:
                return response["channel"]
        except Exception as e:
            logger.debug(f"Could not get conversation info: {str(e)}")
        
        # For direct messages
        try:
            response = client.conversations_open(users=channel_id)
            if response["ok"]:
                return response["channel"]
        except Exception as e:
            logger.debug(f"Could not open conversation: {str(e)}")
        
        # For multi-person direct messages
        try:
            response = client.conversations_open(channel=channel_id)
            if response["ok"]:
                return response["channel"]
        except Exception as e:
            logger.debug(f"Could not open channel: {str(e)}")
        
        logger.error(f"Failed to get channel info for {channel_id}")
        return None
    
    except Exception as e:
        logger.error(f"Error getting channel info for {channel_id}: {str(e)}")
        return None

def get_thread_tweets(client, user_id, tweet_id, max_tweets=10):
    """
    Get tweets in a thread.
    
    Parameters:
    - client: Twitter client instance
    - user_id: ID of the user who posted the tweets
    - tweet_id: ID of the tweet in the thread
    - max_tweets: Maximum number of tweets to fetch
    
    Returns:
    - List of tweets in the thread
    """
    logger.info(f"Getting thread tweets for tweet {tweet_id} from user {user_id}")
    
    try:
        # Get the conversation ID for the tweet
        tweet_response = client.get_tweet(
            id=tweet_id,
            tweet_fields=["conversation_id"]
        )
        
        if not tweet_response or not tweet_response.data:
            logger.error(f"Failed to fetch tweet {tweet_id}")
            return None
        
        tweet = tweet_response.data
        
        if not hasattr(tweet, "conversation_id"):
            logger.error(f"Tweet {tweet_id} has no conversation_id field")
            return None
        
        conversation_id = tweet.conversation_id
        
        # Get all tweets in the conversation from the same user
        tweets_response = client.get_users_tweets(
            id=user_id,
            max_results=max_tweets,
            tweet_fields=["created_at", "text", "public_metrics"],
        )
        
        if not tweets_response or not tweets_response.data:
            logger.error(f"Failed to fetch tweets for user {user_id}")
            return None
        
        # Filter tweets that are part of the same conversation
        thread_tweets = []
        for t in tweets_response.data:
            if hasattr(t, "conversation_id") and t.conversation_id == conversation_id:
                thread_tweets.append({
                    "id": t.id,
                    "text": t.text,
                    "created_at": t.created_at.isoformat() if hasattr(t, "created_at") else None,
                })
        
        # Sort by ID (which is chronological for tweets)
        thread_tweets.sort(key=lambda x: x["id"])
        
        return thread_tweets
    
    except Exception as e:
        logger.error(f"Error getting thread tweets for {tweet_id}: {str(e)}")
        return None

if __name__ == "__main__":
    logger.info("Starting AI Bot server")
    uvicorn.run(app, host="0.0.0.0", port=8000)

