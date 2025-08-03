#!/usr/bin/env python3
"""
Simple script to delete the latest bot message from specific Slack channels.
Supports both top-level messages and thread replies.
"""

import os
import sys
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def delete_latest_bot_message():
    """Delete bot thread replies from specified channels."""
    
    # Initialize Slack client
    slack_token = os.environ.get("SLACK_BOT_TOKEN")
    if not slack_token:
        print("Error: SLACK_BOT_TOKEN not found in environment variables")
        return False
    
    client = WebClient(token=slack_token)
    
    # Get bot user ID
    try:
        auth_response = client.auth_test()
        bot_user_id = auth_response["user_id"]
        print(f"Bot user ID: {bot_user_id}")
    except SlackApiError as e:
        print(f"Error getting bot user ID: {e.response['error']}")
        return False
    
    # Channel IDs to clean up
    channels = {
        "Engineering test Channel": "C07QK3HB9V2"
    }
    
    total_success_count = 0
    
    for channel_name, channel_id in channels.items():
        print(f"\nProcessing {channel_name} ({channel_id})...")
        
        try:
            # Get recent messages from the channel
            response = client.conversations_history(
                channel=channel_id,
                limit=100  # Get more messages to find all bot messages
            )
            
            if not response["ok"]:
                print(f"Error getting messages: {response.get('error', 'Unknown error')}")
                continue
            
            messages = response["messages"]
            
            # Method 1: Find bot thread replies from conversations_history
            bot_thread_replies = []
            thread_timestamps = set()  # Track threads we've seen
            
            for message in messages:
                # Collect all thread timestamps to check later
                if "thread_ts" in message:
                    thread_timestamps.add(message["thread_ts"])
                
                # Find bot replies in threads
                if (message.get("user") == bot_user_id and 
                    "thread_ts" in message and 
                    message["thread_ts"] != message["ts"]):
                    bot_thread_replies.append(message)
            
            print(f"Found {len(bot_thread_replies)} bot thread replies from recent messages")
            
            # Method 2: Check each thread individually for more bot messages
            print(f"Checking {len(thread_timestamps)} individual threads for additional bot messages...")
            
            for thread_ts in thread_timestamps:
                try:
                    thread_response = client.conversations_replies(
                        channel=channel_id,
                        ts=thread_ts,
                        limit=100
                    )
                    
                    if thread_response["ok"]:
                        thread_messages = thread_response["messages"]
                        
                        for thread_message in thread_messages:
                            # Skip the parent message and messages we already found
                            if (thread_message.get("user") == bot_user_id and 
                                "thread_ts" in thread_message and 
                                thread_message["thread_ts"] != thread_message["ts"] and
                                thread_message not in bot_thread_replies):
                                
                                # Check if we already have this message (avoid duplicates)
                                message_exists = any(existing["ts"] == thread_message["ts"] for existing in bot_thread_replies)
                                if not message_exists:
                                    bot_thread_replies.append(thread_message)
                                    print(f"Found additional bot reply in thread {thread_ts}")
                    
                except SlackApiError as e:
                    print(f"Error checking thread {thread_ts}: {e.response['error']}")
                    continue
            
            if not bot_thread_replies:
                print(f"No bot thread replies found in {channel_name}")
                continue
            
            print(f"Found {len(bot_thread_replies)} total bot thread reply(ies) in {channel_name}")
            
            # Process each bot thread reply
            channel_success_count = 0
            for i, message in enumerate(bot_thread_replies):
                message_ts = message["ts"]
                thread_ts = message["thread_ts"]
                message_text = message.get("text", "")[:100] + "..." if len(message.get("text", "")) > 100 else message.get("text", "")
                
                print(f"\nThread Reply {i+1}/{len(bot_thread_replies)}:")
                print(f"Message: '{message_text}'")
                print(f"Message timestamp: {message_ts}")
                print(f"Thread timestamp: {thread_ts}")
                
                # Confirm deletion
                confirm = input(f"Delete this thread reply from {channel_name}? (y/N): ").strip().lower()
                if confirm != 'y':
                    print("Skipped.")
                    continue
                
                try:
                    delete_response = client.chat_delete(
                        channel=channel_id,
                        ts=message_ts
                    )
                    
                    if delete_response["ok"]:
                        print(f"✅ Successfully deleted thread reply from {channel_name}")
                        channel_success_count += 1
                    else:
                        print(f"❌ Failed to delete thread reply from {channel_name}: {delete_response.get('error', 'Unknown error')}")
                        
                except SlackApiError as e:
                    print(f"❌ Error deleting thread reply: {e.response['error']}")
                    continue
            
            total_success_count += channel_success_count
            print(f"\n📊 Channel Summary: Deleted {channel_success_count}/{len(bot_thread_replies)} bot thread replies from {channel_name}")
                
        except SlackApiError as e:
            print(f"❌ Error processing {channel_name}: {e.response['error']}")
            continue
    
    print(f"\n🎉 Successfully deleted {total_success_count} bot thread replies total")
    return total_success_count > 0

def list_bot_thread_messages(channel_id, client, bot_user_id):
    """Helper function to detect and list bot messages in threads."""
    bot_thread_messages = []
    
    # Get recent messages
    response = client.conversations_history(channel=channel_id, limit=100)
    if not response["ok"]:
        return bot_thread_messages
    
    messages = response["messages"]
    thread_timestamps = set()
    
    # Find thread replies and collect thread timestamps
    for message in messages:
        if "thread_ts" in message:
            thread_timestamps.add(message["thread_ts"])
        
        # Bot message in thread (not the parent)
        if (message.get("user") == bot_user_id and 
            "thread_ts" in message and 
            message["thread_ts"] != message["ts"]):
            bot_thread_messages.append({
                "ts": message["ts"],
                "thread_ts": message["thread_ts"],
                "text": message.get("text", ""),
                "source": "history"
            })
    
    # Check each thread individually
    for thread_ts in thread_timestamps:
        try:
            thread_response = client.conversations_replies(
                channel=channel_id, 
                ts=thread_ts
            )
            
            if thread_response["ok"]:
                for thread_message in thread_response["messages"]:
                    if (thread_message.get("user") == bot_user_id and 
                        "thread_ts" in thread_message and 
                        thread_message["thread_ts"] != thread_message["ts"]):
                        
                        # Avoid duplicates
                        if not any(existing["ts"] == thread_message["ts"] for existing in bot_thread_messages):
                            bot_thread_messages.append({
                                "ts": thread_message["ts"],
                                "thread_ts": thread_message["thread_ts"],
                                "text": thread_message.get("text", ""),
                                "source": "thread_replies"
                            })
        except SlackApiError:
            continue
    
    return bot_thread_messages

if __name__ == "__main__":
    print("🤖 Slack Bot Message Cleanup Script")
    print("=" * 40)
    
    # Warning
    print("⚠️  WARNING: This will delete bot thread replies from:")
    print("   - Engineering test Channel")
    print("   (Only deletes bot replies to threads, not bot's own thread starters)")
    print()
    
    confirm = input("Are you sure you want to proceed? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled.")
        sys.exit(0)
    
    success = delete_latest_bot_message()
    
    if success:
        print("\n✅ All messages deleted successfully!")
    else:
        print("\n⚠️  Some messages could not be deleted. Check the output above.") 