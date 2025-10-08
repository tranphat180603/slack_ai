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
    """Delete the most recent bot message from specified channels."""
    
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
        "Marketing channel": "C07QK3HB9V2",
        "Product channel": "C07C44USZKR",
        "TM Moonshots channel": "C092DANQ5RT",
        "TM API channel": "C07F3SD76EA",
        "blocked": "C09D3N6EJ4D"
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
            
            # Find normal bot messages (top-level messages or thread starters)
            bot_messages = []
            
            for message in messages:
                # Find bot messages that are NOT thread replies
                # (either no thread_ts or thread_ts == ts which means it's a thread starter)
                if (message.get("user") == bot_user_id and 
                    ("thread_ts" not in message or message.get("thread_ts") == message["ts"])):
                    bot_messages.append(message)
            
            if not bot_messages:
                print(f"No bot messages found in {channel_name}")
                continue
            
            print(f"Found {len(bot_messages)} bot message(s) in {channel_name}")
            
            # Sort messages by timestamp (most recent first) and take only the most recent
            if bot_messages:
                bot_messages.sort(key=lambda x: float(x["ts"]), reverse=True)
                latest_message = bot_messages[0]
                
                message_ts = latest_message["ts"]
                message_text = latest_message.get("text", "")[:100] + "..." if len(latest_message.get("text", "")) > 100 else latest_message.get("text", "")
                
                print(f"\nMost Recent Bot Message:")
                print(f"Message: '{message_text}'")
                print(f"Message timestamp: {message_ts}")
                
                # Check if it's a thread starter
                if "thread_ts" in latest_message and latest_message["thread_ts"] == latest_message["ts"]:
                    print("Type: Thread starter")
                else:
                    print("Type: Regular message")
                
                # Confirm deletion
                confirm = input(f"Delete this most recent message from {channel_name}? (y/N): ").strip().lower()
                if confirm == 'y':
                    try:
                        delete_response = client.chat_delete(
                            channel=channel_id,
                            ts=message_ts
                        )
                        
                        if delete_response["ok"]:
                            print(f"✅ Successfully deleted most recent message from {channel_name}")
                            total_success_count += 1
                        else:
                            print(f"❌ Failed to delete message from {channel_name}: {delete_response.get('error', 'Unknown error')}")
                            
                    except SlackApiError as e:
                        print(f"❌ Error deleting message: {e.response['error']}")
                else:
                    print("Skipped.")
            
            print(f"\n📊 Channel Summary: Processed most recent bot message from {channel_name}")
                
        except SlackApiError as e:
            print(f"❌ Error processing {channel_name}: {e.response['error']}")
            continue
    
    print(f"\n🎉 Successfully deleted {total_success_count} most recent bot message(s) total")
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
    print("⚠️  WARNING: This will delete the MOST RECENT bot message from:")
    print("   - Marketing channel")
    print("   - Product channel") 
    print("   - TM Moonshots channel")
    print("   - TM API channel")
    print("   (Deletes most recent normal bot message or thread starter, not thread replies)")
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