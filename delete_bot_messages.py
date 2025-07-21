#!/usr/bin/env python3
"""
Simple script to delete the latest bot message from specific Slack channels.
"""

import os
import sys
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def delete_latest_bot_message():
    """Delete the latest bot message from specified channels."""
    
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
        "Marketing Channel": "C07D7F5531N",
        "Product Channel": "C07C44USZKR",
        "TM API Channel": "C07F3SD76EA",
        "TM Moonshot Channel": "C092DANQ5RT"
    }
    
    success_count = 0
    
    for channel_name, channel_id in channels.items():
        print(f"\nProcessing {channel_name} ({channel_id})...")
        
        try:
            # Get recent messages from the channel
            response = client.conversations_history(
                channel=channel_id,
                limit=50  # Get last 50 messages to find the latest bot message
            )
            
            if not response["ok"]:
                print(f"Error getting messages: {response.get('error', 'Unknown error')}")
                continue
            
            messages = response["messages"]
            
            # Find the latest message from the bot
            latest_bot_message = None
            for message in messages:
                if message.get("user") == bot_user_id:
                    latest_bot_message = message
                    break
            
            if not latest_bot_message:
                print(f"No bot messages found in {channel_name}")
                continue
            
            # Delete the message
            message_ts = latest_bot_message["ts"]
            message_text = latest_bot_message.get("text", "")[:100] + "..." if len(latest_bot_message.get("text", "")) > 100 else latest_bot_message.get("text", "")
            
            print(f"Found bot message: '{message_text}'")
            print(f"Message timestamp: {message_ts}")
            
            # Confirm deletion
            confirm = input(f"Delete this message from {channel_name}? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Skipped.")
                continue
            
            delete_response = client.chat_delete(
                channel=channel_id,
                ts=message_ts
            )
            
            if delete_response["ok"]:
                print(f"✅ Successfully deleted message from {channel_name}")
                success_count += 1
            else:
                print(f"❌ Failed to delete message from {channel_name}: {delete_response.get('error', 'Unknown error')}")
                
        except SlackApiError as e:
            print(f"❌ Error processing {channel_name}: {e.response['error']}")
            continue
    
    print(f"\n🎉 Successfully deleted {success_count} out of {len(channels)} messages")
    return success_count == len(channels)

if __name__ == "__main__":
    print("🤖 Slack Bot Message Cleanup Script")
    print("=" * 40)
    
    # Warning
    print("⚠️  WARNING: This will delete the latest bot message from:")
    print("   - Marketing Channel")
    print("   - TM Moonshot Channel") 
    print("   - TM API Channel")
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