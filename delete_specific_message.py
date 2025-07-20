#!/usr/bin/env python3
"""
Script to delete a specific message containing "NEW APPLICATION!!!" text.
"""

import os
import sys
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def delete_specific_message():
    """Delete the specific message containing 'NEW APPLICATION!!!' from all channels."""
    
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
    
    # Channel IDs to search
    channels = {
        "Grant Program Channel": "C093RQ9CRNV"
    }
    
    # The specific text we're looking for
    target_text = "NEW APPLICATION!!!"
    
    success_count = 0
    found_messages = []
    
    for channel_name, channel_id in channels.items():
        print(f"\nSearching in {channel_name} ({channel_id})...")
        
        try:
            # Get recent messages from the channel (search more messages)
            response = client.conversations_history(
                channel=channel_id,
                limit=100  # Search last 100 messages
            )
            
            if not response["ok"]:
                print(f"Error getting messages: {response.get('error', 'Unknown error')}")
                continue
            
            messages = response["messages"]
            
            # Find messages containing the target text
            matching_messages = []
            for message in messages:
                message_text = message.get("text", "")
                if target_text in message_text:
                    # Check if it's from the bot or any user
                    matching_messages.append(message)
            
            if not matching_messages:
                print(f"No messages containing '{target_text}' found in {channel_name}")
                continue
            
            print(f"Found {len(matching_messages)} message(s) containing '{target_text}' in {channel_name}")
            
            # Process each matching message
            for i, message in enumerate(matching_messages):
                message_ts = message["ts"]
                message_text = message.get("text", "")
                user_id = message.get("user", "Unknown")
                
                # Show preview of the message
                preview = message_text[:200] + "..." if len(message_text) > 200 else message_text
                print(f"\nMessage {i+1}:")
                print(f"User: {user_id}")
                print(f"Timestamp: {message_ts}")
                print(f"Preview: {preview}")
                
                # Confirm deletion
                confirm = input(f"Delete this message from {channel_name}? (y/N): ").strip().lower()
                if confirm != 'y':
                    print("Skipped.")
                    continue
                
                try:
                    delete_response = client.chat_delete(
                        channel=channel_id,
                        ts=message_ts
                    )
                    
                    if delete_response["ok"]:
                        print(f"✅ Successfully deleted message from {channel_name}")
                        success_count += 1
                        found_messages.append({
                            'channel': channel_name,
                            'timestamp': message_ts,
                            'preview': preview[:50] + "..."
                        })
                    else:
                        print(f"❌ Failed to delete message: {delete_response.get('error', 'Unknown error')}")
                        
                except SlackApiError as e:
                    print(f"❌ Error deleting message: {e.response['error']}")
                    continue
                
        except SlackApiError as e:
            print(f"❌ Error processing {channel_name}: {e.response['error']}")
            continue
    
    print(f"\n🎉 Successfully deleted {success_count} message(s)")
    if found_messages:
        print("\nDeleted messages:")
        for msg in found_messages:
            print(f"  - {msg['channel']}: {msg['preview']}")
    
    return success_count > 0

if __name__ == "__main__":
    print("🎯 Specific Message Deletion Script")
    print("=" * 50)
    
    # Warning
    print("⚠️  This will search for and delete messages containing:")
    print("   'NEW APPLICATION!!!' from all channels")
    print()
    
    confirm = input("Are you sure you want to proceed? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled.")
        sys.exit(0)
    
    success = delete_specific_message()
    
    if success:
        print("\n✅ Target message(s) found and deleted!")
    else:
        print("\n⚠️  No matching messages found or deleted.") 