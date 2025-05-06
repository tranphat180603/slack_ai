import os
import logging
from typing import Dict, List, Optional
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import dotenv
import asyncio

dotenv.load_dotenv()

from posthog_client import PosthogClient

logger = logging.getLogger("slack_reporter")

class SlackReporter:
    """Class for sending Posthog reports to Slack channels."""
    
    def __init__(self, slack_token: str = None):
        """
        Initialize the Slack reporter.
        
        Args:
            slack_token: Slack API token (defaults to SLACK_BOT_TOKEN env var)
        """
        self.slack_token = slack_token or os.environ.get("SLACK_BOT_TOKEN")
        if not self.slack_token:
            raise ValueError("Slack API token is required")
        
        self.slack_client = WebClient(token=self.slack_token)
        logger.info("SlackReporter initialized")
        
        # Channel mappings
        self.channel_map = {
            "Marketing": os.environ.get("MARKETING_CHANNEL_ID", "marketing"),
            "Product": os.environ.get("PRODUCT_CHANNEL_ID", "product")
        }
    
    def get_channel_id(self, dashboard_name: str) -> str:
        """
        Get the appropriate Slack channel ID for a given dashboard.
        
        Args:
            dashboard_name: Name of the dashboard
            
        Returns:
            Slack channel ID or name
        """
        # Remove "Dashboard" suffix if present for mapping lookup
        clean_name = dashboard_name.replace(" Dashboard", "")
        
        # Return the mapped channel or default to the dashboard name as channel
        return self.channel_map.get(clean_name, dashboard_name.lower())
    
    async def send_message(self, channel_id: str, text: str, thread_ts: Optional[str] = None) -> Dict:
        """
        Send a message to a Slack channel.
        
        Args:
            channel_id: Channel ID or name
            text: Message text
            thread_ts: Optional thread timestamp to reply to
            
        Returns:
            Slack API response
        """
        try:
            response = self.slack_client.chat_postMessage(
                channel=channel_id,
                text=text,
                thread_ts=thread_ts,
                unfurl_links=False
            )
            logger.info(f"Message sent to channel {channel_id}")
            return response
        except SlackApiError as e:
            logger.error(f"Error sending message to Slack: {e.response['error']}")
            raise
    
    async def send_daily_report(self, dashboard_name: str) -> bool:
        """
        Generate and send a daily report for a dashboard to the appropriate Slack channel.
        
        Args:
            dashboard_name: Name of the dashboard
            
        Returns:
            True if report was sent successfully, False otherwise
        """
        try:
            # Initialize PosthogClient
            posthog_client = PosthogClient()
            
            # Generate report
            report = posthog_client.generate_daily_report(dashboard_name)
            
            # Determine if report should be sent (only send if there are significant changes)
            if report.startswith("No significant changes"):
                logger.info(f"No significant changes for {dashboard_name}, skipping alert")
                return True
            
            # Get the appropriate channel
            channel_id = self.get_channel_id(dashboard_name)
            
            # Send the report
            await self.send_message(channel_id, report)
            logger.info(f"Daily report for {dashboard_name} sent to {channel_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending daily report for {dashboard_name}: {str(e)}")
            return False
    
    async def send_weekly_report(self, dashboard_names: List[str]) -> bool:
        """
        Generate and send a comprehensive weekly report for multiple dashboards.
        
        Args:
            dashboard_names: List of dashboard names to include
            
        Returns:
            True if report was sent successfully, False otherwise
        """
        try:
            # Initialize PosthogClient
            posthog_client = PosthogClient()
            
            # Generate report
            
            for dashboard_name in dashboard_names:
                channel_id = self.get_channel_id(dashboard_name)
                # Generate report
                report = posthog_client.generate_weekly_report(dashboard_names)
                
                # Only send once per channel
                await self.send_message(channel_id, report)
                logger.info(f"Weekly report sent to {channel_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending weekly report: {str(e)}")
            return False 
        
if __name__ == "__main__":
    # Create an async function to run
    async def main():
        reporter = SlackReporter(os.getenv("SLACK_BOT_TOKEN"))
        await reporter.send_weekly_report(["Product Dashboard", "Marketing Dashboard"])
    
    # Run the async function
    asyncio.run(main())