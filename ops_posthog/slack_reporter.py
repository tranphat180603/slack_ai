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
            "Marketing Dashboard": os.environ.get("MARKETING_CHANNEL_ID", "C07D7F5531N"),
            "Product Dashboard": os.environ.get("PRODUCT_CHANNEL_ID", "C07C44USZKR"),
            "Data API Dashboard": os.environ.get("DATA_API_CHANNEL_ID", "C07F3SD76EA")
        }
    
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
                unfurl_links=False,
                unfurl_media=False,
                mrkdwn=True
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
            
            # Get the appropriate channel from the channel map
            channel_id = self.channel_map.get(dashboard_name)
            if not channel_id:
                logger.warning(f"No channel mapping found for {dashboard_name}, using fallback")
                # Fall back to dashboard name as channel name
                channel_id = dashboard_name.lower().replace(" dashboard", "")
            
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
            
            # Process each dashboard and send to appropriate channel
            for dashboard_name in dashboard_names:
                # Get channel ID from mapping
                channel_id = self.channel_map.get(dashboard_name)
                
                if not channel_id:
                    logger.warning(f"No channel mapping found for {dashboard_name}, skipping report")
                    continue
                
                # Generate report for this dashboard
                report = posthog_client.generate_weekly_report([dashboard_name], slack_channel_id=channel_id)
                
                logger.info(f"Sending weekly report for {dashboard_name} to channel {channel_id}")
                
                # Send the report to the appropriate channel
                await self.send_message(channel_id, report)
                logger.info(f"Weekly report for {dashboard_name} sent to {channel_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending weekly report: {str(e)}")
            return False
        
if __name__ == "__main__":
    # Create an async function to run
    async def main():
        reporter = SlackReporter(os.getenv("SLACK_BOT_TOKEN"))
        await reporter.send_weekly_report(["Product Dashboard", "Marketing Dashboard", "Data API Dashboard"])
    
    # Run the async function
    asyncio.run(main())