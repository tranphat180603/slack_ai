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
    
    def __init__(self, slack_token: str = None, enable_images: bool = True):
        """
        Initialize the Slack reporter.
        
        Args:
            slack_token: Slack API token (defaults to SLACK_BOT_TOKEN env var)
            enable_images: Whether to generate and upload insight images (defaults to True)
        """
        self.slack_token = slack_token or os.environ.get("SLACK_BOT_TOKEN")
        if not self.slack_token:
            raise ValueError("Slack API token is required")
        
        self.slack_client = WebClient(token=self.slack_token)
        self.enable_images = enable_images
        logger.info(f"SlackReporter initialized (images: {'enabled' if enable_images else 'disabled'})")
        
        # Channel mappings - now maps channel IDs to lists of dashboards
        self.channel_dashboards = {
            "C07D7F5531N": {  # marketing channel
                "channel_name": "Marketing",
                "dashboards": ["Marketing Dashboard"]
            },
            "C07C44USZKR": {  # product channel
                "channel_name": "Product",
                "dashboards": ["Product Dashboard", "Usage Analytics Dashboard", "Trading Dashboard", "Alerts Dashboard"]
            },
            "C07F3SD76EA": {  # tm-api channel
                "channel_name": "TM API",
                "dashboards": ["API Dashboard", "API Cohort Analysis Dashboard"]
            },
            "C092DANQ5RT": {  # tm-moonshot channel
                "channel_name": "TM Moonshot",
                "dashboards": ["Moonshot Analytics"]
            },
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
            
            # Generate report (images are not typically used for daily reports, but respect the setting)
            report = posthog_client.generate_daily_report(dashboard_name)
            
            # Determine if report should be sent (only send if there are significant changes)
            if report.startswith("No significant changes"):
                logger.info(f"No significant changes for {dashboard_name}, skipping alert")
                return True
            
            # Find the channel that contains this dashboard
            channel_id = None
            for ch_id, ch_config in self.channel_dashboards.items():
                if dashboard_name in ch_config["dashboards"]:
                    channel_id = ch_id
                    break
            
            if not channel_id:
                logger.warning(f"No channel mapping found for {dashboard_name}")
                return False
            
            # Send the report
            await self.send_message(channel_id, report)
            logger.info(f"Daily report for {dashboard_name} sent to {channel_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending daily report for {dashboard_name}: {str(e)}")
            return False
    
    async def send_weekly_report(self, channel_ids: Optional[List[str]] = None) -> bool:
        """
        Generate and send comprehensive weekly reports for channels.
        
        Args:
            channel_ids: List of specific channel IDs to send reports to. If None, sends to all channels.
            
        Returns:
            True if all reports were sent successfully, False otherwise
        """
        try:
            # Initialize PosthogClient
            posthog_client = PosthogClient()
            
            # Use all channels if none specified
            if channel_ids is None:
                channel_ids = list(self.channel_dashboards.keys())
            
            success_count = 0
            
            # Process each channel
            for channel_id in channel_ids:
                if channel_id not in self.channel_dashboards:
                    logger.warning(f"Channel ID {channel_id} not found in configuration, skipping")
                    continue
                
                channel_config = self.channel_dashboards[channel_id]
                channel_name = channel_config["channel_name"]
                dashboard_names = channel_config["dashboards"]
                
                print(f"Generating weekly report for {channel_name} channel with dashboards: {dashboard_names}")
                
                # Generate combined report for all dashboards in this channel
                # Only pass slack_channel_id if images are enabled
                report = posthog_client.generate_weekly_report(
                    dashboard_names, 
                    slack_channel_id=channel_id if self.enable_images else None
                )

                report_with_title = f"*Weekly Report for {channel_name} Team*\n\n" + report
                
                # Send the report to the channel
                await self.send_message(channel_id, report_with_title + "\n\n") #test channel: C08C6ENV0G0
                print(f"Weekly report for {channel_name} channel sent to {channel_id}")
                logger.info(f"Weekly report for {channel_name} channel sent to {channel_id}")
                success_count += 1

            return success_count == len(channel_ids)
            
        except Exception as e:
            logger.error(f"Error sending weekly report: {str(e)}")
            return False
        
if __name__ == "__main__":
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Send weekly Posthog reports to Slack')
    parser.add_argument('--no-images', action='store_true', 
                        help='Disable image generation and upload (faster for testing)')
    
    args = parser.parse_args()
    
    # Create an async function to run
    async def main():
        # Check if images should be disabled via environment variable or command line
        enable_images = not args.no_images and os.getenv("DISABLE_POSTHOG_IMAGES", "").lower() != "true"
        
        reporter = SlackReporter(
            slack_token=os.getenv("SLACK_BOT_TOKEN"),
            enable_images=enable_images
        )
        
        print(f"🚀 Starting weekly report generation...")
        print(f"📸 Images: {'enabled' if enable_images else 'disabled'}")
        print("=" * 50)
        
        # Send to all channels
        success = await reporter.send_weekly_report()
        
        print("=" * 50)
        print(f"✅ Report generation completed: {'Success' if success else 'Failed'}")
    
    # Run the async function
    asyncio.run(main())

