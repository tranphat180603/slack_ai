import os
import asyncio
import logging
from datetime import datetime, time
import schedule
from typing import List, Dict, Any, Optional

from .posthog_client import PosthogClient
from .slack_reporter import SlackReporter

logger = logging.getLogger("posthog_scheduler")

class PosthogScheduler:
    """Scheduler for Posthog report generation and delivery."""
    
    def __init__(self):
        """Initialize the Posthog scheduler."""
        self.posthog_client = None
        self.slack_reporter = None
        self.running = False
        self.dashboards = {
            "Marketing": os.environ.get("MARKETING_DASHBOARD_NAME", "Marketing Dashboard"),
            "Product": os.environ.get("PRODUCT_DASHBOARD_NAME", "Product Dashboard")
        }
        
        logger.info("PosthogScheduler initialized")
    
    def initialize_clients(self):
        """Initialize the Posthog and Slack clients if needed."""
        if not self.posthog_client:
            try:
                self.posthog_client = PosthogClient()
                logger.info("PosthogClient initialized")
            except Exception as e:
                logger.error(f"Error initializing PosthogClient: {str(e)}")
                raise
        
        if not self.slack_reporter:
            try:
                self.slack_reporter = SlackReporter()
                logger.info("SlackReporter initialized")
            except Exception as e:
                logger.error(f"Error initializing SlackReporter: {str(e)}")
                raise
    
    async def send_daily_alert(self, dashboard_name: str):
        """
        Generate and send a daily alert for a specific dashboard.
        
        Args:
            dashboard_name: Name of the dashboard
        """
        try:
            logger.info(f"Generating daily alert for {dashboard_name}")
            self.initialize_clients()
            
            success = await self.slack_reporter.send_daily_report(dashboard_name)
            if success:
                logger.info(f"Daily alert for {dashboard_name} completed successfully")
            else:
                logger.error(f"Failed to send daily alert for {dashboard_name}")
                
        except Exception as e:
            logger.error(f"Error in daily alert for {dashboard_name}: {str(e)}")
    
    async def send_weekly_report(self):
        """Generate and send a weekly report for all dashboards."""
        try:
            logger.info("Generating weekly report")
            self.initialize_clients()
            
            dashboard_names = list(self.dashboards.values())
            success = await self.slack_reporter.send_weekly_report(dashboard_names)
            
            if success:
                logger.info("Weekly report completed successfully")
            else:
                logger.error("Failed to send weekly report")
                
        except Exception as e:
            logger.error(f"Error in weekly report: {str(e)}")
    
    async def run_daily_alerts(self):
        """Run all daily alerts."""
        for dashboard_name in self.dashboards.values():
            await self.send_daily_alert(dashboard_name)
    
    def schedule_tasks(self):
        """Schedule the daily and weekly tasks."""
        # Schedule daily alerts at 9:00 AM
        schedule.every().day.at("09:00").do(
            lambda: asyncio.run(self.run_daily_alerts())
        )
        
        # Schedule weekly report on Friday at 4:00 PM
        schedule.every().friday.at("16:00").do(
            lambda: asyncio.run(self.send_weekly_report())
        )
        
        logger.info("Scheduled daily alerts (9:00 AM) and weekly report (Friday 4:00 PM)")
    
    def start(self):
        """Start the scheduler."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.schedule_tasks()
        
        logger.info("Starting scheduler")
        while self.running:
            schedule.run_pending()
            # Sleep for 1 minute before checking again
            import time
            time.sleep(60)
    
    def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping scheduler")
        self.running = False

# Singleton instance
scheduler = PosthogScheduler()

def run_scheduler():
    """Run the Posthog scheduler."""
    scheduler.start()

def stop_scheduler():
    """Stop the Posthog scheduler."""
    scheduler.stop()

# Command-line entry point
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    run_scheduler() 