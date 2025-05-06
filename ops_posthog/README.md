# Posthog Integration for TMAI Agent

This module provides integration between Posthog analytics and the TMAI Agent, allowing automated dashboard reporting and alerts to be sent to specific Slack channels.

## Features

- **AI-Powered Analytics**: Leverages OpenAI to analyze dashboard data and provide meaningful insights
- **Daily Alerts**: Automatically analyzes dashboard charts for significant changes and sends alerts to Slack
- **Weekly Reports**: Generates comprehensive weekly reports of all dashboard data and trends
- **Cross-Dashboard Analysis**: For weekly reports, analyzes correlations between metrics in different dashboards
- **Slack Integration**: Sends reports to appropriate channels based on dashboard type

## Posthog Data Exploration

You can explore your Posthog data using the built-in command-line interface in `posthog_client.py`:

```bash
# List all dashboards
python posthog_client.py --action list_dashboards

# Examine a specific dashboard's structure and charts
python posthog_client.py --action dashboard_details --dashboard "Marketing Dashboard" --pretty

# Look at data from a specific insight/chart with different time ranges
python posthog_client.py --action insight_data --insight YOUR_INSIGHT_ID --days 7 --pretty
python posthog_client.py --action insight_data --insight YOUR_INSIGHT_ID --days 30 --pretty

# Generate AI-powered daily insights
python posthog_client.py --action ai_daily_insights --dashboard "Marketing Dashboard" --days 14

# Generate AI-powered weekly insights for multiple dashboards
python posthog_client.py --action ai_weekly_insights --dashboard "Marketing Dashboard" --days 28

# Traditional threshold-based reports (for comparison)
python posthog_client.py --action daily_report --dashboard "Marketing Dashboard" --days 14
python posthog_client.py --action weekly_report --dashboard "Marketing Dashboard" --days 28
```

### AI-Powered Analytics

The integration now uses OpenAI's models to analyze Posthog data and provide more meaningful, actionable insights:

- **Daily Insights**: Focused analysis of recent performance with recommendations
- **Weekly Insights**: Comprehensive analysis with executive summary, trends, user behavior insights, and strategic recommendations
- **Cross-Dashboard Synthesis**: Identifies correlations between metrics in different dashboards (for weekly reports with multiple dashboards)

Example AI-generated insights include:
- Identification of anomalies and unusual patterns in the data
- Potential causality between different metrics
- User behavior insights based on engagement patterns
- Strategic recommendations based on the observed data
- Early warning signs of potential issues

### Supported Chart Types

The integration supports various Posthog chart types:

- Time series charts (trend charts)
- Funnel visualizations
- Table data 
- Breakdown charts
- Growth accounting charts
- Retention charts

## Setup

### Environment Variables

The following environment variables should be added to your `.env` file:

```
# Posthog API credentials
POSTHOG_API_KEY=your_posthog_api_key
POSTHOG_PROJECT_ID=your_posthog_project_id
POSTHOG_BASE_URL=https://us.posthog.com/api  # Or your instance URL

# OpenAI API key for AI-powered analytics
OPENAI_API_KEY=your_openai_api_key

# Dashboard names (optional - defaults will be used if not specified)
MARKETING_DASHBOARD_NAME=Marketing Dashboard
PRODUCT_DASHBOARD_NAME=Product Dashboard

# Slack channel IDs/names
MARKETING_CHANNEL_ID=marketing  # Channel name or ID
PRODUCT_CHANNEL_ID=product      # Channel name or ID

# Slack API token (uses existing SLACK_BOT_TOKEN from main app)
# SLACK_BOT_TOKEN should already be set in your main .env file
```

### Integration

To add the Posthog integration to your TMAI Agent:

1. Add the required environment variables to your `.env` file
2. Start the scheduler on application startup:

```python
from ops_posthog import run_scheduler

# Start the scheduler when your application initializes
run_scheduler()
```

## Usage

### Starting/Stopping the Scheduler

```python
from ops_posthog import run_scheduler, stop_scheduler

# Start the scheduler
run_scheduler()

# Stop the scheduler
stop_scheduler()
```

### Manual Report Generation

You can also trigger reports manually:

```python
import asyncio
from ops_posthog import PosthogClient, SlackReporter

async def generate_reports():
    # Create client instances
    posthog = PosthogClient()
    slack = SlackReporter()
    
    # Generate and send AI-powered daily insights for Marketing dashboard
    insights = posthog.generate_ai_insights("Marketing Dashboard", days=7, insight_type="daily")
    await slack.send_message("marketing", insights)
    
    # Generate and send AI-powered weekly insights for all dashboards
    dashboards = ["Marketing Dashboard", "Product Dashboard"]
    weekly_insights = posthog.generate_weekly_report(dashboards)
    await slack.send_message("marketing", weekly_insights)
    await slack.send_message("product", weekly_insights)

# Run the async function
asyncio.run(generate_reports())
```

## Customization

### AI Model Selection

By default, the integration uses `gpt-4o-mini` for generating insights. You can modify this in the `generate_ai_insights` method to use a different model.

### Report Scheduling

The default schedule is:
- Daily alerts: Every day at 9:00 AM
- Weekly reports: Every Friday at 4:00 PM

You can modify these in the `PosthogScheduler.schedule_tasks()` method.

## Extending

### Adding New Dashboards

To add support for a new dashboard:

1. Add a new environment variable for the dashboard name
2. Add a new entry to the `self.dashboards` dictionary in `PosthogScheduler.__init__()`
3. Add a new channel mapping in `SlackReporter.__init__()`

## Troubleshooting

### Data Access Issues

If you encounter issues accessing Posthog data:

1. Verify your API key and project ID are correct
2. Check that your user account has access to the dashboards you're trying to analyze
3. Use the command-line debugging tools to examine the raw data structure:
   ```
   python posthog_client.py --action insight_data --insight YOUR_INSIGHT_ID --pretty
   ```

### API Endpoint Errors

The client tries multiple API endpoints to accommodate different types of Posthog insights. If you encounter HTTP errors:

1. Verify your Base URL is correct (e.g., `https://us.posthog.com/api` for cloud or your custom URL for self-hosted)
2. Check that your API key has sufficient permissions
3. Look at the debug information in the analysis results to understand the data structure

### AI-Generated Insights Issues

If you encounter issues with AI-generated insights:

1. Verify your OpenAI API key is valid and has sufficient quota
2. Check the structure of your data to make sure it's being properly formatted for the AI
3. Try adjusting the prompt templates in the `_get_daily_analysis_prompt` and `_get_weekly_analysis_prompt` methods 