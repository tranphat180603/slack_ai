#!/usr/bin/env python3
"""
Demo script for AI-powered Posthog analytics.
This simulates how the AI analytics would work with example Posthog data.
"""

import os
import sys
import json
from datetime import datetime, timedelta
import dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import OpenAI client
from llm.openai_client import OpenaiClient

# Load environment variables
dotenv.load_dotenv()

def generate_sample_dashboard_data(dashboard_name="Marketing Dashboard"):
    """Generate sample dashboard data for demonstration purposes."""
    
    # Create a realistic sample of Posthog dashboard data
    dashboard_data = {
        "dashboard_name": dashboard_name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "period": "14 days",
        "metrics": []
    }
    
    # Add realistic marketing metrics
    if dashboard_name == "Marketing Dashboard":
        dashboard_data["metrics"] = [
            {
                "name": "Website Unique Users",
                "description": "Number of unique visitors to the website",
                "current_value": 17169,
                "previous_value": 18869,
                "change_percent": -9.0,
                "current_period": "Apr 28 - May 4",
                "previous_period": "Apr 21 - Apr 27"
            },
            {
                "name": "Signup Conversion Rate",
                "description": "Percentage of visitors who sign up",
                "current_value": 3.2,
                "previous_value": 2.8,
                "change_percent": 14.3,
                "current_period": "Apr 28 - May 4",
                "previous_period": "Apr 21 - Apr 27"
            },
            {
                "name": "Social Media Referrals",
                "description": "Visits from social media platforms",
                "current_value": 4,
                "previous_value": 21,
                "change_percent": -81.0,
                "current_period": "Apr 28 - May 4",
                "previous_period": "Apr 21 - Apr 27"
            },
            {
                "name": "Email Campaign Clicks",
                "description": "Clicks from email campaigns",
                "current_value": 342,
                "previous_value": 295,
                "change_percent": 15.9,
                "current_period": "Apr 28 - May 4",
                "previous_period": "Apr 21 - Apr 27"
            },
            {
                "name": "Trial Activations",
                "description": "Number of new trial signups",
                "current_value": 87,
                "previous_value": 92,
                "change_percent": -5.4,
                "current_period": "Apr 28 - May 4",
                "previous_period": "Apr 21 - Apr 27"
            }
        ]
    elif dashboard_name == "Product Dashboard":
        dashboard_data["metrics"] = [
            {
                "name": "Daily Active Users",
                "description": "Number of unique users active daily",
                "current_value": 753,
                "previous_value": 812,
                "change_percent": -7.3,
                "current_period": "Apr 28 - May 4",
                "previous_period": "Apr 21 - Apr 27"
            },
            {
                "name": "Feature Usage: Analytics",
                "description": "Number of users engaging with the analytics feature",
                "current_value": 342,
                "previous_value": 287,
                "change_percent": 19.2,
                "current_period": "Apr 28 - May 4",
                "previous_period": "Apr 21 - Apr 27"
            },
            {
                "name": "Feature Usage: AI Agent",
                "description": "Number of users engaging with the AI agent feature",
                "current_value": 521,
                "previous_value": 324,
                "change_percent": 60.8,
                "current_period": "Apr 28 - May 4",
                "previous_period": "Apr 21 - Apr 27"
            },
            {
                "name": "Session Duration",
                "description": "Average session duration in minutes",
                "current_value": 12.7,
                "previous_value": 9.8,
                "change_percent": 29.6,
                "current_period": "Apr 28 - May 4",
                "previous_period": "Apr 21 - Apr 27"
            },
            {
                "name": "Retention Rate (7-day)",
                "description": "Percentage of users returning within 7 days",
                "current_value": 42.1,
                "previous_value": 38.5,
                "change_percent": 9.4,
                "current_period": "Apr 28 - May 4",
                "previous_period": "Apr 21 - Apr 27"
            }
        ]
        
    return dashboard_data

def format_metrics_for_prompt(metrics):
    """Format metrics into a readable text format for the prompt."""
    metrics_text = ""
    
    for i, metric in enumerate(metrics, 1):
        change_direction = "increase" if metric["change_percent"] > 0 else "decrease"
        
        metrics_text += f"{i}. {metric['name']}\n"
        metrics_text += f"   Description: {metric['description']}\n"
        metrics_text += f"   Current value: {metric['current_value']}\n"
        metrics_text += f"   Previous value: {metric['previous_value']}\n"
        metrics_text += f"   Change: {abs(metric['change_percent']):.1f}% {change_direction}\n"
        if metric.get("current_period") and metric.get("previous_period"):
            metrics_text += f"   Periods compared: {metric['current_period']} vs {metric['previous_period']}\n"
        metrics_text += "\n"
        
    return metrics_text

def get_daily_analysis_prompt(dashboard_name, data):
    """Generate prompt for daily analysis."""
    metrics_text = format_metrics_for_prompt(data["metrics"])
    
    prompt = f"""Analyze the following Posthog analytics data for {dashboard_name} over the past {data['period']}.

Dashboard: {dashboard_name}
Date: {data['date']}
Period: {data['period']}

Metrics:
{metrics_text}

Please provide a daily analysis with the following sections:
1. Summary - Overall health and key changes
2. Significant Changes - Detailed analysis of notable metrics (both positive and negative)
3. Recommendations - 2-3 actionable insights based on the data

Format the analysis as a Slack message. Use emoji indicators for clarity (📈 for increases, 📉 for decreases).
Focus on actionable insights rather than just describing the data.
"""
    return prompt

def get_weekly_analysis_prompt(dashboard_name, data):
    """Generate prompt for weekly analysis."""
    metrics_text = format_metrics_for_prompt(data["metrics"])
    
    prompt = f"""Analyze the following Posthog analytics data for {dashboard_name} over the past {data['period']} for a weekly report.

Dashboard: {dashboard_name}
Date: {data['date']}
Period: {data['period']}

Metrics:
{metrics_text}

Please provide a comprehensive weekly analysis with the following sections:
1. Executive Summary - Key performance indicators and overall trends
2. Detailed Analysis - In-depth look at important metrics, trends, and patterns
3. User Behavior Insights - What the data reveals about how users are interacting with the product
4. Areas of Concern - Metrics that need attention or investigation
5. Growth Opportunities - Areas showing potential for optimization
6. Recommendations - 3-5 data-driven, actionable recommendations

Format the analysis as a Slack message. Use emoji indicators for clarity (📈 for increases, 📉 for decreases).
Focus on extracting valuable insights rather than just describing numbers.
"""
    return prompt

def generate_ai_insights(dashboard_name, insight_type="daily"):
    """Generate AI-powered insights for dashboard data."""
    # Get sample dashboard data
    data = generate_sample_dashboard_data(dashboard_name)
    
    # Prepare the prompt based on insight type
    if insight_type.lower() == "daily":
        prompt = get_daily_analysis_prompt(dashboard_name, data)
    else:
        prompt = get_weekly_analysis_prompt(dashboard_name, data)
    
    # Initialize OpenAI client
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return "Error: OpenAI API key not configured. Please add OPENAI_API_KEY to your .env file."
    
    # Use GPT-4o for best analytics capabilities
    ai_client = OpenaiClient(openai_api_key, model="gpt-4o-mini")
    
    # Generate insights
    system_prompt = """You are an expert data analyst specializing in product analytics and user behavior.
You analyze Posthog data and provide clear, concise, and actionable insights.
Focus on identifying trends, anomalies, and opportunities for improvement.
Format your analysis with clear sections, bullet points for key insights, and use emoji indicators (📈 for increases, 📉 for decreases).
Your insights should be data-driven, specific, and include numeric values where relevant.
"""
    insights = ai_client.response(prompt=prompt, system_prompt=system_prompt)
    
    return insights

def generate_cross_dashboard_analysis(dashboards=["Marketing Dashboard", "Product Dashboard"]):
    """Generate a cross-dashboard analysis comparing metrics across different dashboards."""
    # Get insights for each dashboard
    insights = {}
    for dashboard in dashboards:
        insights[dashboard] = generate_ai_insights(dashboard, "weekly")
    
    # Combine insights
    combined_insights = ""
    for dashboard, analysis in insights.items():
        combined_insights += f"\n\n## {dashboard}\n\n{analysis}\n\n"
        combined_insights += "---\n\n"
    
    # Initialize OpenAI client for cross-dashboard analysis
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    ai_client = OpenaiClient(openai_api_key, model="gpt-4o-mini")
    
    # Generate cross-dashboard insights
    system_prompt = """You are a senior data analyst specializing in cross-functional business analytics.
You synthesize insights across multiple dashboards to identify holistic patterns and business opportunities."""
    
    cross_dashboard_prompt = f"""Based on the individual dashboard analyses below, provide a cross-dashboard synthesis that identifies:
1. Overarching trends across all dashboards
2. Correlations between metrics in different dashboards
3. Strategic recommendations based on the full picture

Individual dashboard analyses:
{combined_insights}

Present your analysis in a clear, concise format suitable for executives and stakeholders.
"""
    cross_analysis = ai_client.response(prompt=cross_dashboard_prompt, system_prompt=system_prompt)
    
    # Add the cross-analysis to the report
    final_report = f"# Weekly Cross-Dashboard Analysis\n\n{cross_analysis}\n\n---\n\n{combined_insights}"
    return final_report

def main():
    """Main function to demonstrate AI-powered analytics."""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Demo AI-powered Posthog analytics')
    parser.add_argument('--dashboard', choices=['marketing', 'product', 'cross'], 
                       default='marketing', help='Dashboard to analyze')
    parser.add_argument('--type', choices=['daily', 'weekly'], 
                       default='daily', help='Type of analysis')
    
    args = parser.parse_args()
    
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in your .env file or export it directly.")
        return
    
    print(f"\nGenerating AI-powered {args.type} insights...")
    
    if args.dashboard == 'cross':
        insights = generate_cross_dashboard_analysis()
        print("\nCross-Dashboard Analysis:")
    else:
        dashboard_name = "Marketing Dashboard" if args.dashboard == 'marketing' else "Product Dashboard"
        insights = generate_ai_insights(dashboard_name, args.type)
        print(f"\n{args.type.capitalize()} Insights for {dashboard_name}:")
    
    print("=" * 80)
    print(insights)
    print("=" * 80)

if __name__ == "__main__":
    main() 