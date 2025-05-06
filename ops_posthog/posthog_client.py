import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import dotenv
import argparse
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import OpenAI client for AI-powered analysis
from llm.openai_client import OpenaiClient

dotenv.load_dotenv()

logger = logging.getLogger("posthog_client")

class PosthogClient:
    """Client for interacting with the Posthog API."""
    
    def __init__(self, api_key: str = None, project_id: str = None, base_url: str = None):
        """
        Initialize the Posthog client.
        
        Args:
            api_key: Posthog API key (defaults to POSTHOG_API_KEY env var)
            project_id: Posthog project ID (defaults to POSTHOG_PROJECT_ID env var)
            base_url: Posthog API base URL (defaults to POSTHOG_BASE_URL env var or https://us.posthog.com/api)
        """
        self.api_key = api_key or os.environ.get("POSTHOG_API_KEY")
        if not self.api_key:
            raise ValueError("Posthog API key is required")
            
        self.project_id = project_id or os.environ.get("POSTHOG_PROJECT_ID")
        if not self.project_id:
            raise ValueError("Posthog project ID is required")
            
        self.base_url = base_url or os.environ.get("POSTHOG_BASE_URL", "https://us.posthog.com/api")
        
        # Remove trailing slash if present
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]
            
        # Create session with authentication headers
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        logger.info(f"PosthogClient initialized for project {self.project_id}")
        
    def _get(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a GET request to the Posthog API."""
        url = f"{self.base_url}/{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def _post(self, endpoint: str, data: Dict) -> Dict:
        """Make a POST request to the Posthog API."""
        url = f"{self.base_url}/{endpoint}"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
        
    def get_dashboards(self) -> List[Dict]:
        """
        Get all dashboards in the project.
        
        Returns:
            List of dashboard objects
        """
        endpoint = f"projects/{self.project_id}/dashboards"
        response = self._get(endpoint)
        return response.get("results", [])
    
    def get_dashboard_items(self, dashboard_id: str) -> List[Dict]:
        """
        Get all insights/charts in a dashboard.
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            List of dashboard items/tiles
        """
        try:
            # Get the full dashboard data
            dashboard = self.get_dashboard_by_id(dashboard_id)
            
            # Extract the tiles/items from the dashboard
            if "tiles" in dashboard:
                logger.info(f"Found {len(dashboard['tiles'])} tiles in dashboard {dashboard_id}")
                return dashboard.get("tiles", [])
            elif "items" in dashboard:
                logger.info(f"Found {len(dashboard['items'])} items in dashboard {dashboard_id}")
                return dashboard.get("items", [])
            else:
                # Try to find any field that might contain the items
                for key in dashboard.keys():
                    if isinstance(dashboard[key], list) and len(dashboard[key]) > 0:
                        if any(isinstance(item, dict) and ('insight' in item or 'chart' in item) for item in dashboard[key][:5]):
                            logger.info(f"Found items in field '{key}' of dashboard {dashboard_id}")
                            return dashboard[key]
                
                logger.warning(f"No items found in dashboard {dashboard_id}. Available fields: {list(dashboard.keys())}")
                return []
        except Exception as e:
            logger.error(f"Error getting dashboard items: {str(e)}")
            return []
    
    
    def get_dashboard_by_name(self, name: str) -> Optional[Dict]:
        """
        Get a dashboard by name.
        
        Args:
            name: Dashboard name
            
        Returns:
            Dashboard object if found, None otherwise
        """
        dashboards = self.get_dashboards()
        for dashboard in dashboards:
            if dashboard.get("name") == name:
                return dashboard
        return None
    
    def get_dashboard_by_id(self, dashboard_id: str) -> Dict:
        """
        Get a dashboard by ID.
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            Dashboard object
        """
        endpoint = f"projects/{self.project_id}/dashboards/{dashboard_id}"
        return self._get(endpoint)
    
    
    def get_insight_data(self, insight_id: str, days: int = 7) -> Dict:
        """
        Get data for a specific insight/chart.
        
        Args:
            insight_id: Insight ID
            days: Number of days of data to retrieve (default: 7)
            
        Returns:
            Insight data object
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for Posthog API
        date_from = start_date.strftime("%Y-%m-%d")
        date_to = end_date.strftime("%Y-%m-%d")
        
        # Try different API endpoints
        try:
            # First try the direct insight endpoint
            endpoint = f"projects/{self.project_id}/insights/{insight_id}"
            return self._get(endpoint)
        except Exception as e:
            logger.warning(f"Error accessing insight directly: {str(e)}")
            
            # Try alternative endpoints
            try:
                # Try the trends endpoint
                endpoint = f"projects/{self.project_id}/insights/trend"
                params = {
                    "insight_id": insight_id,
                    "date_from": date_from,
                    "date_to": date_to
                }
                return self._get(endpoint, params=params)
            except Exception as e2:
                logger.warning(f"Error accessing trends endpoint: {str(e2)}")
                
                # Last resort - try the funnel endpoint
                try:
                    endpoint = f"projects/{self.project_id}/insights/funnel"
                    params = {
                        "insight_id": insight_id,
                        "date_from": date_from,
                        "date_to": date_to
                    }
                    return self._get(endpoint, params=params)
                except Exception as e3:
                    logger.error(f"All insight endpoints failed: {str(e3)}")
                    return {
                        "error": f"Failed to access insight data: {str(e3)}",
                        "id": insight_id,
                        "fallback_result": []
                    }
            
        return results
    
    def get_dashboard_data(self, dashboard_name: str, days: int = 7) -> Dict:
        """
        Get and analyze data for all insights in a dashboard.
        
        Args:
            dashboard_name: Name of the dashboard
            days: Number of days of data to retrieve
            
        Returns:
            Dashboard data with analysis of each insight
        """
        dashboard = self.get_dashboard_by_name(dashboard_name)
        if not dashboard:
            return {"error": f"Dashboard '{dashboard_name}' not found"}
        
        dashboard_id = dashboard.get("id")
        dashboard_items = self.get_dashboard_items(dashboard_id)
        
        results = {
            "dashboard_id": dashboard_id,
            "dashboard_name": dashboard_name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "period": f"{days} days",
            "insights": []
        }

        for item in dashboard_items:
            insight_id = item.get("insight", {}).get("id")
            if insight_id:
                try:
                    insight_data = self.get_insight_data(insight_id, days)                    
                    # Extract only essential information
                    simplified_insight = {
                        "id": insight_data.get("id"),
                        "name": insight_data.get("name", "Unnamed insight"),
                        "description": insight_data.get("description", ""),
                        "data_points": insight_data.get("result", [])[0].get("data", []),
                        "data_range": insight_data.get("result", [])[0].get("days", []),
                        "data_labels": insight_data.get("result", [])[0].get("labels", []),
                        "series": [series.get("serie_name", "") for series in insight_data.get("result", [])[0].get("series", [])]
                    }
                    
                    results["insights"].append(simplified_insight)
                except Exception as e:
                    logger.error(f"Error analyzing insight {insight_id}: {str(e)}")
                    results["insights"].append({
                        "id": insight_id,
                        "name": item.get("insight", {}).get("name", "Unnamed insight"),
                        "description": "",
                        "error": str(e)
                    })
        
        return results
    
    def generate_ai_insights(self, dashboard_name: str, days: int = 7, insight_type: str = "daily") -> str:
        """
        Generate AI-powered insights about dashboard data.
        
        Args:
            dashboard_name: Name of the dashboard
            days: Number of days to analyze
            insight_type: Type of insight to generate (daily or weekly)
            
        Returns:
            AI-generated insights about the dashboard data
        """
        try:
            # Get dashboard data
            dashboard_data = self.get_dashboard_data(dashboard_name, days)
            
            if "error" in dashboard_data:
                return f"Error generating insights: {dashboard_data['error']}"
            
            # Prepare the prompt based on insight type
            if insight_type.lower() == "daily":
                prompt = self._get_daily_analysis_prompt(dashboard_name, dashboard_data, days)
            elif insight_type.lower() == "weekly":
                prompt = self._get_weekly_analysis_prompt(dashboard_name, dashboard_data, days)
            else:
                prompt = self._get_daily_analysis_prompt(dashboard_name, dashboard_data, days)
            
            # Initialize OpenAI client
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error("OPENAI_API_KEY environment variable not set")
                return "Error: OpenAI API key not configured"
            
            # Use OpenAI for analysis
            ai_client = OpenaiClient(openai_api_key, model="gpt-4.1-mini-2025-04-14")
            
            # Generate insights with a system prompt that guides the AI
            system_prompt = """You are TMAI Agent, a helpful assistant operate within the company called Token Metrics, a company works in the field of crypto and AI.
You analyze Posthog data and provide clear, concise, and actionable insights.
Focus on identifying trends, anomalies, and opportunities for improvement.
Format your analysis with clear sections, bullet points for key insights, and use emoji indicators (📈 for increases, 📉 for decreases).
Your insights should be data-driven, specific, and include numeric values where relevant.
"""
            full_response = ""
            # Get AI response
            print(f"System Prompt: {system_prompt}")
            print(f"User Prompt: {prompt}")
            insights = ai_client.response(prompt=prompt, system_prompt=system_prompt, stream=True)
            for chunk in insights:
                if chunk.type == "response.output_text.delta":
                    print(chunk.delta, end="", flush=True)
                    full_response += chunk.delta
            return full_response
        
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            return f"Error generating AI insights: {str(e)}"
    
    
    def _get_daily_analysis_prompt(self, dashboard_name: str, data: Dict, days: int) -> str:
        """Generate prompt for daily analysis."""        
        prompt = f"""Analyze the following Posthog analytics data for {dashboard_name} over the past {days} days.
 
Dashboard: {dashboard_name}
Date: {data.get("date", datetime.now().strftime("%Y-%m-%d"))}
Period: {data.get("period", f"{days} days")}

Insights:
{data.get("insights")}
"""
        prompt += "\n"
        
        prompt += """
Please provide a daily analysis with the following sections:
1. Summary - Overall health and key changes
2. Significant Changes - Detailed analysis of notable metrics (both positive and negative)
3. Recommendations - 2-3 actionable insights based on the data

Format the analysis as a Slack message. Use emoji indicators for clarity (📈 for increases, 📉 for decreases).
Focus on actionable insights rather than just describing the data.
"""
        return prompt
    
    def _get_weekly_analysis_prompt(self, dashboard_name: str, data: Dict, days: int) -> str:
        """Generate prompt for weekly analysis."""
        
        prompt = f"""Analyze the following Posthog analytics data for {dashboard_name} over the past {days} days.

Dashboard: {dashboard_name}
Date: {data.get("date", datetime.now().strftime("%Y-%m-%d"))}
Period: {data.get("period", f"{days} days")}

Insights:
{data.get("insights")}
"""
        prompt += "\n"
        
        prompt += """
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
    
        
    def generate_daily_report(self, dashboard_name: str) -> str:
        """
        Generate a daily report for a dashboard, using AI to analyze changes.
        
        Args:
            dashboard_name: Name of the dashboard
            
        Returns:
            AI-generated report focusing on meaningful insights
        """
        return self.generate_ai_insights(dashboard_name, days=7, insight_type="daily")
    
    def generate_weekly_report(self, dashboard_names: List[str]) -> str:
        """
        Generate a comprehensive weekly report for multiple dashboards using AI analysis.
        
        Args:
            dashboard_names: List of dashboard names to include
            
        Returns:
            AI-generated comprehensive weekly report
        """
        # For multi-dashboard reports, combine the data first
        combined_insights = ""
        
        for dashboard_name in dashboard_names:
            dashboard_insights = self.generate_ai_insights(dashboard_name, days=28, insight_type="weekly")
            combined_insights += f"\n\n## {dashboard_name}\n\n{dashboard_insights}\n\n"
            combined_insights += "---\n\n"
        
        # If there are multiple dashboards, add a cross-dashboard analysis
        if len(dashboard_names) > 1:
            # Initialize OpenAI client for cross-dashboard analysis
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            ai_client = OpenaiClient(openai_api_key, model="gpt-4.1-mini-2025-04-14")
            
            # Generate cross-dashboard insights
            system_prompt = """You are TMAI Agent, a helpful assistant operate within the company called Token Metrics, a company works in the field of crypto and AI.
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
        
        # If only one dashboard, return the single analysis
        return combined_insights
        
    def create_export_job(self, export_type: str, export_id: str, export_format: str = "image/png") -> Dict:
        """
        Create an export job to generate a screenshot.
        
        Args:
            export_type: Type of export ('insight' or 'dashboard')
            export_id: ID of the insight or dashboard to export
            export_format: Format of the export (default: 'image/png')
            
        Returns:
            Export job details
        """
        endpoint = f"projects/{self.project_id}/exports"
        data = {
            export_type: export_id,
            "export_format": export_format
        }
        
        logger.info(f"Creating export job for {export_type} {export_id}")
        return self._post(endpoint, data)
    
    def get_export_by_id(self, export_id: str) -> Dict:
        endpoint = f"projects/{self.project_id}/exports/{export_id}"
        return self._get(endpoint)

    def get_export_content(self, export_id: str) -> bytes:
        endpoint = f"projects/{self.project_id}/exports/{export_id}/content"
        url = f"{self.base_url}/{endpoint}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.content

    
    def wait_for_export(self, export_id: str, max_retries: int = 10, delay: int = 2) -> Dict:
        """
        Wait for an export job to complete.
        
        Args:
            export_id: Export job ID
            max_retries: Maximum number of retries
            delay: Delay between retries in seconds
            
        Returns:
            Export job details when complete
        """
        import time
        
        logger.info(f"Waiting for export job {export_id} to complete")
        
        for i in range(max_retries):
            export_details = self.get_export_by_id(export_id)
            
            if export_details.get("has_content"):
                logger.info(f"Export job {export_id} completed")
                return export_details
                
            logger.info(f"Export job {export_id} not yet complete, retrying ({i+1}/{max_retries})")
            time.sleep(delay)
            
        logger.warning(f"Export job {export_id} did not complete within {max_retries} retries")
        return {"error": "Export timed out", "export_id": export_id}
    
    def get_dashboard_screenshot(self, dashboard_id: str, save_path: Optional[str] = None) -> Union[bytes, str]:
        """
        Get a screenshot of a dashboard.
        
        Args:
            dashboard_id: Dashboard ID
            save_path: Optional path to save the screenshot
            
        Returns:
            If save_path is provided, returns the path to the saved file.
            Otherwise, returns the binary content of the screenshot.
        """
        try:
            # Create export job
            export_job = self.create_export_job("dashboard", dashboard_id)
            export_id = export_job.get("id")
            
            if not export_id:
                logger.error("Failed to create export job")
                return b"" if save_path is None else ""
                
            # Wait for export to complete
            export_details = self.wait_for_export(export_id)
            
            if "error" in export_details:
                logger.error(f"Error waiting for export: {export_details['error']}")
                return b"" if save_path is None else ""
                
            # Get export content
            content = self.get_export_content(export_id)
            
            # Save to file if path provided
            if save_path:
                with open(save_path, "wb") as f:
                    f.write(content)
                logger.info(f"Dashboard screenshot saved to {save_path}")
                return save_path
                
            return content
            
        except Exception as e:
            logger.error(f"Error getting dashboard screenshot: {str(e)}")
            return b"" if save_path is None else ""
    
    def get_insight_screenshot(self, insight_id: str, save_path: Optional[str] = None) -> Union[bytes, str]:
        """
        Get a screenshot of an insight.
        
        Args:
            insight_id: Insight ID
            save_path: Optional path to save the screenshot
            
        Returns:
            If save_path is provided, returns the path to the saved file.
            Otherwise, returns the binary content of the screenshot.
        """
        try:
            # Create export job
            export_job = self.create_export_job("insight", insight_id)
            export_id = export_job.get("id")
            
            if not export_id:
                logger.error("Failed to create export job")
                return b"" if save_path is None else ""
                
            # Wait for export to complete
            export_details = self.wait_for_export(export_id)
            
            if "error" in export_details:
                logger.error(f"Error waiting for export: {export_details['error']}")
                return b"" if save_path is None else ""
                
            # Get export content
            content = self.get_export_content(export_id)
            
            # Save to file if path provided
            if save_path:
                with open(save_path, "wb") as f:
                    f.write(content)
                logger.info(f"Insight screenshot saved to {save_path}")
                return save_path
                
            return content
            
        except Exception as e:
            logger.error(f"Error getting insight screenshot: {str(e)}")
            return b"" if save_path is None else ""
    
    def save_dashboard_screenshots(self, dashboard_name: str, output_dir: str) -> List[str]:
        """
        Save screenshots of all insights in a dashboard to separate files.
        
        Args:
            dashboard_name: Name of the dashboard
            output_dir: Directory to save screenshots to
            
        Returns:
            List of saved file paths
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get dashboard
        dashboard = self.get_dashboard_by_name(dashboard_name)
        if not dashboard:
            logger.error(f"Dashboard '{dashboard_name}' not found")
            return []
            
        dashboard_id = dashboard.get("id")
        logger.info(f"Found dashboard '{dashboard_name}' with ID {dashboard_id}")
        
        # Save dashboard screenshot
        dashboard_path = os.path.join(output_dir, f"{dashboard_name.replace(' ', '_')}_dashboard.png")
        try:
            logger.info(f"Saving dashboard screenshot to {dashboard_path}")
            result = self.get_dashboard_screenshot(dashboard_id, dashboard_path)
            if not result:
                logger.warning(f"Failed to save dashboard screenshot to {dashboard_path}")
            elif isinstance(result, str):
                logger.info(f"Dashboard screenshot saved to {result}")
            saved_paths = [dashboard_path] if os.path.exists(dashboard_path) else []
        except Exception as e:
            logger.error(f"Error saving dashboard screenshot: {str(e)}")
            saved_paths = []
        
        # Try to get dashboard items
        try:
            # Get dashboard items
            items = self.get_dashboard_items(dashboard_id)
            logger.info(f"Found {len(items)} items in dashboard {dashboard_id}")
            
            # Log the structure of the first item for debugging
            if items and len(items) > 0:
                logger.debug(f"First item structure: {json.dumps(list(items[0].keys()))}")
            
            # Save screenshot for each insight
            for idx, item in enumerate(items):
                try:
                    # Try to extract insight ID from different possible structures
                    insight_id = None
                    insight_name = "unnamed_insight"
                    
                    if "insight" in item and isinstance(item["insight"], dict) and "id" in item["insight"]:
                        insight_id = item["insight"].get("id")
                        insight_name = item["insight"].get("name", f"insight_{idx}")
                    elif "id" in item:
                        insight_id = item.get("id")
                        insight_name = item.get("name", f"insight_{idx}")
                    elif "card" in item and isinstance(item["card"], dict) and "id" in item["card"]:
                        insight_id = item["card"].get("id")
                        insight_name = item["card"].get("name", f"insight_{idx}")
                    
                    if not insight_id:
                        logger.warning(f"Could not determine insight ID for item {idx}. Available keys: {list(item.keys())}")
                        continue
                    
                    # Create safe filename
                    safe_name = insight_name.replace(" ", "_").replace("/", "_").replace(":", "").replace("?", "")
                    insight_path = os.path.join(output_dir, f"{safe_name}_{insight_id}.png")
                    
                    logger.info(f"Saving insight {insight_id} screenshot to {insight_path}")
                    result = self.get_insight_screenshot(insight_id, insight_path)
                    
                    if not result:
                        logger.warning(f"Failed to save insight screenshot to {insight_path}")
                    elif isinstance(result, str):
                        logger.info(f"Insight screenshot saved to {result}")
                        saved_paths.append(insight_path)
                        
                except Exception as e:
                    logger.error(f"Error saving screenshot for item {idx}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing dashboard items: {str(e)}")
        
        logger.info(f"Saved {len(saved_paths)} screenshots to {output_dir}")
        return saved_paths

def main():
    """
    Main function for exploring and testing Posthog data.
    """
    import json
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Explore Posthog data')
    parser.add_argument('--action', choices=['list_dashboards', 'dashboard_details', 'insight_data', 
                                          'daily_report', 'weekly_report', 'ai_daily_insights', 'ai_weekly_insights', 'export_dashboard_screenshots'],
                        default='list_dashboards', help='Action to perform')
    parser.add_argument('--dashboard', type=str, help='Dashboard name')
    parser.add_argument('--insight', type=str, help='Insight ID')
    parser.add_argument('--days', type=int, default=7, help='Number of days of data')
    parser.add_argument('--pretty', action='store_true', help='Pretty print JSON')
    
    args = parser.parse_args()
    
    # Initialize client
    api_key = os.environ.get("POSTHOG_API_KEY")
    project_id = os.environ.get("POSTHOG_PROJECT_ID")
    base_url = os.environ.get("POSTHOG_BASE_URL")
    
    if not api_key or not project_id:
        print("Error: POSTHOG_API_KEY and POSTHOG_PROJECT_ID environment variables must be set")
        print("Consider adding them to your .env file")
        return
    
    posthog_client = PosthogClient(api_key, project_id, base_url)
    print(f"Connected to Posthog project {project_id}")
    
    # Execute requested action
    if args.action == 'list_dashboards':
        dashboards = posthog_client.get_dashboards()
        print(f"\nFound {len(dashboards)} dashboards:")
        for idx, dashboard in enumerate(dashboards, 1):
            print(f"{idx}. {dashboard.get('name')} (ID: {dashboard.get('id')})")
            
    elif args.action == 'dashboard_details':
        if not args.dashboard:
            print("Error: --dashboard is required for dashboard_details action")
            return
            
        dashboard = posthog_client.get_dashboard_by_name(args.dashboard)
        if not dashboard:
            print(f"Error: Dashboard '{args.dashboard}' not found")
            return
            
        dashboard_id = dashboard.get('id')
        items = posthog_client.get_dashboard_items(dashboard_id)
        
        print(f"\nDashboard: {dashboard.get('name')} (ID: {dashboard_id})")
        print(f"Description: {dashboard.get('description', 'No description')}")
        print(f"Found {len(items)} insights/charts:")
        
        for idx, item in enumerate(items, 1):
            insight = item.get('insight', {})
            print(f"\n{idx}. {insight.get('name', 'Unnamed')} (ID: {insight.get('id')})")
            print(f"   Type: {insight.get('type', 'Unknown')}")
            print(f"   Description: {insight.get('description', 'No description')}")
            
            if args.pretty:
                # Print a sample of the item structure
                print("\n   Sample structure:")
                sample = {k: v for k, v in insight.items() if k in ['id', 'name', 'type', 'description', 'filters']}
                print(json.dumps(sample, indent=4)[:500] + "...")
    
    elif args.action == 'insight_data':
        if not args.insight:
            print("Error: --insight is required for insight_data action")
            return
            
        print(f"\nGetting data for insight {args.insight} over {args.days} days:")
        insight_data = posthog_client.get_insight_data(args.insight, args.days)
        
        # Print structure of the insight data
        if args.pretty:
            print("\nData structure:")
            print(json.dumps(insight_data, indent=4))
        else:
            print(f"Result type: {type(insight_data)}")
            print(f"Keys: {list(insight_data.keys())}")
            
        # Analyze the data
        analysis = posthog_client.analyze_insight_data(insight_data)
        print("\nAnalysis results:")
        print(json.dumps(analysis, indent=4))
    
    elif args.action == 'daily_report':
        if not args.dashboard:
            print("Error: --dashboard is required for daily_report action")
            return
            
        print(f"\nGenerating daily report for {args.dashboard} over {args.days} days:")
        dashboard_data = posthog_client.get_dashboard_data(args.dashboard, args.days)
        
        if "error" in dashboard_data:
            print(f"Error: {dashboard_data['error']}")
            return
            
        # Generate the full report
        report = posthog_client.generate_daily_report(args.dashboard)
        print("\nFormatted report for Slack:")
        print("=" * 80)
        print(report)
        print("=" * 80)
    
    elif args.action == 'weekly_report':
        if not args.dashboard:
            print("Using default dashboards: Marketing Dashboard and Product Dashboard")
            dashboards = ["Marketing Dashboard", "Product Dashboard"]
        else:
            dashboards = [args.dashboard]
            
        print(f"\nGenerating weekly report for {', '.join(dashboards)} over {args.days} days:")
        
        # Generate the full report
        report = posthog_client.generate_weekly_report(dashboards)
        print("\nFormatted report for Slack:")
        print("=" * 80)
        print(report)
        print("=" * 80)
    
    elif args.action == 'ai_daily_insights':
        if not args.dashboard:
            print("Error: --dashboard is required for ai_daily_insights action")
            return
            
        print(f"\nGenerating AI-powered daily insights for {args.dashboard} over {args.days} days:")
        insights = posthog_client.generate_ai_insights(args.dashboard, args.days, "daily")
        
        print("\nAI-Generated Insights:")
        print("=" * 80)
        print(insights)
        print("=" * 80)
    
    elif args.action == 'ai_weekly_insights':
        if not args.dashboard:
            print("Using default dashboards: Marketing Dashboard and Product Dashboard")
            dashboards = ["Marketing Dashboard", "Product Dashboard"]
        else:
            dashboards = [args.dashboard]
            
        print(f"\nGenerating AI-powered weekly insights for {', '.join(dashboards)} over {args.days} days:")
        insights = posthog_client.generate_weekly_report(dashboards)
        
        print("\nAI-Generated Insights:")
        print("=" * 80)
        print(insights)
        print("=" * 80)
    elif args.action == 'export_dashboard_screenshots':
        if not args.dashboard:
            print("Error: --dashboard is required for export_dashboard_screenshots action")
            return
            
        print(f"\nExporting screenshots for {args.dashboard} to screenshots/ directory:")
        saved_paths = posthog_client.save_dashboard_screenshots(args.dashboard, "screenshots")
            

if __name__ == "__main__":
    main()