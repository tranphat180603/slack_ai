import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import dotenv
import argparse
import sys
from datetime import timezone

# Import Slack WebClient for file uploads
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import OpenAI client for AI-powered analysis
from llm.openai_client import OpenaiClient

dotenv.load_dotenv()

logger = logging.getLogger("posthog_client")
logger.setLevel(logging.INFO)

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
        results = response.get("results", [])
        dashboards = []
        for dashboard in results:
            if dashboard.get("pinned") == False:
                continue
            dashboard_data = {
                "id": dashboard.get("id"),
                "name": dashboard.get("name"),
                "created_at": dashboard.get("created_at"),
                "pinned": dashboard.get("pinned")
            }
            dashboards.append(dashboard_data)
        return dashboards
    
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
        end_date = datetime.now() - timedelta(days=1) #skip the current day
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
    
    def get_dashboard_data(self, dashboard_name: str, days: int = 7, slack_channel_id: Optional[str] = None, slack_thread_ts: Optional[str] = None) -> Dict:
        """
        Get and analyze data for all insights in a dashboard.
        Also uploads insight screenshots to Slack if slack_channel_id is provided.
        
        Args:
            dashboard_name: Name of the dashboard
            days: Number of days of data to retrieve
            slack_channel_id: Optional Slack channel ID to upload screenshots to.
            slack_thread_ts: Optional Slack thread timestamp for uploads.
            
        Returns:
            Dashboard data with analysis of each insight and Slack permalinks for images.
        """
        screenshots_tmp_dir = "temp_insight_screenshots"
        if not os.path.exists(screenshots_tmp_dir):
            os.makedirs(screenshots_tmp_dir)

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
            "insights": [],
            "insights_images": {}
        }

        for item in dashboard_items:
            insight_id = item.get("insight", {}).get("id")
            if insight_id:
                try:
                    insight_data = self.get_insight_data(insight_id, days)
                    
                    simplified_insight = {
                        "id": insight_data.get("id"),
                        "short_id": insight_data.get("short_id"),
                        "name": insight_data.get("name", insight_data.get("derived_name", "Unnamed insight")),
                        "description": insight_data.get("description", ""),
                        "last_refresh": insight_data.get("last_refresh"),
                        "type": "UNKNOWN"  # Default type
                    }

                    query_data = insight_data.get("query", {})
                    determined_kind = None

                    if query_data:
                        source_data = query_data.get("source", {})
                        determined_kind = source_data.get("kind")
                        if not determined_kind and query_data.get("kind") == 'InsightVizNode':
                            # For InsightVizNode, the actual query kind is often nested in source.source
                            nested_source_data = source_data.get("source", {})
                            if isinstance(nested_source_data, dict): # Ensure nested_source_data is a dict
                                determined_kind = nested_source_data.get("kind")
                        if not determined_kind: # if still not found in query, try top-level kind from query
                            determined_kind = query_data.get("kind")
                    
                    if not determined_kind:
                        insight_kind_from_filters = insight_data.get("filters", {}).get("insight")
                        if insight_kind_from_filters:
                            if insight_kind_from_filters.upper() == 'TRENDS':
                                determined_kind = 'TrendsQuery'
                            elif insight_kind_from_filters.upper() == 'LIFECYCLE':
                                determined_kind = 'LifecycleQuery'
                            elif insight_kind_from_filters.upper() == 'RETENTION':
                                determined_kind = 'RetentionQuery'
                            else:
                                determined_kind = insight_kind_from_filters
                    
                    simplified_insight["type"] = determined_kind if determined_kind else "UNKNOWN"
                    
                    result_list = insight_data.get("result", [])

                    if determined_kind == "TrendsQuery":
                        simplified_insight["series_data"] = []
                        if result_list and isinstance(result_list, list) and len(result_list) > 0:
                            simplified_insight["common_labels"] = result_list[0].get("labels", [])
                            simplified_insight["common_days"] = result_list[0].get("days", [])
                            
                            query_series_configs = query_data.get("source", {}).get("series", [])
                            # If kind was from nested source (e.g. InsightVizNode -> TrendsQuery)
                            if not query_series_configs and query_data.get("source", {}).get("source", {}):
                                 nested_source = query_data.get("source", {}).get("source", {})
                                 if isinstance(nested_source, dict):
                                     query_series_configs = nested_source.get("series", [])


                            for i, series_result_item in enumerate(result_list):
                                action_info = series_result_item.get("action", {})
                                series_config = {}
                                order_index = action_info.get("order")

                                if order_index is not None and order_index < len(query_series_configs):
                                    series_config = query_series_configs[order_index]
                                elif i < len(query_series_configs):
                                    series_config = query_series_configs[i]

                                series_name = action_info.get("custom_name")
                                if not series_name: series_name = series_config.get("custom_name")
                                if not series_name: series_name = series_result_item.get("label", f"Series {i+1}")

                                event_name = action_info.get("name", series_config.get("event", series_config.get("name")))
                                math_op = action_info.get("math", series_config.get("math"))
                                math_prop = action_info.get("math_property", series_config.get("math_property"))

                                series_entry = {
                                    "series_name": series_name,
                                    "event": event_name,
                                    "math_operation": math_op,
                                    "math_property": math_prop,
                                    "data_points": series_result_item.get("data", []),
                                    "labels": series_result_item.get("labels", simplified_insight.get("common_labels", [])),
                                    "days": series_result_item.get("days", simplified_insight.get("common_days", []))
                                }
                                
                                # Include aggregated_value if present (important for bar chart insights)
                                if "aggregated_value" in series_result_item:
                                    series_entry["aggregated_value"] = series_result_item.get("aggregated_value")
                                
                                simplified_insight["series_data"].append(series_entry)
                            if simplified_insight["series_data"]:
                                simplified_insight["series_names"] = [s["series_name"] for s in simplified_insight["series_data"]]

                    elif determined_kind == "LifecycleQuery":
                        simplified_insight["lifecycle_stages"] = []
                        if result_list and isinstance(result_list, list):
                            for stage_item in result_list:
                                action_info = stage_item.get("action", {})
                                simplified_insight["lifecycle_stages"].append({
                                    "status": stage_item.get("status"),
                                    "stage_label": stage_item.get("label"),
                                    "event_name": action_info.get("name"),
                                    "math_operation": action_info.get("math"),
                                    "data_points": stage_item.get("data", []),
                                    "labels": stage_item.get("labels", []),
                                    "days": stage_item.get("days", [])
                                })
                    
                    elif determined_kind == "RetentionQuery":
                        simplified_insight["cohorts"] = []
                        if result_list and isinstance(result_list, list):
                            for cohort_item in result_list:
                                retention_periods_data = []
                                for value_item in cohort_item.get("values", []):
                                    retention_periods_data.append({
                                        "period_label": value_item.get("label"),
                                        "count": value_item.get("count")
                                    })
                                simplified_insight["cohorts"].append({
                                    "cohort_start_date": cohort_item.get("date"),
                                    "cohort_label": cohort_item.get("label"),
                                    "retention_periods": retention_periods_data
                                })
                    else: # Fallback for UNKNOWN or other types
                        if result_list and isinstance(result_list, list) and len(result_list) > 0:
                            # Attempt generic series extraction if it looks like trends
                            if all(isinstance(item, dict) and "data" in item and "labels" in item for item in result_list):
                                simplified_insight["generic_series_data"] = []
                                for i, item_res in enumerate(result_list):
                                    action_info = item_res.get("action", {})
                                    series_name = action_info.get("custom_name", item_res.get("label", f"Series {i+1}"))
                                    simplified_insight["generic_series_data"].append({
                                        "series_name": series_name,
                                        "event": action_info.get("name"),
                                        "math_operation": action_info.get("math"),
                                        "math_property": action_info.get("math_property"),
                                        "data_points": item_res.get("data", []),
                                        "labels": item_res.get("labels", []),
                                        "days": item_res.get("days", [])
                                    })
                                if simplified_insight["generic_series_data"]:
                                    simplified_insight["series_names"] = [s["series_name"] for s in simplified_insight["generic_series_data"]]
                        
                        if not simplified_insight.get("generic_series_data") and not simplified_insight.get("series_data"):
                            simplified_insight["raw_result"] = result_list
                    
                    if slack_channel_id:
                        print(f"Slack channel provided, uploading screenshot to Slack")
                        # Attempt to generate and save screenshot, then upload to Slack
                        if simplified_insight.get("name") and not simplified_insight.get("error"):
                            insight_name_for_file = simplified_insight['name']
                            safe_insight_name = insight_name_for_file.replace(" ", "_").replace("/", "_").replace(":", "").replace("?", "")
                            max_len_safe_name = 100 
                            safe_insight_name = safe_insight_name[:max_len_safe_name]
                            
                            image_filename = f"{insight_id}_{safe_insight_name}.png"
                            local_save_path = os.path.join(screenshots_tmp_dir, image_filename)
                            
                            try:
                                screenshot_result = self.get_insight_screenshot(str(insight_id), save_path=local_save_path)
                                
                                if screenshot_result and isinstance(screenshot_result, str) and os.path.exists(screenshot_result):
                                    print(f"Successfully generated screenshot locally for insight '{simplified_insight['name']}' and saved to {screenshot_result}")
                                    
                                    # Upload to Slack if channel_id and token are available
                                    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
                                    if slack_channel_id and slack_bot_token:
                                        try:
                                            slack_client = WebClient(token=slack_bot_token)
                                            upload_response = slack_client.files_upload_v2(
                                                channel=slack_channel_id,
                                                file=local_save_path,
                                                thread_ts=slack_thread_ts if slack_thread_ts else None,
                                                initial_comment=f"Screenshot for insight: {simplified_insight['name']}"
                                            )
                                            if upload_response.get("ok") and upload_response.get("files") and len(upload_response.get("files")) > 0:
                                                # Ensure 'files' is a list and has at least one element
                                                file_info_list = upload_response.get("files")
                                                if isinstance(file_info_list, list) and len(file_info_list) > 0:
                                                    file_info = file_info_list[0]
                                                    if isinstance(file_info, dict):
                                                        permalink = file_info.get("permalink")
                                                        if permalink:
                                                            results["insights_images"][simplified_insight['name']] = permalink
                                                            logger.info(f"Successfully uploaded screenshot for insight '{simplified_insight['name']}' to Slack. Permalink: {permalink}")
                                                        else:
                                                            logger.error(f"Slack upload for '{simplified_insight['name']}' successful but no permalink found in file info: {file_info}")
                                                    else:
                                                        logger.error(f"File info in Slack response is not a dictionary for '{simplified_insight['name']}'. File info: {file_info}")
                                                else:
                                                    logger.error(f"'files' field in Slack response is not a list or is empty for '{simplified_insight['name']}'. Response: {upload_response}")
                                            else:
                                                logger.error(f"Slack upload failed for insight '{simplified_insight['name']}'. Response: {upload_response.get('error', 'Unknown error') if upload_response else 'No response'}")
                                        except SlackApiError as slack_err:
                                            logger.error(f"Slack API error uploading screenshot for '{simplified_insight['name']}': {slack_err}")
                                        except Exception as upload_exc:
                                            logger.error(f"Generic error uploading screenshot for '{simplified_insight['name']}': {upload_exc}")
                                        finally:
                                            # Clean up local temporary file after attempting upload
                                            if os.path.exists(local_save_path):
                                                try:
                                                    os.remove(local_save_path)
                                                    logger.info(f"Removed temporary screenshot: {local_save_path}")
                                                except OSError as e_remove:
                                                    logger.error(f"Error removing temporary screenshot {local_save_path}: {e_remove}")
                                    else:
                                        if not slack_channel_id:
                                            logger.warning("slack_channel_id not provided. Skipping Slack upload for insight screenshot.")
                                        if not slack_bot_token:
                                            logger.warning("SLACK_BOT_TOKEN environment variable not set. Skipping Slack upload.")
                                        # If Slack upload is skipped, store the local path for now as a fallback.
                                        # This behavior can be changed if only permalinks are desired.
                                        results["insights_images"][simplified_insight['name']] = local_save_path 

                                else:
                                    logger.warning(f"Failed to save screenshot locally for insight '{simplified_insight['name']}' (ID: {insight_id}). Result from get_insight_screenshot: {screenshot_result}")
                            except Exception as screenshot_exc:
                                logger.error(f"Exception during screenshot generation or upload for insight '{simplified_insight['name']}' (ID: {insight_id}): {str(screenshot_exc)}")

                    results["insights"].append(simplified_insight)
                except Exception as e:
                    logger.error(f"Error getting insight {insight_id}: {str(e)}")
                    results["insights"].append({
                        "id": insight_id,
                        "name": item.get("insight", {}).get("name", "Unnamed insight"),
                        "description": "",
                        "error": str(e)
                    })
        
        return results
    
    def generate_ai_insights(self, dashboard_name: str, days: int = 7, insight_type: str = "daily", slack_channel_id: Optional[str] = None, slack_thread_ts: Optional[str] = None) -> str:
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
            dashboard_data = self.get_dashboard_data(dashboard_name, days, slack_channel_id, slack_thread_ts)
            
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
Write a concise and straight to the point report. Don't include any fluff. Don't beat around the bush.
"""         
            print(f"System Prompt: {system_prompt}")
            print(f"User Prompt: {prompt}")
            full_response = ""
            # Get AI response
            insights = ai_client.response(prompt=prompt, system_prompt=system_prompt, stream=True)
            for chunk in insights:
                if chunk.type == "response.output_text.delta":
                    print(chunk.delta, end="", flush=True)
                    full_response += chunk.delta
            
            print("\n \n")
            return full_response
        
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            return f"Error generating AI insights: {str(e)}"
        
    def _get_date_and_week_range(
        self,
        date_str: Optional[str] = None,
        fmt: str = "%Y-%m-%d"
    ) -> Tuple[str, str, str]:
        """
        Return (current_date, week_start, week_end) all formatted as YYYY-MM-DD in UTC timezone.
        
        - date_str: optional date string; if None, uses today in UTC.
        - fmt: the format for parsing/formatting dates.
        """
        # 1) Determine the "current" date in UTC
        if date_str:
            current = datetime.strptime(date_str, fmt).date()
        else:
            current = datetime.now(timezone.utc).date()

        # 2) Find Monday of that week (weekday() == 0)
        week_start = current - timedelta(days=current.weekday())
        # 3) Sunday is 6 days after Monday
        week_end = week_start + timedelta(days=6)

        # 4) Return all three as strings
        return (
            current.strftime(fmt),
            week_start.strftime(fmt),
            week_end.strftime(fmt),
        )
    
    
    def _get_daily_analysis_prompt(self, dashboard_name: str, data: Dict, days: int) -> str:
        """Generate prompt for daily analysis."""        
        prompt = f"""Analyze the following Posthog analytics data for {dashboard_name} over the past {days} days.
 
Dashboard: {dashboard_name}
Today's date: {self._get_date_and_week_range()[0]}
Current week that spans from: start date: {self._get_date_and_week_range()[1]} to end date: {self._get_date_and_week_range()[2]}

Insights:
{data.get("insights")}
"""
        prompt += "\n"
        
        prompt += """
Please provide a daily analysis with the following sections:
1. Summary - Overall health and key changes
2. Significant Changes (If any) - Detailed analysis of notable metrics (both positive and negative)
3. Recommendations - 2-3 actionable insights based on the data

Format the analysis as a Slack message. Use emoji indicators for clarity.
Focus on actionable insights rather than just describing the data.
Be aware of the time period (today's date and current week) of the data you are analyzing. Don't say data is drop when it's actually not the end of the week/month.
"""
        return prompt
    
    def _get_weekly_analysis_prompt(self, dashboard_name: str, data: Dict, days: int) -> str:
        """Generate prompt for weekly analysis."""
        
        prompt = f"""Analyze the following Posthog analytics data for {dashboard_name} over the past {days} days.

Dashboard: {dashboard_name}
Today's date: {self._get_date_and_week_range()[0]}
Current week that spans from: start date: {self._get_date_and_week_range()[1]} to end date: {self._get_date_and_week_range()[2]}
Period: {data.get("period", f"{days} days")}

Insights:
{data.get("insights")}

Insight images:
{data.get("insights_images")}
"""
        prompt += "\n"
        
        prompt += """
Please provide a comprehensive weekly analysis with the following sections:
1. Insights - Blazing fast point-by-point look at only top 3 important/notable metrics, trends, and patterns. Include the image URL of each insight under it's analysis.
2. Recommendations - 3 data-driven, actionable recommendations

General rule:
- Get straight to the point without any heading. Make sure the entire report does not exceed 386 words.
- Focus on extracting valuable insights rather than just describing numbers.
- Pay attention to the time period (today's date and current week) of the data you are analyzing. If the current week's not finish, focus on the previous weeks' data. Only give comparison on weeks that are complete (have full week's data).
- Only give a brief update of the current week's data. And it should not be compared with the previous weeks' data since it's not complete yet.
- For each insight, if an image URL is available in 'Insight images', include it directly under its analysis using the Slack link format: <IMAGE_URL|View {Insight Name} Image>. Do NOT use Markdown [Text](URL) format. If an image URL is not related to the insight, just ignore it.
- Remember, if a week is not complete, compare it with completd weeks data makes no sense at all. Just skip it entirely, focus on the complete weeks.
- Add 2 new lines between each section.

IMPORTANT: Creatively use these Slack's supported markdown as much as you can to make the report more readable. But do not use other markdown formatting that isn't listed here:
For highlighting text, let's use inline code block for formatting. Sometimes, you can also use bold and italics but I believe inline code block is more readable.
- Use *text* for bold
- Use _text_ for italics
- Use `code` for inline code
- For block code
- Use > for block quotes
- Use line breaks with \n
- For links, use: <URL|Source (or whatever display text you want)>
- For lists, use - followed by a space
- Do not use #, ##, ### for headers
AGAIN: DO NOT USE OTHER MARKDOWN FORMATTING THAT IS NOT LISTED HERE.
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
    
    def generate_weekly_report(self, dashboard_names: List[str], slack_channel_id: Optional[str] = None, slack_thread_ts: Optional[str] = None) -> str:
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
            dashboard_insights = self.generate_ai_insights(dashboard_name, days=28, insight_type="weekly", slack_channel_id=slack_channel_id, slack_thread_ts=slack_thread_ts)
            dashboard_insights = f"*Weekly Report for {dashboard_name}*\n\n`{dashboard_insights}`\n\n---\n\n"
        
        # If there are multiple dashboards, add a cross-dashboard analysis
        if len(dashboard_names) > 1:
            # Initialize OpenAI client for cross-dashboard analysis
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            ai_client = OpenaiClient(openai_api_key, model="gpt-4.1-2025-04-14")
            
            # Generate cross-dashboard insights
            system_prompt = """You are TMAI Agent, a helpful assistant operate within the company called Token Metrics, a company works in the field of crypto and AI.
You synthesize insights across multiple dashboards to identify holistic patterns and business opportunities."""
            
            cross_dashboard_prompt = f"""Based on the individual dashboard analyses below, provide a cross-dashboard synthesis that identifies:
1. Overarching trends across all dashboards
2. Correlations between metrics in different dashboards
3. Strategic recommendations based on the full picture

Individual dashboard analyses:
{dashboard_insights}

Present your analysis in a clear, concise format suitable for executives and stakeholders.
"""
            cross_analysis = ai_client.response(prompt=cross_dashboard_prompt, system_prompt=system_prompt)
            
            # Add the cross-analysis to the report
            final_report = f"# Weekly Cross-Dashboard Analysis\n\n{cross_analysis}\n\n---\n\n{combined_insights}"
            return final_report
        
        # If only one dashboard, return the single analysis
        return dashboard_insights
        
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

    
    def wait_for_export(self, export_id: str, max_retries: int = 60, delay: int = 5) -> Dict:
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
                                          'daily_report', 'weekly_report', 'ai_daily_insights', 'ai_weekly_insights', 'export_dashboard_screenshots', 'test_date_and_week_range'],
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
        print(f"Dashboard data: \n{dashboards}\n")
            
    elif args.action == 'dashboard_details':
        if not args.dashboard:
            print("Error: --dashboard is required for dashboard_details action")
            return
            
        dashboard = posthog_client.get_dashboard_by_name(args.dashboard)
        if not dashboard:
            print(f"Error: Dashboard '{args.dashboard}' not found")
            return
            
        data = posthog_client.get_dashboard_data(args.dashboard)
        for dashboard in data:
            print(f"Dashboard name: {dashboard.get('name')}")
            print(f"Dashboard id: {dashboard.get('id')}")
            print(f"Dashboard created at: {dashboard.get('created_at')}")
            print(f"Dashboard insights: \n{dashboard.get('insights')}\n")
        

    elif args.action == 'insight_data':
        if not args.insight:
            print("Error: --insight is required for insight_data_by_date action")
            return
            
        print(f"\nGetting data for insight {args.insight} over {args.days} days:")
        insight_data = posthog_client.get_insight_data(args.insight, args.days)

    
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
    
    elif args.action == 'test_date_and_week_range':
        print(posthog_client._get_date_and_week_range())

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