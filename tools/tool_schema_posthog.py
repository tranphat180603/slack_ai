# Tool schemas for PostHog
# These define the parameters accepted by each function

# Schema for getting all dashboards
GET_DASHBOARDS_SCHEMA = {
    "type": "function",
    "name": "get_dashboards",
    "description": "Get a list of all dashboards in the PostHog project",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

# Schema for getting dashboard by name
GET_DASHBOARD_BY_NAME_SCHEMA = {
    "type": "function",
    "name": "get_dashboard_by_name",
    "description": "Find a dashboard by its name",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the dashboard to find"
            }
        },
        "required": ["name"]
    }
}

# Schema for getting dashboard by ID
GET_DASHBOARD_BY_ID_SCHEMA = {
    "type": "function",
    "name": "get_dashboard_by_id",
    "description": "Get a dashboard by its ID",
    "parameters": {
        "type": "object",
        "properties": {
            "dashboard_id": {
                "type": "string",
                "description": "The ID of the dashboard to get"
            }
        },
        "required": ["dashboard_id"]
    }
}


# Schema for getting insight data
GET_INSIGHT_DATA_SCHEMA = {
    "type": "function",
    "name": "get_insight_data",
    "description": "Get data for a specific insight/chart",
    "parameters": {
        "type": "object",
        "properties": {
            "insight_id": {
                "type": "string",
                "description": "The ID of the insight to get data for"
            },
            "days": {
                "type": "integer",
                "description": "Number of days of data to retrieve (default: 7)"
            }
        },
        "required": ["insight_id"]
    }
}

# Schema for getting dashboard data
GET_DASHBOARD_DATA_SCHEMA = {
    "type": "function",
    "name": "get_dashboard_data",
    "description": "Get all data for a dashboard including all insights",
    "parameters": {
        "type": "object",
        "properties": {
            "dashboard_name": {
                "type": "string",
                "description": "The name of the dashboard to get data for"
            },
            "days": {
                "type": "integer",
                "description": "Number of days of data to retrieve (default: 7)"
            }
        },
        "required": ["dashboard_name"]
    }
}

# Schema for getting dashboard screenshot
GET_DASHBOARD_SCREENSHOT_SCHEMA = {
    "type": "function",
    "name": "get_dashboard_screenshot",
    "description": "Get a screenshot of a dashboard as a PNG image and optionally upload to Slack",
    "parameters": {
        "type": "object",
        "properties": {
            "dashboard_id": {
                "type": "string",
                "description": "The ID of the dashboard to screenshot"
            }
        },
        "required": ["dashboard_id"]
    }
}

# Schema for getting insight screenshot
GET_INSIGHT_SCREENSHOT_SCHEMA = {
    "type": "function",
    "name": "get_insight_screenshot",
    "description": "Get a screenshot of a specific insight/chart as a PNG image and optionally upload to Slack",
    "parameters": {
        "type": "object",
        "properties": {
            "insight_id": {
                "type": "string",
                "description": "The ID of the insight to screenshot"
            }
        },
        "required": ["insight_id"]
    }
}

# Schema for saving all dashboard screenshots
SAVE_DASHBOARD_SCREENSHOTS_SCHEMA = {
    "type": "function",
    "name": "save_dashboard_screenshots",
    "description": "Save screenshots of a dashboard and all its insights to a directory",
    "parameters": {
        "type": "object",
        "properties": {
            "dashboard_name": {
                "type": "string",
                "description": "The name of the dashboard"
            },
            "output_dir": {
                "type": "string",
                "description": "Directory path to save the screenshots (e.g., '/tmp/screenshots')"
            }
        },
        "required": ["dashboard_name", "output_dir"]
    }
}

# Collection of all PostHog schemas
POSTHOG_SCHEMAS = {
    "get_dashboards": GET_DASHBOARDS_SCHEMA,
    "get_dashboard_by_name": GET_DASHBOARD_BY_NAME_SCHEMA,
    "get_dashboard_by_id": GET_DASHBOARD_BY_ID_SCHEMA,
    "get_insight_data": GET_INSIGHT_DATA_SCHEMA,
    "get_dashboard_data": GET_DASHBOARD_DATA_SCHEMA,
    "get_dashboard_screenshot": GET_DASHBOARD_SCREENSHOT_SCHEMA,
    "get_insight_screenshot": GET_INSIGHT_SCREENSHOT_SCHEMA,
    "save_dashboard_screenshots": SAVE_DASHBOARD_SCREENSHOTS_SCHEMA
}