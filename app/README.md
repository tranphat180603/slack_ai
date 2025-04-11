# TMAI Slack Agent

A powerful AI assistant that integrates with Slack and Linear to provide automated support, search capabilities, and task management.

## Architecture

The TMAI Slack Agent uses a three-phase workflow:

1. **Planning Phase**: Determines which tools to use based on the user query
2. **Execution Phase**: Executes tools sequentially with intelligent re-planning
3. **Response Generation Phase**: Creates a cohesive response from all gathered information

```
+-------------------------------------------------------------------------+
|                          TMAI SLACK AGENT                               |
+--------------------------------|--------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------------+
|                            USER MESSAGE                                  |
+--------------------------------|--------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------------+
|                      CONVERSATION MANAGEMENT                             |
|                                                                          |
|  +------------------+                     +------------------+           |
|  | Load Context &   |<------------------>| Context Manager   |           |
|  | History from DB  |                     +------------------+           |
|  +------------------+                                                    |
|                                                                          |
|  +------------------+                                                    |
|  | Progressive Msg  |-------------------------+                          |
|  | Handler          |                         |                          |
|  +------------------+                         |                          |
+--------------------------------|--------------------------------------+
                                 |              |
                                 |              v
                                 |    +----------------------+
                                 |    |   SLACK UPDATES     |
                                 |    +----------------------+
                                 |
                                 v
+-------------------------------------------------------------------------+
|                           AGENT WORKFLOW                                 |
|                                                                          |
|  +------------------+       +------------------+      +------------------+
|  |    Commander     |       |     Captain      |      |     Soldier      |
|  | (Planning Phase) |       | (Evaluation &    |      | (Tool Execution) |
|  |                  |       |  Re-Planning)    |      |                  |
|  |  - Determine     |------>|  - Evaluate      |<-----|  - Execute tools |
|  |    required tools|       |    results       |      |    based on plan |
|  |  - Create        |       |  - Decide if     |      |  - Call APIs     |
|  |    execution plan|       |    replanning    |      |    (Linear/Slack)|
|  +------------------+       |    is needed     |      +------------------+
|                             +------------------+              |
|                                     |                         |
+--------------------------------|-----------------------------+
                                 |    |
                                 |    |
                                 v    v
+-------------------------------------------------------------------------+
|                        RESPONSE GENERATION                               |
|                                                                          |
|  +------------------+       +------------------+      +------------------+
|  | Process Results  |------>| Generate Final   |----->| Format & Send    |
|  | and Context      |       | Response         |      | to Slack         |
|  +------------------+       +------------------+      +------------------+
|                                                                          |
+-------------------------------------------------------------------------+
```

### Core Components

- **TMAISlackAgent**: Main agent class that orchestrates the workflow
- **ConversationManager**: Manages conversation history and context
- **ProgressiveMessageHandler**: Manages UI updates during processing
- **Tool Integrations**: Slack and Linear API clients

## Configuration

### Environment Variables

```
# Required
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_USER_TOKEN=xoxp-your-user-token
OPENAI_API_KEY=sk-your-openai-key
LINEAR_API_KEY=lin-your-linear-key

# Optional
PORT=3000                     # Default: 3000
AI_MODEL=gpt-4o              # Default: gpt-4o
ENVIRONMENT=production       # Default: development
LOG_LEVEL=WARNING            # Default: INFO
DATABASE_URL=postgres://...  # Required for conversation persistence
```

### Feature Flags

Set these environment variables to enable/disable features:

```
ENABLE_PROGRESSIVE_RESPONSES=true  # Show thinking stages (default: true)
ENABLE_RERANKING=true              # Enable semantic search reranking (default: true)
MAX_ITERATIONS=3                   # Max replanning iterations (default: 3)
```

## Deployment

### Standard Deployment

1. Clone the repository:
   ```
   git clone https://github.com/your-org/slack_ai.git
   cd slack_ai
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set environment variables (see Configuration section)

4. Run the application:
   ```
   cd app
   uvicorn app:app --host 0.0.0.0 --port 3000
   ```

### EdgeFunctions Deployment

For deploying to EdgeFunctions.com:

1. Create an account at [EdgeFunctions.com](https://edgefunctions.com)

2. Package the application:
   ```
   zip -r tmai-edge.zip app/* llm/* linear_db/* slack_ops/* tools/* rate_limiter.py requirements.txt
   ```

3. Create a new Edge Function in the dashboard:
   - Name: `tmai-slack-agent`
   - Runtime: `Python 3.10`
   - Memory: `512MB` (minimum)
   - Timeout: `60s`

4. Upload the zip file and set environment variables in the EdgeFunctions dashboard

5. Configure the function as an HTTP handler:
   - Entry point: `app.app:app`
   - Method: `*` (all methods)

6. Deploy and obtain your function URL

7. Update your Slack Event Subscriptions with the EdgeFunctions URL

## Usage

### In Slack

Users can interact with TMAI in two ways:

1. **Direct Messages**: Send DMs to the TMAI bot
2. **Mentions**: Mention @TMAI in channels where the bot is present

Example commands:

```
@TMAI What's the status of the mobile app project?
@TMAI Create a bug ticket for the login page crash
@TMAI Search for messages about the API rate limit issue
```

### Advanced Features

- **Thread Awareness**: The bot maintains context throughout conversation threads
- **Progressive Responses**: Shows "thinking" stages during processing
- **Re-planning**: Automatically adjusts its approach based on discovered information
- **Rate Limiting**: Intelligent handling of API rate limits

## Monitoring and Logs

When deployed, you can monitor the application through:

1. Slack API dashboard for event delivery
2. EdgeFunctions logs for execution details
3. Application logs (set LOG_LEVEL for verbosity)

Key metrics to monitor:
- Response time (logged as "Processed message in X seconds")
- Error rates (search for ERROR level logs)
- Rate limit events (search for "rate limit exceeded")

## Troubleshooting

### Common Issues

1. **Rate Limiting**: If you see frequent rate limit errors, consider:
   - Increasing delays between API calls
   - Implementing a queue for high-traffic periods
   - Optimizing the number of API calls per request

2. **Timeouts**: If requests time out, check:
   - EdgeFunctions timeout settings (increase if needed)
   - Complex queries that trigger too many tool calls
   - Linear and Slack API responsiveness

3. **Authentication**: If auth fails, verify:
   - API keys are correctly set in environment variables
   - Bot permissions are properly configured in Slack
   - Linear API key has sufficient permissions

## Development

### Adding New Tools

1. Define the tool schema in `tools/tool_schema_xyz.py`
2. Implement the tool in the appropriate module
3. Update the `_execute_tool` method in `TMAI_slack_agent.py` to handle the new tool
4. Update the tool mapping in the agent initialization

### Local Testing

Run the development server:
```
ENVIRONMENT=development uvicorn app:app --reload
```

Use ngrok for local webhook testing:
```
ngrok http 3000
``` 