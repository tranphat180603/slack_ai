 # TMAI Slack AI Assistant

 **TMAI** is an intelligent, extensible Slack bot built with FastAPI, OpenAI GPT models,
 and a suite of integrated tools (Linear, website search, Google Drive, image analysis).
 It leverages a three‑phase **Commander–Captain–Soldier** agent workflow to plan,
 execute, and generate contextual responses directly in Slack channels or DMs.

 <!-- toc -->
 ## Table of Contents
 - [Features](#features)
 - [Architecture](#architecture)
 - [Getting Started](#getting-started)
 - [Configuration](#configuration)
 - [Usage](#usage)
 - [Developer Guide](#developer-guide)
 - [Documentation](#documentation)
 - [Testing](#testing)
 - [Contributing](#contributing)
 <!-- tocstop -->

 ## Features

 - Conversational AI in Slack:
   - Direct Messages, mentions (`@TMAI`), and thread‑aware context
 - Three‑phase agent workflow:
   1. **Commander**: plans which tools to use based on user query
   2. **Captain**: evaluates results, re‑plans if needed
   3. **Soldier**: executes tool calls (APIs, DB queries, file processing)
 - **Tool Integrations**:
   - **Linear**: search, create, update issues via interactive Slack modals
   - **Website DB**: semantic search over custom crawled website content
   - **Google Drive**: file listing, download, upload (if enabled)
   - **Slack**: history search, message actions, interactive components
   - **Image Analysis**: base64‑encoded image summarization
   - **URL Processing**: fetch and summarize webpage content
 - Progressive UI updates:
   - "Thinking" messages with live status (progressive message handler)
   - Optional cancellation via a **Stop** button
 - Conversation persistence:
   - PostgreSQL (TimescaleDB) storage with automatic cleanup of old threads
   - Fallback to in‑memory storage if DB unavailable
 - Rate limiting:
   - Global and per‑tool limits to prevent abuse
 - Slash command scaffolding (/slack/ai_command endpoint)
 - Debug endpoints (development mode) for active context inspection
 - Fully containerized deployment (Docker + docker‑compose)

 ## Architecture

 The core components and data flow:

 1. **FastAPI Server** (`app/app.py`):
    - Receives Slack events and interactivity callbacks
    - Manages environment, logging, and background tasks
 2. **TMAISlackAgent** (`app/TMAI_slack_agent.py`):
    - Implements planning, tool execution, and response generation
    - Coordinates with `ContextManager` and `ConversationManager`
 3. **Conversation Database** (`ops_conversation_db/`):
    - Persists threaded conversation history in PostgreSQL
    - Provides history context to the agent
 4. **Tool Adapters** (`ops_linear_db/`, `ops_slack/`, `ops_website_db/`, `ops_gdrive/`):
    - Encapsulate API interactions and business logic per integration
 5. **LLM Wrapper** (`llm/openai_client.py`):
    - Handles OpenAI API calls, retries, and token usage logging
 6. **Context Manager** (`app/context_manager.py`):
    - Tracks per‑request execution state and supports cancellation

 For a detailed flow diagram and prompt structure, see [docs/explain.md](docs/explain.md).

 ## Getting Started

 ### Prerequisites
 - Docker & Docker Compose (or Python 3.10+)
 - Slack App with Bot & User Tokens (`SLACK_BOT_TOKEN`, `SLACK_USER_TOKEN`)
 - OpenAI API Key (`OPENAI_API_KEY`)
 - (Optional) Linear API Key (`LINEAR_API_KEY`)
 - (Optional) Google OAuth credentials for Drive (`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`)

 ### Quick Start with Docker Compose
 ```bash
 git clone https://github.com/your-org/tmai-slack-agent.git
 cd tmai-slack-agent
 # Copy or create a .env file with required variables
 cp .env.example .env        # if provided
 docker-compose up --build
 ```
 After startup, your FastAPI app will be listening on `http://localhost:8000`.

 ### Local Development (without Docker)
 ```bash
 # Install dependencies
 pip install -r requirements.txt

 # Create a .env file with your environment variables
 # Run FastAPI with auto-reload
 ENVIRONMENT=development uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
 ```
 Use [ngrok](https://ngrok.com) or similar to expose your local server for Slack webhooks.

 ## Configuration

 Configure via environment variables (see `.env`):

 ```ini
 # Required
 SLACK_BOT_TOKEN=...
 SLACK_USER_TOKEN=...
 OPENAI_API_KEY=...

 # Optional integrations
 LINEAR_API_KEY=...
 GOOGLE_CLIENT_ID=...
 GOOGLE_CLIENT_SECRET=...

 # Models (default all use o3-mini)
 AI_MODEL=o3-mini
 COMMANDER_MODEL=o3-mini
 CAPTAIN_MODEL=o3-mini
 SOLDIER_MODEL=o3-mini

 # App settings
 PORT=8000
 ENVIRONMENT=development   # or production
 ```

 ### Prompts
 All prompt templates live in `prompts/` as YAML files. Customize commands,
 Linear DSL, website searches, Slack messages, and more by editing these files.

 ## Usage

 1. **Invite the Bot** to your Slack workspace and relevant channels: `/invite @TMAI`.
 2. **Send a Direct Message** or **Mention** the bot in any channel:
    ```
    @TMAI What's the status of ENG-1234 in Linear?
    @TMAI summarize https://example.com
    @TMAI show me recent discussions about migrations in #dev
    ```
 3. **Interactive Modals**:
    - Create or update Linear issues via form submission
    - Confirmations appear as threaded messages

 For more examples, see [docs/usage.md](docs/usage.md).

 ## Developer Guide

 ### Code Layout
 ```
 /app                 FastAPI server & core agent implementation
 /llm                 OpenAI client wrapper
 /ops_conversation_db Conversation persistence (Postgres)
 /ops_linear_db       Linear API integration & RAG database
 /ops_slack           Slack tools & modal handlers
 /ops_website_db      Website crawl & semantic search
 /ops_gdrive          Google Drive operations
 /tools               Tool schemas & declaration
 /prompts             YAML prompt templates
 rate_limiter.py      Rate limiting logic
 context_manager.py   Execution context tracking
 ```

 ### Adding New Tools
 1. Define your tool schema in `tools/tool_schema_<name>.py`.
 2. Implement the integration under `ops_<name>/` or inline in the agent.
 3. Update `TOOLS_CONFIG` and tool dispatch logic in `TMAI_slack_agent.py`.

 ## Documentation
 Comprehensive guides, diagrams, and examples are in the `docs/` folder:
 - [explain.md](docs/explain.md)
 - [usage.md](docs/usage.md)
 - [linear_issue_management.md](docs/linear_issue_management.md)

 ## Testing
 Run the test suite with:
 ```bash
 pytest --maxfail=1 --disable-warnings -q
 ```
 Unit tests cover DB connectivity, conversation flows, and tool adapters.

 ## Contributing
 Contributions, issues, and feature requests are welcome!
 1. Fork the repo
 2. Create a feature branch (`git checkout -b feature/your-feature`)
 3. Commit your changes (`git commit -m 'Add awesome feature'`)
 4. Push to the branch (`git push origin feature/your-feature`)
 5. Open a Pull Request

 Please follow existing code style and include tests for new functionality.