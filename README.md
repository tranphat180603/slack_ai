# Slack AI Assistant

A powerful AI assistant that integrates with Slack and Linear to help teams manage projects, answer questions, and automate workflows.

## Overview

Slack AI Assistant is an intelligent agent that lives in your Slack workspace. It provides a natural language interface to:

- Search and manage Linear issues, projects, and cycles
- Access semantic search across your organization's knowledge base
- Execute complex workflows and provide detailed information about your team's work

## Features

### Linear Integration

- **Issue Management**: Search, create, update, and filter issues across teams
- **Project Tracking**: View projects, associated issues, and status updates
- **Cycle Management**: Track sprints and workloads across multiple teams
- **Comment & Attachment Management**: View and manage issue comments and attachments

### Semantic Search

- **Natural Language Queries**: Ask questions about your work in plain English
- **Context-Aware Responses**: Get relevant information based on team context
- **Intelligent Reranking**: LLM-powered reranking for more accurate search results

### Slack Features

- **Channel History Search**: Find previous messages and discussions
- **User Information**: Get details about team members
- **Channel Management**: Access channel information and membership

## Architecture

The system is composed of several key components:

- **Slack Interface**: Handles interactions within Slack channels
- **Linear Client**: Provides direct access to Linear's GraphQL API
- **Semantic Search Engine**: Vector-based search using pgvector for intelligent queries
- **Parameter Adaptation Layer**: Translates natural language into structured API calls
- **Tools Declaration System**: Flexible interface for adding new capabilities

## Setup

### Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- Linear API key
- Slack API credentials
- OpenAI API key

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/slack_ai.git
   cd slack_ai
   ```

2. Create and activate a conda environment
   ```bash
   conda create -n dev_env python=3.8
   conda activate dev_env
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (create a `.env` file)
   ```
   LINEAR_API_KEY=your_linear_api_key
   SLACK_APP_TOKEN=your_slack_app_token
   SLACK_BOT_TOKEN=your_slack_bot_token
   OPENAI_API_KEY=your_openai_api_key
   DATABASE_URL=postgresql://user:password@localhost:5432/db_name
   ```

5. Initialize the database
   ```bash
   python ops_linear_db/linear_rag_db_create.py
   ```

6. Sync Linear data
   ```bash
   python ops_linear_db/linear_rag_sync.py
   ```

## Usage

### Querying Linear Issues

The agent can search for Linear issues using natural language:

- "Show me open bugs in the ENG team"
- "What issues are assigned to Jake in the current cycle?"
- "Find all high priority tasks in the AI team"

### Semantic Search

You can perform semantic searches across your knowledge base:

- "How do I fix the API rate limit error?"
- "What was our approach to implementing feature X?"

**Note:** Semantic search works best for content-based queries rather than metadata-based searches. Queries like "tasks assigned to phat due this week" may not work as expected since the embedded data only contains issue titles and descriptions, not metadata like assignees or due dates.

### Slack Integration

The agent can help with Slack-related tasks:

- "Find messages about the deployment from last week"
- "Who are the members of the #engineering channel?"
- "Get information about @username"

## Development

### Project Structure

- `ops_linear_db/`: Linear database operations and RAG functionality
- `ops_slack/`: Slack integration and tools
- `tools/`: Tool declarations, schemas, and adapters
- `test_*.py`: Test scripts for different components

### Adding New Features

1. Define the schema in `tools/tool_schema_*.py`
2. Implement the functionality in the appropriate module
3. Add parameter adaptation in `tools/tools_declaration.py`
4. Update the tools collection to include your new feature