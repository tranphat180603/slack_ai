# Slack AI Agent

A powerful AI assistant agent for Slack that uses advanced LLM capabilities to answer questions, search for information, and provide context-aware responses.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [System Components](#system-components)
  - [Data Flow](#data-flow)
  - [Integration Points](#integration-points)
  - [Linear RAG System](#linear-rag-system)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)

## Overview

This Slack AI Agent is designed to enhance team productivity by providing a conversational AI interface directly within Slack. It leverages OpenAI's language models to understand queries, retrieve contextual information from various sources, and generate helpful responses.

The agent can:
- Answer general knowledge questions
- Search Slack channel history for relevant messages
- Query Linear (project management tool) issues and provide RAG-based responses
- Extract information from URLs, including GitHub repositories and Twitter/X posts
- Maintain conversation context for more natural interactions

## Architecture

### System Components

```
┌─────────────────┐                                   ┌───────────────────┐
│                 │                                   │                   │
│   Slack User    │─────────── @mention ─────────────►│   FastAPI Server  │
│                 │                                   │                   │
└─────────────────┘                                   └─────────┬─────────┘
        ▲                                                       │
        │                                                       ▼
        │                                             ┌───────────────────┐
        │                                             │                   │
        │                                             │   OpenAI API      │
        │                                             │   Content Analysis│
        │                                             │                   │
        │                                             └─────────┬─────────┘
        │                                                       │
        │                                                       ▼
        │                                             ┌───────────────────┐
        │                                             │                   │
        │                                             │  Tool Selection   │
        │                                             │  & Orchestration  │
        │                                             │                   │
        │                                             └─────────┬─────────┘
        │                                                       │
        │                                                       ▼
┌───────┴───────────┐                             ┌─────────────┼─────────────┐
│                   │                             │             │             │
│  Response Stream  │◄────────────────────────────┤  AI Response Generation   │
│                   │                             │             ▲             │
└───────────────────┘                             └─────────────┼─────────────┘
                                                                │
                                  ┌─────────────────────────────┼─────────────────────────────┐
                                  │                             │                             │
                         ┌────────▼──────────┐         ┌────────▼──────────┐         ┌───────▼───────────┐
                         │                   │         │                   │         │                   │
                         │   Slack History   │         │   Linear RAG      │         │   External URLs   │
                         │   Channel Search  │         │   Knowledge Base  │         │   Content Extract │
                         │                   │         │                   │         │                   │
                         └───────────────────┘         └────────┬──────────┘         └───────────────────┘
                                                                │
                                                       ┌────────▼──────────┐
                                                       │                   │
                                                       │   PostgreSQL DB   │
                                                       │   + pgvector      │
                                                       │                   │
                                                       └───────────────────┘
```

The architecture consists of the following key components:

1. **User Interaction Flow**:
   - A user mentions the AI agent in Slack with a question or request
   - The mention event is sent to the FastAPI server via Slack's Events API
   - The agent processes the request and streams responses back to the user

2. **FastAPI Server (`slack_ai_bot.py`)**: 
   - The central component handling HTTP requests from Slack
   - Exposes endpoints for Slack events and mention events
   - Manages rate limiting and concurrent request processing
   - Orchestrates the entire flow from request to response

3. **Content Analysis System**:
   - Uses OpenAI to analyze the user's query through the `analyze_content` function
   - Classifies queries as simple prompts or tool-requiring prompts
   - Determines which specific tools (data sources) are needed:
     - Slack history retrieval
     - Linear issue search
     - URL content extraction
   - Optimizes search parameters for each data source

4. **Tool Selection & Orchestration**:
   - Based on content analysis, selects and coordinates appropriate tools
   - Manages parallel data retrieval from multiple sources
   - Combines results into a unified context for response generation

5. **Linear RAG System**:
   - **Components**:
     - `linear_client.py`: Custom API client for Linear project management
     - `linear_rag_search.py`: Semantic search implementation
     - `linear_db.py`: Database interface layer
     - `linear_rag_embeddings.py`: Vector embedding generation
     - `linear_rag_db_import.py`: Data import pipeline
   
   - **Functionality**:
     - Maintains up-to-date issue data from Linear
     - Generates and stores vector embeddings for semantic search
     - Supports complex analytical queries and natural language searches
     - Performs specialized analyses like status reports and workload distribution

6. **PostgreSQL with pgvector**:
   - Stores Linear issue data in structured tables
   - Maintains vector embeddings for semantic similarity search
   - Enables efficient hybrid search (semantic + metadata filtering)
   - Supports complex SQL queries for analytical functions

7. **Response Generation**:
   - Uses retrieved context to generate a draft response
   - Refines and formats the response for Slack's markdown format
   - Streams the response to the user in real-time for better UX

### Data Flow

1. **Request Initiation**:
   - User mentions the agent with a question in Slack
   - Slack sends the mention event to the `/slack/events` endpoint
   - Server acknowledges receipt and begins asynchronous processing

2. **Content Analysis**:
   - Agent uses OpenAI to analyze the user's question
   - Determines if the query is a simple question or requires tool use
   - Identifies which specific tools are needed for this query
   - Example classification results:
     ```json
     {
       "content_type": "prompt_requires_tool_use",
       "requires_slack_channel_history": false,
       "perform_RAG_on_linear_issues": true,
       "urls": ["https://github.com/org/repo"]
     }
     ```

3. **Data Retrieval**:
   - For Slack history: Searches channel messages with AI-optimized parameters
   - For Linear issues: Performs RAG-based search using vector embeddings
   - For URLs: Extracts content from GitHub, Twitter, or general websites
   - Each data source formats results for unified context building

4. **Linear RAG Process** (when required):
   - **Query Optimization**: AI analyzes the query to determine optimal search parameters:
     ```json
     {
       "query_type": "status_report",
       "filters": [
         {"field": "team->key", "operator": "=", "value": "ENG"},
         {"field": "assignee->name", "operator": "=", "value": "Sam"}
       ],
       "time_range": {"field": "updatedAt", "operator": ">", "value": "7d"}
     }
     ```
   - **Vector Search**: Converts query to embedding and finds similar issues in pgvector
   - **Hybrid Filtering**: Applies metadata filters from parameters
   - **Result Processing**: Formats and summarizes results for inclusion in context

5. **Response Generation**:
   - Combines retrieved context into a unified format
   - Generates draft response with OpenAI using the context
   - Refines the response for clarity and Slack formatting
   - Streams the final response to the user in real-time

6. **Conversation Maintenance**:
   - Stores the interaction in conversation history
   - Preserves thread context for follow-up questions
   - Manages history expiration for resource efficiency

### Integration Points

- **Slack Integration**:
  - Uses Slack Events API for real-time messaging
  - Supports thread-based conversations
  - Handles message formatting and updates
  - Streams responses for better user experience

- **Linear Integration**:
  - Connects to Linear API via GraphQL
  - Synchronizes issue data to PostgreSQL
  - Supports complex analytical queries
  - Enables natural language search through vector embeddings

- **External Services**:
  - GitHub API for repository analysis
  - Twitter/X API for tweet content extraction
  - General web scraping for other URLs

- **OpenAI Integration**:
  - Uses multiple AI models for different purposes:
    - Content analysis: Determines query type and tool requirements
    - Search optimization: Enhances search parameters
    - Response generation: Creates informative responses
    - Response refinement: Formats for Slack

### Linear RAG System

The Linear RAG (Retrieval Augmented Generation) system is a key component of the architecture, providing semantic search capabilities and issue management:

1. **Data Ingestion Pipeline**:
   - `linear_rag_db_create.py`: Creates database schema with vector support
   - `linear_rag_db_import.py`: Imports issues from Linear API
   - `linear_data_gen.py`: Generates structured data for analytics

2. **Database Schema**:
   ```
   ┌────────────────────┐      ┌────────────────────┐
   │ issues             │      │ issue_embeddings   │
   ├────────────────────┤      ├────────────────────┤
   │ id (PK)            │──┐   │ id (PK)            │
   │ issue_id           │  └──►│ issue_id (FK)      │
   │ title              │      │ embedding (vector) │
   │ description        │      │ embedding_model    │
   │ state              │      └────────────────────┘
   │ assignee           │      
   │ team_id            │      ┌────────────────────┐
   │ cycle_id           │      │ teams              │
   │ priority           │      ├────────────────────┤
   │ created_at         │      │ id (PK)            │
   │ updated_at         │      │ team_id            │
   │ labels             │      │ name               │
   └────────────────────┘      │ key                │
                               └────────────────────┘
   ```

3. **Vector Search Implementation**:
   - Queries are converted to embeddings using OpenAI
   - PostgreSQL pgvector extension enables similarity search
   - Hybrid queries combine vector search with metadata filters:
   ```sql
   SELECT i.*, 
          1 - (e.embedding <=> query_embedding) AS similarity
   FROM issues i
   JOIN issue_embeddings e ON i.issue_id = e.issue_id
   WHERE i.team_id = 'ENG'
   ORDER BY similarity DESC
   LIMIT 10;
   ```

4. **Issue Management Capabilities**:
   - **Search and Analysis**:
     - Natural language semantic search
     - Complex analytical queries
     - Status reports and metrics
   - **Issue Creation**:
     - Natural language processing to generate issue details
     - AI-powered team assignment and priority setting
     - Automatic formatting and structuring of descriptions
     - Example creation format:
     ```json
     {
       "title": "Implement new feature X",
       "description": "Detailed description...",
       "team_key": "ENG",
       "priority": 2  // Priority levels: 0-4
     }
     ```

5. **Query Types**:
   - **Basic Issue Search**: Find issues similar to a natural language query
   - **Status Reports**: Generate team or project status summaries
   - **Performance Metrics**: Calculate completion rates and cycle times
   - **Workload Distribution**: Analyze assignment patterns across team members
   - **Time-based Analysis**: Track progress over sprints or time periods
   - **Issue Creation**: Create new issues using natural language descriptions

6. **Example Operations**:
   - "Show all high-priority issues assigned to the Engineering team"
   - "Generate a status report for the current sprint"
   - "What issues has Sam completed in the last two weeks?"
   - "Compare task completion rates between Frontend and Backend teams"
   - "Create a high-priority issue for the AI team to implement feature X"
   - "Make a new issue to track the database migration task"

7. **System Architecture**:
   ```
   ┌─────────────────┐
   │  Slack Input    │
   │  (User Query)   │
   └────────┬────────┘
            ▼
   ┌────────────────┐    ┌─────────────────┐
   │ Content        │    │ Linear Issue    │
   │ Analysis       │───►│ Creation        │
   └────────┬───────┘    └────────┬────────┘
            │                     │
            ▼                     ▼
   ┌────────────────┐    ┌─────────────────┐
   │ Vector Search  │    │ Issue           │
   │ & Filtering    │    │ Management      │
   └────────┬───────┘    └────────┬────────┘
            │                     │
            └──────────┬──────────┘
                      ▼
   ┌────────────────────────────┐
   │      PostgreSQL DB         │
   │      + pgvector            │
   └────────────────────────────┘
   ```

This updated architecture reflects both the search and creation capabilities of the Linear RAG system, providing a comprehensive solution for issue management and analysis.

## Features

- **Natural Language Understanding**: Process user queries in natural language
- **Context-Aware Responses**: Maintain conversation history for more coherent interactions
- **Multi-Source Information Retrieval**: Search across Slack, Linear, and external URLs
- **Real-time Response Streaming**: Stream responses as they're generated for better UX
- **Advanced Linear Integration**: Search, analyze, and summarize project management data
- **URL Content Extraction**: Extract and analyze content from external URLs
- **Conversation Memory**: Remember previous interactions within threads

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/slack_ai.git
cd slack_ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env` file:
```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key
SLACK_BOT_TOKEN=your_slack_bot_token
SLACK_USER_TOKEN=your_slack_user_token
LINEAR_API_KEY=your_linear_api_key

# Optional API Keys
GITHUB_TOKEN=your_github_token
X_BEARER_TOKEN=your_twitter_bearer_token

# Configuration
AI_MODEL=o1-mini
AI_RATE_LIMIT=30

# Database Configuration for Linear Integration
POSTGRES_HOST=localhost
POSTGRES_DB=linear_rag
POSTGRES_USER=phattran
POSTGRES_PASSWORD=yourpassword
```

4. Set up the PostgreSQL database for Linear integration:
```bash
python linear_rag_db_create.py
python linear_rag_db_import.py
```

## Configuration

The agent behavior can be customized through:

1. **Environment Variables**: Set in `.env` file to control API keys, model selection, etc.

2. **Prompts**: Edit `prompts.yaml` to customize AI system prompts for different components:
   - `analyze_content`: Determines query type and required actions
   - `slack_search_operator`: Optimizes Slack history search parameters
   - `linear_search_operator`: Optimizes Linear query parameters
   - `draft_response`: Generates initial response text
   - `final_response`: Refines response for Slack formatting

## Usage

1. Start the server:
```bash
python slack_ai_bot.py
```

2. Configure your Slack app to send events to your server's endpoint:
   - Expose your server publicly (e.g., using ngrok)
   - Set the Events API URL in Slack App configuration to `https://your-domain.com/slack/events`
   - Subscribe to the `app_mention` event

3. Interact with the agent in Slack:
   - Mention the agent with `@youragent` followed by your question
   - The agent will process your query and respond in the thread

## Development

- **Adding New Features**: Extend the agent by adding new response generation methods in `slack_ai_bot.py`
- **Custom Prompts**: Modify prompts in `prompts.yaml` to change AI behavior
- **New Integrations**: Follow the pattern in existing integrations to add new data sources 