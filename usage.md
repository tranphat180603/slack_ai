# Token Metrics Slack AI Agent - Usage Guide

## Introduction

The Token Metrics Slack AI Agent is an intelligent bot designed to help you with a variety of tasks, from answering questions to searching through Slack history and managing Linear issues. This guide explains how to use the bot effectively.

## Getting Started

To interact with the AI Agent, simply mention it in any channel or send a direct message: (or add it in the channel if you haven't. Use /invite and enter the name TMAI. Finally add that app to a channel)

```
@TMAI Agent how are you doing?
```

## Capabilities

The AI Agent can:

- Answer general knowledge questions
- Search through Slack channel history
- Retrieve information about Linear issues and team status
- Create new Linear issues
- Process and analyze URL content
- Maintain conversation context across threads

## Usage Examples

### General Questions

You can ask the bot general questions about Token Metrics, cryptocurrency, blockchain, or AI:

```
@TMAI Agent what is Token Metrics?
@TMAI Agent explain the current state of Bitcoin
@TMAI Agent give me a summary of the latest AI trends
```

### Slack History Search

Search through past conversations in channels:

```
@TMAI Agent show me recent discussions about API keys
@TMAI Agent what did @jake say about the database migration last week?
@TMAI Agent find messages about the new website design from the last 3 days
```

### Linear Issue Management

#### Querying Linear Issues

Get information about ongoing work:

```
@TMAI Agent, what tasks do I have this week?
@TMAI Agent what's the status of ENG-4859?
@TMAI Agent show me open high-priority issues for the AI team
@TMAI Agent who's working on what in the current cycle?
@TMAI Agent how many issues are in progress for the ENG team?
@TMAI Agent I need you to groom this linear ticket: Modify StarrocksUtility To Support Write Data On To GCS Before Writing To StarRocks.
```

#### Creating Linear Issues

Create new issues directly from Slack:

```
@TMAI Agent create an issue titled "Update API documentation" with description "We need to update our API docs to reflect recent changes" and assign it to @phat
@TMAI Agent create a high priority task for the AI team to fix the recommendation engine
```

### Following Up in Threads

The AI maintains context within threads, so you can ask follow-up questions:

```
You: @TMAI Agent what tasks does the ENG team have this cycle?
AI: [Shows Linear search results for ENG team]
You: what about the AI team?
```

### URL Processing

Share URLs with the bot to analyze content:

```
@TMAI Agent check out this website: https://example.com and tell me what it's about
```

## Best Practices

1. **Be specific** - The more specific your question, the better the response.
2. **Use threads** - Keep conversations in threads to maintain context.
3. **Mention names clearly** - When referring to team members, use @ mentions.
4. **Specify time ranges** - For history searches, mention the time period (e.g., "last week").
5. **Include details for issues** - When creating issues, specify title, description, priority, and assignee.
