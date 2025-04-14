# Linear Issue Management with TMAI Slack Agent

This document explains how to use the TMAI Slack Agent to interact with Linear, including creating and updating issues with support for image attachments.

## Table of Contents
1. [Creating Issues](#creating-issues)
2. [Updating Issues](#updating-issues)
3. [Working with Images](#working-with-images)
4. [Displaying Image Analysis](#displaying-image-analysis)
5. [Advanced Features](#advanced-features)

## Creating Issues

The `createIssue` function allows you to create a new issue in Linear with specified details.

### Required Parameters
- `teamKey`: The team key where the issue will be created (e.g., "ENG", "OPS", "RES", "AI", "MKT", "PRO")
- `title`: Title of the issue

### Optional Parameters
- `description`: Markdown description of the issue 
- `priority`: Priority level (0.0: None, 1.0: Urgent, 2.0: High, 3.0: Medium, 4.0: Low)
- `estimate`: Estimate points for the issue (1-7)
- `assignee_name`: Display name of the user to assign the issue to
- `state_name`: Name of the workflow state (e.g., "Todo", "In Progress", "Done")
- `label_names`: List of label names to apply to the issue
- `project_name`: Name of the project to add the issue to
- `cycle_name`: Name of the cycle to add the issue to
- `parent_issue_number`: Issue number of the parent issue

### Example Usage

When a user asks TMAI to create an issue, you can extract the necessary details from their request and use them to call the `createIssue` function.

Example prompt:
```
Create a new issue in the Engineering team about fixing the authentication bug in the login page
```

The agent would extract:
- Team key: "ENG"
- Title: "Fix authentication bug in login page"
- Description: Can be enhanced with additional details from the conversation

## Updating Issues

The `updateIssue` function allows you to update an existing issue in Linear.

### Required Parameters
- `issue_number`: The number of the issue to update

### Optional Parameters
- `title`: New title for the issue
- `description`: New markdown description for the issue
- `priority`: New priority level (0.0: None, 1.0: Urgent, 2.0: High, 3.0: Medium, 4.0: Low)
- `estimate`: New estimate points
- `assignee_name`: Display name of the user to reassign to
- `state_name`: New workflow state name
- `label_names`: New list of label names
- `project_name`: Name of the project to move the issue to
- `cycle_name`: Name of the cycle to move the issue to

### Example Usage

When a user asks TMAI to update an issue, you need to first identify the issue number and then determine what fields need to be updated.

Example prompt:
```
Update issue TKM-123 to change its priority to high and assign it to John
```

The agent would extract:
- Issue number: 123
- Priority: 2.0 (High)
- Assignee name: "John"

## Working with Images

The TMAI Slack Agent can now process images shared in Slack and use the content of those images when creating or updating Linear issues. This is especially useful when users share screenshots of bugs, UI mockups, or other visual information.

### How It Works

1. **Sharing an Image**: The user shares an image in Slack and asks TMAI to create or update an issue
2. **Image Processing**: The agent downloads and processes the image
3. **Content Analysis**: The agent analyzes the image content to extract relevant information
4. **Issue Management**: The agent uses this information to create or update a Linear issue

### Adding Images to Issues

Images can be incorporated into Linear issues in two ways:

1. **Description Embedding**: The agent can include image descriptions in the issue description using markdown
2. **Attachments**: Images can be added as attachments to an issue using the `createAttachment` function 

#### Creating an Issue with Image Context

When creating an issue based on an image:

1. The agent will analyze the image to understand its content
2. It will use this understanding to craft a meaningful title and description
3. The agent will extract any relevant details like severity, priority, or assignee from the image context

#### Updating an Issue with Image Context

When updating an issue based on an image:

1. The agent will analyze what the image shows (e.g., a bug, a UI improvement)
2. It will update appropriate fields based on this analysis
3. The agent may add the image description to the issue description or create a comment

### Example Usage

Example prompt with image:
```
Create a bug report for this screenshot of the login page error
```

The agent would:
1. Analyze the image to identify the login page error
2. Create a detailed description based on what's visible in the screenshot
3. Set appropriate priority based on the severity of the error shown
4. Create the issue with a title like "Login page error: Invalid credentials handling"

## Displaying Image Analysis

The TMAI Slack Agent can display the image analysis progress and results to users, similar to how it displays order and plan information during the workflow.

### Image Analysis in the Workflow

When an image is shared, the agent performs analysis using this process:

```python
image_analysis = image_module_client.response(
    prompt="Please analyze this image in detail, extracting all text and providing a comprehensive description.",
    system_prompt=extraction_formatted["system"],
    image_data=image_data
)

if image_analysis:
    logger.info("[Commander] Successfully extracted image context")
    image_context = image_analysis
```

### Displaying Analysis Progress

To display the image analysis progress to users, you can add a step in the `ProgressiveMessageHandler` class that shows the image is being processed:

```python
# Update stage for image analysis
message_handler.update_stage("Analyzing image content")
await message_handler.send_thinking_message(initial=False)

# After analysis is complete
message_handler.update_stage("Analyzing image content", completed=True)
```

### Displaying Analysis Results

To display the image analysis results in a code block similar to order and plan displays:

```python
# Display image analysis results in code block
if image_context:
    # Truncate if too long (for UI purposes)
    display_context = image_context
    if len(display_context) > 500:
        display_context = display_context[:497] + "..."
        
    image_analysis_output = f"```\nImage Analysis:\n{display_context}\n```"
    
    response = await asyncio.to_thread(
        self.slack_client.chat_postMessage,
        channel=ai_request.channel_id,
        thread_ts=effective_thread_ts,
        text=f"{image_analysis_output}"
    )
    
    # Track message ID for later deletion
    message_handler.progress_message_ids.append(response['ts'])
```

### Using Image Analysis in Issue Creation

Here's how to integrate the image analysis results when creating a Linear issue:

```python
# Extract key information from image analysis for Linear issue
def extract_issue_details_from_image(image_context):
    # Example logic to extract issue details from image context
    title = "Issue based on image analysis"
    description = f"## Image Analysis\n\n{image_context}\n\n## Additional Notes\n\nThis issue was created based on image analysis."
    priority = 2.0  # Default to High priority for issues from screenshots
    
    return {
        "title": title,
        "description": description,
        "priority": priority
    }

# When creating a Linear issue
if image_context:
    issue_details = extract_issue_details_from_image(image_context)
    
    # Combine with user-provided details
    for key, value in issue_details.items():
        if key not in user_provided_details or not user_provided_details[key]:
            user_provided_details[key] = value
    
    # Create the issue
    createIssue(**user_provided_details)
```

### Example Progress Display Flow

1. User shares an image
2. Agent shows "Analyzing image content" progress message
3. Agent displays truncated image analysis in a code block
4. Agent proceeds with Commander, Captain, and Soldier roles as usual
5. Image context is incorporated into the Linear issue creation/update process

## Advanced Features

### Attachment Management

The agent can create attachments for issues using the `createAttachment` function:

```python
createAttachment(
    issueNumber=123,
    title="Error Screenshot",
    url="https://example.com/image.png"
)
```

### Context-Aware Issue Creation

The agent can use the broader conversation context, including shared images, to create more detailed and relevant issues. This can include:

1. Recognizing specific UI components in screenshots
2. Identifying error messages and their severity
3. Understanding user flows from sequential screenshots
4. Extracting text content from images for inclusion in issue descriptions

### Best Practices

When working with images for issue management:

1. Always confirm your understanding of the image with the user
2. Extract as much detail as possible from the image for the issue description
3. Use proper formatting in descriptions to reference image content
4. Set appropriate priority and labels based on visual evidence
5. Include the context in which the image was shared to enhance issue clarity 