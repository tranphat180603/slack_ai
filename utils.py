import re


def safe_append(context_list, content):
    """
    Safely append content to the context list, ensuring it's always a string.
    
    Args:
        context_list: The list to append to (usually context_parts)
        content: The content to append (could be string, list, or other)
    """
    if isinstance(content, list):
        # If it's a list, join it with newlines
        context_list.append("\n".join(str(item) for item in content))
    elif content is not None:
        # For any other type, convert to string
        context_list.append(str(content))
    # If None, don't append anything

def format_for_slack(text: str) -> str:
    """
    Convert standard markdown formatting to Slack-compatible mrkdwn formatting.
    
    Args:
        text: Text with potential markdown formatting
        
    Returns:
        Text with Slack-compatible formatting
    """
    if not text:
        return text
        
    # Replace double asterisks with single (for bold)
    text = re.sub(r'\*\*([^*]+)\*\*', r'*\1*', text)
    
    # Replace double underscores with single (for italic)
    text = re.sub(r'__([^_]+)__', r'_\1_', text)
    
    # Replace markdown headers with bold text
    text = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
    
    # Replace triple backticks with single backticks for inline code
    text = re.sub(r'```([^`]+)```', r'`\1`', text)
    
    # Fix numbered lists (ensure there's a space after the period)
    text = re.sub(r'^(\d+)\.([^\s])', r'\1. \2', text, flags=re.MULTILINE)
    
    # Fix bullet points (ensure there's a space after the asterisk)
    text = re.sub(r'^\*([^\s])', r'* \1', text, flags=re.MULTILINE)
    
    # Remove language specifiers from code blocks
    text = re.sub(r'```[a-zA-Z0-9]+\n', r'```\n', text)
    
    return text