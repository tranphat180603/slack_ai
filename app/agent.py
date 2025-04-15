import os
import json
import logging
import dotenv
import yaml
import time
from typing import Dict, List, Any, Optional, Union
import re

from llm.openai_client import OpenaiClient, CancellableOpenAIClient, CancellationError
from tools import LINEAR_SCHEMAS, SLACK_SCHEMAS, SEMANTIC_SEARCH_SCHEMAS

# Configure logger
logger = logging.getLogger("agent")

if os.getenv("environment") == "development":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

dotenv.load_dotenv()

class APICallTracker:
    """Tracks API calls made by agent components."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset API call statistics."""
        self.component_calls = {}
        self.function_calls = {}
        self.start_time = time.time()
    
    def track_call(self, component_name: str, function_name: str, execution_time: float = None):
        """Track an API call."""
        # Track by component
        if component_name not in self.component_calls:
            self.component_calls[component_name] = {
                "total_calls": 0,
                "functions": {}
            }
        
        # Increment total calls for this component
        self.component_calls[component_name]["total_calls"] += 1
        
        # Track function within component
        if function_name not in self.component_calls[component_name]["functions"]:
            self.component_calls[component_name]["functions"][function_name] = {
                "calls": 0,
                "total_execution_time": 0
            }
        
        self.component_calls[component_name]["functions"][function_name]["calls"] += 1
        
        # Add execution time if provided
        if execution_time is not None:
            self.component_calls[component_name]["functions"][function_name]["total_execution_time"] += execution_time
        
        # Track by function across components
        if function_name not in self.function_calls:
            self.function_calls[function_name] = {
                "total_calls": 0,
                "components": {},
                "total_execution_time": 0
            }
        
        # Increment total calls for this function
        self.function_calls[function_name]["total_calls"] += 1
        
        # Track component within function
        if component_name not in self.function_calls[function_name]["components"]:
            self.function_calls[function_name]["components"][component_name] = 0
        
        self.function_calls[function_name]["components"][component_name] += 1
        
        # Add execution time if provided
        if execution_time is not None:
            self.function_calls[function_name]["total_execution_time"] += execution_time
    
    def get_report(self) -> Dict[str, Any]:
        """Generate a report of API calls."""
        total_runtime = time.time() - self.start_time
        
        # Count total calls across all components
        total_calls = sum(comp["total_calls"] for comp in self.component_calls.values())
        
        return {
            "total_runtime_seconds": total_runtime,
            "total_calls": total_calls,
            "components": self.component_calls,
            "functions": self.function_calls
        }
    
    def log_report(self, log_level=logging.DEBUG):
        """Log API call report at the specified level."""
        report = self.get_report()
        
        if logger.isEnabledFor(log_level):
            logger.log(log_level, "===== API CALL REPORT =====")
            logger.log(log_level, f"Total runtime: {report['total_runtime_seconds']:.2f} seconds")
            logger.log(log_level, f"Total API calls: {report['total_calls']}")
            
            logger.log(log_level, "\nCalls by component:")
            for component, data in report['components'].items():
                logger.log(log_level, f"  {component}: {data['total_calls']} calls")
                for function, func_data in data['functions'].items():
                    avg_time = func_data['total_execution_time'] / func_data['calls'] if func_data['calls'] > 0 else 0
                    logger.log(log_level, f"    - {function}: {func_data['calls']} calls, avg time: {avg_time:.4f}s")
            
            logger.log(log_level, "\nCalls by function:")
            for function, data in report['functions'].items():
                avg_time = data['total_execution_time'] / data['total_calls'] if data['total_calls'] > 0 else 0
                logger.log(log_level, f"  {function}: {data['total_calls']} calls, avg time: {avg_time:.4f}s")
                logger.log(log_level, f"    Used by: {', '.join(data['components'].keys())}")
            
            logger.log(log_level, "============================")


# Create a global API call tracker instance
api_call_tracker = APICallTracker()

class Commander:
    def __init__(self, model: str, prompts: Dict = None, context_id: str = None):
        self.model = model
        self.client = CancellableOpenAIClient(os.getenv("OPENAI_API_KEY"), model=model, context_id=context_id)
        self.prompts = prompts or {}
        logger.info(f"Commander initialized with model {model}")

    def set_context_id(self, context_id: str):
        """Set the context ID for cancellation checks."""
        self.client.set_context_id(context_id)

    def format_prompt(self, prompt_name: str, prompt_vars: Dict[str, Any]) -> Dict[str, str]:
        """Format a prompt template with provided variables."""
        if not self.prompts:
            logger.warning("No prompts loaded, using direct prompt")
            return {"system": None, "user": json.dumps(prompt_vars)}
            
        # Parse prompt name (e.g., "commander.assign_tasks" -> ["commander", "assign_tasks"])
        parts = prompt_name.split(".")
        if len(parts) < 2:
            logger.warning(f"Invalid prompt name format: {prompt_name}")
            return {"system": None, "user": json.dumps(prompt_vars)}
        
        # Handle different prompt naming patterns
        if len(parts) == 2:
            # For backwards compatibility: "commander.assign_tasks" -> "command.commander.assign_tasks"
            platform, agent, action = "command", parts[0], parts[1]
        else:
            # Standard format: "command.commander.assign_tasks"
            platform, agent, action = parts[0], parts[1], parts[2]
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Commander] Format prompt called with name: {prompt_name}")
            logger.debug(f"[Commander] Using platform={platform}, agent={agent}, action={action}")
        
        # Look for prompt using platform.agent.action structure
        if platform in self.prompts and agent in self.prompts[platform] and action in self.prompts[platform][agent]:
            template_data = self.prompts[platform][agent][action]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Commander] Found prompt template using platform.agent.action structure")
        else:
            # Fallback for backwards compatibility
            if agent in self.prompts and action in self.prompts[agent]:
                template_data = self.prompts[agent][action]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[Commander] Found prompt template using legacy agent.action structure")
            else:
                logger.warning(f"Prompt not found: {prompt_name}")
                return {"system": None, "user": json.dumps(prompt_vars)}
        
        system_template = template_data.get("system_template", "")
        user_template = template_data.get("user_template", "")
        
        # Format templates with variables
        formatted_user = user_template
        for var_name, var_value in prompt_vars.items():
            placeholder = f"{{{{{var_name}}}}}"
            if isinstance(var_value, (dict, list)):
                var_value = json.dumps(var_value, indent=2)
            formatted_user = formatted_user.replace(placeholder, str(var_value))
        
        return {
            "system": system_template,
            "user": formatted_user
        }

    def assign_tasks(self, user_query: str, history: Union[List[Dict], str] = None, image_data: str = None, image_context: str = None) -> Dict[str, Any]:
        """Assign tasks between different platforms based on user query
        
        Parameters:
            user_query (str): The user's request
            history (Union[List[Dict], str], optional): Conversation history
            image_data (str, optional): Base64-encoded image data for vision models
            image_context (str, optional): Pre-analyzed image context (if already analyzed)
        """
        logger.debug(f"[Commander] assign_tasks called with query: {user_query}")
        
        start_time = time.time()
        
        # Format history for prompt
        history_text = ""
        if history:
            # Handle history as either a list of dicts or a pre-formatted string
            if isinstance(history, list):
                history_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in history])
            else:
                # If history is already a string, use it directly
                history_text = history
        
        # If image data is provided but no pre-analyzed context, analyze it
        if image_data and not image_context:
            logger.info("[Commander] Image data provided but no pre-analyzed context")
            image_context = self.analyze_image(image_data)
        
        # Prepare prompt variables
        prompt_vars = {
            "user_query": user_query,
            "history": history_text
        }
        
        # Add image context if available
        if image_context:
            prompt_vars["image_context"] = image_context
            logger.debug(f"[Commander] Added image_context to prompt variables: {image_context[:100]}...")
        else:
            logger.debug("[Commander] No image_context available for prompt")
        
        # Format the prompt using template
        formatted = self.format_prompt("commander.assign_tasks", prompt_vars)
        
        try:
            # Get response from LLM based on model type
            if self.model.startswith("gpt"):
                # Log the final prompt sent to the API
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[Commander] Sending prompt to API for assign_tasks:\nSystem: {formatted['system']}\nUser: {formatted['user']}")
                
                response = self.client.response(
                    prompt=formatted["user"], 
                    system_prompt=formatted["system"]
                )
            elif self.model.startswith("o"):
                # For Claude models, combine system_template and user_template
                combined_prompt = ""
                if formatted["system"]:
                    combined_prompt = formatted["system"] + "\n\n"
                combined_prompt += formatted["user"]
                
                # Log the final prompt sent to the API
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[Commander] Sending prompt to API for assign_tasks:\n{combined_prompt}")
                
                # Check if we have image context but can't use it with reasoning models
                if image_data and image_context:
                    logger.info(f"[Commander] Adding image context directly to prompt for image model")
                    
                    # First, check if the Mustache template is in the prompt
                    template_pattern = r"{{#image_context}}\s*Image analysis:.*?{{/image_context}}"
                    if re.search(template_pattern, combined_prompt, re.DOTALL):
                        # Replace the entire Mustache block with the content
                        combined_prompt = re.sub(
                            template_pattern, 
                            f"Image analysis: {image_context}",
                            combined_prompt,
                            flags=re.DOTALL
                        )
                        logger.debug(f"[Commander] Replaced Mustache template with image context")
                    else:
                        # If template not found, just append the image context to the end
                        combined_prompt += f"\n\nImage analysis: {image_context}"
                        logger.debug(f"[Commander] Appended image context to prompt")
                        
                    # Log a sample of the final prompt
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[Commander] Final combined prompt with image: {combined_prompt[:300]}...")
                elif image_context:
                    # We have image context but no image data (shouldn't happen)
                    logger.warning(f"[Commander] Have image_context but no image_data, this is unexpected")
                
                # Note: Cannot support images with reasoning models
                response = self.client.response_reasoning(
                    prompt=combined_prompt,
                    reasoning_effort="high"
                )
            else:
                # Default to response method
                # Log the final prompt sent to the API
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[Commander] Sending prompt to API for assign_tasks:\nSystem: {formatted['system']}\nUser: {formatted['user']}")
                
                response = self.client.response(
                    prompt=formatted["user"], 
                    system_prompt=formatted["system"]
                )
                

                if (response.startswith("```json")):
                    response = response.replace("```json", "").replace("```", "")

            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Track API call
            api_call_tracker.track_call("Commander", "assign_tasks", execution_time)
            
            # Log the response from the API
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Commander] API response for assign_tasks: {response}")
            
            # Parse JSON response with more robust error handling
            try:
                # First, clean up the response from any markdown formatting
                if response.startswith("```json"):
                    response = response.replace("```json", "").replace("```", "").strip()
                elif response.startswith("```"):
                    response = response.replace("```", "").strip()
                    
                # Try to find JSON content if response has additional text
                if not response.startswith("{") and "{" in response:
                    # Try to extract JSON from the response
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        potential_json = response[json_start:json_end]
                        try:
                            result = json.loads(potential_json)
                            logger.info(f"[Commander] Successfully extracted JSON from mixed response")
                            return result
                        except json.JSONDecodeError:
                            logger.warning(f"[Commander] Failed to extract valid JSON from response")
                
                # If we reach here, try to parse the entire response as JSON
                result = json.loads(response)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from Commander response: {e}")
                logger.error(f"Response was: {response[:200]}...")  # Log part of the response for debugging
                
                # Try to salvage the situation by extracting meaningful text
                try:
                    # For direct text responses that aren't JSON, create a response object
                    cleaned_text = response.replace("```", "").strip()
                    return {
                        "order": user_query,
                        "platform": [],
                        "direct_response": cleaned_text or "I'm having trouble processing your request. Could you please rephrase or provide more details?"
                    }
                except Exception as inner_e:
                    logger.error(f"Even fallback parsing failed: {inner_e}")
                    # Final fallback response
                    return {
                        "order": user_query,
                        "platform": [],
                        "direct_response": "I'm having trouble processing your request. Could you please rephrase or provide more details?"
                    }
        except CancellationError:
            logger.info("[Commander] assign_tasks was cancelled")
            return {
                "order": user_query,
                "platform": [],
                "direct_response": "I had to stop processing your request because you asked me to stop."
            }
        except Exception as e:
            logger.error(f"Failed to get Commander response: {e}")
            # Try to salvage the situation by extracting meaningful text
            try:
                # For direct text responses that aren't JSON, create a response object
                cleaned_text = response.replace("```", "").strip()
                return {
                    "order": user_query,
                    "platform": [],
                    "direct_response": cleaned_text or "I'm having trouble processing your request. Could you please rephrase or provide more details?"
                }
            except Exception as inner_e:
                logger.error(f"Even fallback parsing failed: {inner_e}")
                # Final fallback response
                return {
                    "order": user_query,
                    "platform": [],
                    "direct_response": "I'm having trouble processing your request. Could you please rephrase or provide more details?"
                }

    def format_slack_message(self, message: str) -> str:
        """
        Format a message according to Slack markdown rules.
        Converts common markdown to Slack-specific format.
        
        Rules:
        - Use *text* for bold (not **text**)
        - Use _text_ for italics (not *text* for italics)
        - Use ~text~ for strikethrough
        - Use `code` for inline code
        - Use ```code block``` for multi-line code blocks
        - Use > for block quotes
        - For links, use <URL> or <URL|display text>
        - For lists, use - followed by a space
        - No #, ##, ### for headers
        """
        if not message:
            return message

        # Replace double asterisks with single asterisks for bold (GitHub/CommonMark style to Slack style)
        message = message.replace("**", "*")
        
        # Fix asterisks for bold that might have been doubled inadvertently
        message = message.replace("****", "**")
        
        # Replace #, ##, ### headers with bold text
        lines = message.split("\n")
        for i in range(len(lines)):
            if lines[i].startswith("# "):
                lines[i] = "*" + lines[i][2:] + "*"
            elif lines[i].startswith("## "):
                lines[i] = "*" + lines[i][3:] + "*"
            elif lines[i].startswith("### "):
                lines[i] = "*" + lines[i][4:] + "*"
            
            # Ensure lists use - (hyphen) instead of * (asterisk)
            if lines[i].strip().startswith("* "):
                lines[i] = "- " + lines[i].strip()[2:]
        
        # Rejoin processed lines
        message = "\n".join(lines)
        
        # Fix link formatting:
        # Convert markdown links [text](url) to Slack format <url|text>
        import re
        message = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', message)
        
        # Fix standalone URLs that aren't already in Slack format
        # Look for URLs not already wrapped in < >
        url_pattern = r'(?<![<|])(https?://[^\s<>"\']+(![^\s<>"\')])*)'
        message = re.sub(url_pattern, r'<\1>', message)
        
        # Handle common error: double backticks in code blocks
        # First, normalize any ```` or more to triple backticks
        message = re.sub(r'```{3,}', '```', message)
        
        # Fix any double backticks to be single backticks for inline code
        # but only if they're not part of a code block
        in_code_block = False
        result_lines = []
        
        for line in message.split('\n'):
            if line.count('```') % 2 == 1:  # Line has odd number of triple backticks
                in_code_block = not in_code_block
                result_lines.append(line)
            else:
                if not in_code_block:
                    # Only replace double backticks outside code blocks
                    line = line.replace('``', '`')
                result_lines.append(line)
        
        message = '\n'.join(result_lines)
        
        # Handles asterisks for italics by replacing them with underscores
        # This is complex because we need to avoid replacing bold formatting
        # For simplicity, we'll use regex to find italic patterns not inside code blocks
        def replace_italic(match):
            return f"_{match.group(1)}_"
        
        # Find single asterisks around text (but not if they are double asterisks for bold)
        # and replace with underscores, but only outside code blocks
        in_code_block = False
        result_lines = []
        
        for line in message.split('\n'):
            if '```' in line:
                code_blocks = line.split('```')
                for i, block in enumerate(code_blocks):
                    if i % 2 == 0 and i < len(code_blocks) - 1:  # Outside code block and not the last segment
                        # Replace italics with underscores outside code blocks
                        block = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', replace_italic, block)
                    code_blocks[i] = block
                line = '```'.join(code_blocks)
            else:  # Line doesn't contain code blocks
                line = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', replace_italic, line)
            
            result_lines.append(line)
            
        message = '\n'.join(result_lines)
        
        # Fix any remaining markdown issues
        
        # Ensure *** (bold+italic) becomes *_text_*
        message = re.sub(r'\*\*\*([^*]+)\*\*\*', r'*_\1_*', message)
        
        # Final check for any unwanted triple asterisks
        message = message.replace("***", "*_")
        message = message.replace("***", "_*")
        
        return message

    def response(self, order: str, execution_results: Dict[str, Any]) -> str:
        """Generate a response based on execution results
        
        Parameters:
            order (str): The original user request
            execution_results (Dict[str, Any]): Results from function executions
        """
        logger.debug(f"[Commander] response generation called with order: {order}")

        start_time = time.time()
        
        # Check for Linear functions that require approval
        linear_approval_functions = ["createIssue", "updateIssue"]
        needs_approval = False
        
        for func_name, result in execution_results.items():
            if func_name in linear_approval_functions:
                # Mark this function as requiring approval
                logger.info(f"[Commander] Identified {func_name} that requires approval")
                execution_results[func_name] = {
                    "function": func_name,
                    "parameters": result.get("parameters", {}),
                    "result": None,
                    "requires_modal_approval": True,
                    "description": f"Waiting for approval to {func_name}"
                }
                needs_approval = True
        
        # Prepare prompt variables
        prompt_vars = {
            "order": order,
            "execution_results": json.dumps(execution_results, indent=2),
            "needs_approval": needs_approval
        }
        
        # Format the prompt using template
        formatted = self.format_prompt("commander.response", prompt_vars)
        
        # Create a new OpenaiClient with gpt-4.1-2025-04-14 model specifically for this response
        from llm.openai_client import OpenaiClient
        response_client = OpenaiClient(os.getenv("OPENAI_API_KEY"), model="gpt-4.1-2025-04-14")
        
        # Log the final prompt sent to the API
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Commander] Sending prompt to API for response using gpt-4.1-2025-04-14:\nSystem: {formatted['system']}\nUser: {formatted['user']}")
        
        try:
            # Always use the response function with the new client
            response = response_client.response(
                prompt=formatted["user"], 
                system_prompt=formatted["system"]
            )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Track API call
            api_call_tracker.track_call("Commander", "response", execution_time)
            
            # Log the response from the API
            if logger.isEnabledFor(logging.DEBUG):
                # For the final response, we don't need to parse it as JSON since it's a plain string
                logger.debug(f"[Commander] API response for final response: {response}")
                
            # Format the response for Slack before returning
            formatted_response = self.format_slack_message(response)
            
            # If needs approval, add mention to end of response if not already there
            if needs_approval and "approval" not in formatted_response.lower():
                formatted_response += "\n\n_Note: I've prepared a Linear action that requires your approval. Please review the form when it appears._"
            
            return formatted_response
        except CancellationError:
            logger.info("[Commander] response generation was cancelled")
            return "I had to stop processing your request because you asked me to stop."
        except Exception as e:
            logger.error(f"Error generating final response: {e}")
            return "I'm having trouble generating a response. Please try again later."
        
    def analyze_image(self, image_data: str) -> Optional[str]:
        """
        Analyze an image and extract descriptive context from it.
        
        Parameters:
            image_data (str): Base64-encoded image data
            
        Returns:
            Optional[str]: Descriptive analysis of the image or None if analysis failed
        """
        if not image_data:
            logger.warning("[Commander] No image data provided to analyze_image")
            return None
            
        logger.info("[Commander] Analyzing image data")
        start_time = time.time()
        
        # Format the image extraction prompt
        extraction_formatted = self.format_prompt("commander.extract_image_data", {})
        
        try:
            # Basic validation of image data
            if not isinstance(image_data, str):
                logger.error(f"[Commander] Image data is not a string type: {type(image_data)}")
                raise ValueError("Image data must be a base64-encoded string")
            
            if len(image_data) < 100:
                logger.warning(f"[Commander] Image data seems too short: {len(image_data)} bytes")
            
            # Call the vision model to analyze the image
            # Make sure to use a model that supports vision
            vision_model = "gpt-4.1-nano-2025-04-14"  # gpt-4.1 has vision capabilities
            logger.info(f"[Commander] Using vision model: {vision_model}")
            image_module_client = OpenaiClient(os.getenv("OPENAI_API_KEY"), model=vision_model)
            
            # Check OpenAI client configuration
            logger.info(f"[Commander] OpenAI client created with model: {image_module_client.model}")
            
            image_analysis = image_module_client.response(
                prompt="Please analyze this image in detail, extracting all text and providing a comprehensive description.",
                system_prompt=extraction_formatted["system"],
                image_data=image_data
            )
            
            if image_analysis:
                logger.info("[Commander] Successfully extracted image context")
                
                # Track this API call
                api_call_tracker.track_call("Commander", "analyze_image", time.time() - start_time)
                
                return image_analysis
            else:
                logger.warning("[Commander] Image analysis returned empty result")
                return None
                
        except Exception as e:
            logger.error(f"[Commander] Error analyzing image: {str(e)}")
            return None

class Captain:
    def __init__(self, model: str, prompts: Dict = None, context_id: str = None):
        self.model = model
        self.client = CancellableOpenAIClient(os.getenv("OPENAI_API_KEY"), model=model, context_id=context_id)
        self.prompts = prompts or {}
        logger.info(f"Captain initialized with model {model}")
        if logger.isEnabledFor(logging.DEBUG) and prompts:
            logger.debug(f"Captain loaded with {len(prompts)} prompt categories")
            
    def set_context_id(self, context_id: str):
        """Set the context ID for cancellation checks."""
        self.client.set_context_id(context_id)

    def format_prompt(self, prompt_name: str, prompt_vars: Dict[str, Any]) -> Dict[str, str]:
        """Format a prompt template with provided variables."""
        if not self.prompts:
            logger.warning("No prompts loaded for Captain, using direct prompt")
            return {"system": None, "user": json.dumps(prompt_vars)}
            
        # Parse prompt name (e.g., "linear.captain.plan" -> ["linear", "captain", "plan"])
        parts = prompt_name.split(".")
        if len(parts) < 3:
            logger.warning(f"Invalid prompt name format: {prompt_name}")
            return {"system": None, "user": json.dumps(prompt_vars)}
            
        platform, agent_type, action = parts[0], parts[1], parts[2]
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Captain] Format prompt called with name: {prompt_name}")
            logger.debug(f"[Captain] Parsed parts: platform={platform}, agent_type={agent_type}, action={action}")
            logger.debug(f"[Captain] Available prompts categories: {list(self.prompts.keys())}")
        
        # Look for the prompt using a single, consistent structure
        if platform in self.prompts and agent_type in self.prompts[platform] and action in self.prompts[platform][agent_type]:
            template_data = self.prompts[platform][agent_type][action]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Captain] Found prompt template using platform.agent.action structure")
        else:
            logger.warning(f"Prompt not found: {prompt_name}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Captain] Available prompts structure: {json.dumps({k: list(v.keys()) for k, v in self.prompts.items() if isinstance(v, dict)})}")
            return {"system": None, "user": json.dumps(prompt_vars)}
        
        system_template = template_data.get("system_template", "")
        user_template = template_data.get("user_template", "")
    
        
        # Format templates with variables
        formatted_user = user_template
        for var_name, var_value in prompt_vars.items():
            placeholder = f"{{{{{var_name}}}}}"
            if isinstance(var_value, (dict, list)):
                var_value = json.dumps(var_value, indent=2)
            formatted_user = formatted_user.replace(placeholder, str(var_value))
        
        
        return {
            "system": system_template,
            "user": formatted_user
        }

    def plan(self, order: str, platforms: List[str], previous_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a plan based on available tools and user order"""
        logger.debug(f"[Captain] plan called with order: {order}")
        
        start_time = time.time()
        
        # Get tools for the specific platforms
        tools_text = self._get_tools_for_platforms(platforms)
        
        # Determine the prompt name based on the first platform
        platform = platforms[0] if platforms else "linear"
        prompt_name = f"{platform}.captain.plan"
        logger.debug(f"[Captain] Using prompt name: {prompt_name}")
        
        # Prepare prompt variables with the correct platform-specific variable name
        prompt_vars = {
            "order": order
        }
        
        # Include previous results if available
        if previous_results:
            prompt_vars["previous_results"] = json.dumps(previous_results, indent=2)
        
        # Add tools using the appropriate variable name based on platform
        if platform == "linear":
            prompt_vars["linear_tools"] = tools_text
        elif platform == "slack":
            prompt_vars["slack_tools"] = tools_text
        elif platform == "github":
            prompt_vars["github_tools"] = tools_text
        elif platform == "url":
            prompt_vars["url_tools"] = tools_text
        else:
            # Fallback to a generic name
            prompt_vars["tools"] = tools_text
        
        # Format the prompt using template
        formatted = self.format_prompt(prompt_name, prompt_vars)
        
        # For response_reasoning, combine system_template and user_template
        combined_prompt = formatted["system"] + "\n\n" + formatted["user"]
        
        # Log the final prompt sent to the API
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Captain] Sending prompt to API for plan:\n{combined_prompt}")
        
        try:
            if self.model.startswith("o"):
                # Get response from LLM
                response = self.client.response_reasoning(
                    prompt=combined_prompt,
                    reasoning_effort="high"
                )
            else:
                # Get response from LLM
                response = self.client.response(
                    prompt=combined_prompt,
                    system_prompt=formatted["system"]
                )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Track API call
            api_call_tracker.track_call("Captain", "plan", execution_time)
            
            # Log the response from the API
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Captain] API response for plan: {json.dumps(json.loads(response), indent=2, ensure_ascii=False)}")
            
            # Parse JSON response
            try:
                if response.startswith("```json"):
                    response = response[len("```json"):].strip()
                if response.endswith("```"):
                    response = response[:-len("```")]
                result = json.loads(response)
                
                # Ensure the plan has the expected structure
                if "function_levels" not in result:
                    logger.warning("Plan response missing 'function_levels', adding empty levels")
                    result["function_levels"] = []
                    
                # For backward compatibility, convert old format to new if needed
                if "functions" in result and "function_levels" not in result:
                    logger.warning("Converting legacy functions format to function_levels")
                    function_levels = []
                    
                    # Add ready_to_execute functions as level 1
                    if "ready_to_execute" in result["functions"]:
                        ready = result["functions"]["ready_to_execute"]
                        if ready:
                            function_levels.append(ready)
                    
                    # Add not_ready_to_execute functions as level 2+
                    if "not_ready_to_execute" in result["functions"]:
                        not_ready = result["functions"]["not_ready_to_execute"]
                        if not_ready:
                            # Extract just the function names if they're in dict format
                            not_ready_names = []
                            for item in not_ready:
                                if isinstance(item, dict) and "name" in item:
                                    not_ready_names.append(item["name"])
                                elif isinstance(item, str):
                                    not_ready_names.append(item)
                            
                            if not_ready_names:
                                function_levels.append(not_ready_names)
                    
                    # Replace the functions with function_levels
                    result["function_levels"] = function_levels
                    del result["functions"]
                
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from Captain plan: {e}")
                # Fallback to empty plan
                return {
                    "plan_description": "Failed to create a plan",
                    "function_levels": []
                }
        except CancellationError:
            logger.info("[Captain] plan was cancelled")
            return {
                "plan_description": "Planning was interrupted",
                "function_levels": []
            }
        except Exception as e:
            logger.error(f"Failed to get Captain plan: {e}")
            # Fallback to empty plan
            return {
                "plan_description": "Failed to create a plan",
                "function_levels": []
            }
    
    def evaluate(self, order: str, plan: Dict[str, Any], execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate execution results and determine if plan should continue"""
        logger.debug(f"[Captain] evaluate called with order: {order}")
        
        start_time = time.time()
        
        # Determine the platform from the plan
        # Extract the platform from the first executed function, or default to 'linear'
        platform = 'linear'  # Default platform
        
        # Try to determine platform from function_levels
        if plan and 'function_levels' in plan and plan['function_levels']:
            for level in plan['function_levels']:
                if level:  # If there's at least one function in this level
                    first_function = level[0]
                    if first_function in LINEAR_SCHEMAS:
                        platform = 'linear'
                        break
                    elif first_function in SLACK_SCHEMAS:
                        platform = 'slack'
                        break
                    # Add more platform detections as needed
        
        # Use the same platform-based prompt naming as in the plan method
        prompt_name = f"{platform}.captain.evaluate"
        logger.debug(f"[Captain] Using prompt name: {prompt_name}")
        
        # Prepare prompt variables - these should remain as their original types
        # to be properly JSON serialized in format_prompt
        prompt_vars = {
            "order": order,
            "plan": plan,
            "execution_results": execution_results
        }
        
        # Format the prompt using template
        formatted = self.format_prompt(prompt_name, prompt_vars)
        
        # For response_reasoning, combine system_template and user_template
        combined_prompt = formatted["system"] + "\n\n" + formatted["user"]
        
        # Log the final prompt sent to the API
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Captain] Sending prompt to API for evaluate:\n{combined_prompt}")
        
        try:
            # Get response from LLM
            if self.model.startswith("o"):
                response = self.client.response_reasoning(
                    prompt=combined_prompt,
                    reasoning_effort="high"
                )
            else:
                response = self.client.response(
                    prompt=combined_prompt,
                    system_prompt=formatted["system"]
                )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Track API call
            api_call_tracker.track_call("Captain", "evaluate", execution_time)
            
            # Log the response from the API
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Captain] API response for evaluate: {json.dumps(json.loads(response), indent=2, ensure_ascii=False)}")
            
            # Parse JSON response
            try:
                result = json.loads(response)
                
                # Ensure all expected fields are present
                if "change_plan" not in result:
                    result["change_plan"] = False
                if "error_description" not in result:
                    result["error_description"] = None
                if "response_ready" not in result:
                    result["response_ready"] = False
                    
                # For backward compatibility, handle execution_complete if present but not needed
                if "execution_complete" in result:
                    logger.warning("Ignoring deprecated 'execution_complete' field in evaluate response")
                    del result["execution_complete"]
                    
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from Captain evaluate: {e}")
                # Fallback to continue with current plan
                return {
                    "change_plan": False,
                    "error_description": None,
                    "response_ready": False
                }
        except CancellationError:
            logger.info("[Captain] evaluate was cancelled")
            return {
                "change_plan": False,
                "error_description": "Evaluation was interrupted",
                "response_ready": True
            }
        except Exception as e:
            logger.error(f"Failed to get Captain evaluate: {e}")
            # Fallback to continue with current plan
            return {
                "change_plan": False,
                "error_description": None,
                "response_ready": False
            }
    
    def _get_tools_for_platforms(self, platforms: List[str]) -> str:
        """Get and format tools description for specified platforms"""
        start_time = time.time()
        
        # Ensure platforms is always a list
        if isinstance(platforms, str):
            platforms = [platforms]
            logger.debug(f"[Captain] Converted platform string '{platforms[0]}' to list")
            
        # Import tools_desc.yaml content
        try:
            with open('tools/tools_desc.yaml', 'r') as file:
                tools_desc = yaml.safe_load(file)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[Captain] Loaded tools_desc.yaml with {len(tools_desc)} categories")
        except Exception as e:
            logger.error(f"Failed to load tools_desc.yaml: {e}")
            return "No tools available"
        
        # Collect tools for the specified platforms
        tools_text = ""
        for platform in platforms:
            platform_key = f"{platform}_tools"
            if platform_key in tools_desc:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[Captain] Processing tools for platform: {platform}")
                
                # Process each category of tools
                for category_data in tools_desc[platform_key]:
                    category = category_data.get('category', 'Uncategorized Tools')
                    tools = category_data.get('tools', [])
                    
                    # Add category header
                    tools_text += f"CATEGORY: {category}\n"
                    tools_text += "=" * (len(category) + 10) + "\n\n"
                    
                    # Process each tool in this category
                    for tool in tools:
                        # Get the name and description with fallbacks
                        tool_name = tool.get('name', 'unnamed_tool')
                        tool_desc = tool.get('description', 'No description provided')
                        
                        tools_text += f"Tool: {tool_name}\n"
                        tools_text += f"Description: {tool_desc}\n"
                        tools_text += "Inputs:\n"
                        
                        # Handle inputs based on type
                        inputs = tool.get('inputs')
                        if inputs is None:
                            tools_text += "  (No inputs defined)\n"
                            logger.warning(f"Tool {tool_name} has no 'inputs' field")
                        elif isinstance(inputs, dict):
                            # Inputs is a dictionary, iterate through key-value pairs
                            for input_name, input_desc in inputs.items():
                                required = input_name in tool.get('required_inputs', [])
                                tools_text += f"  - {input_name}: {input_desc} {'(Required)' if required else '(Optional)'}\n"
                        elif isinstance(inputs, str):
                            # Inputs is a string, just display it directly
                            tools_text += f"  {inputs}\n"
                            logger.warning(f"Tool {tool_name} has 'inputs' as a string: '{inputs}'")
                        else:
                            # Unknown inputs type
                            tools_text += f"  (Invalid inputs format: {type(inputs).__name__})\n"
                            logger.warning(f"Tool {tool_name} has 'inputs' in invalid format: {type(inputs).__name__}")
                        
                        # Get outputs safely
                        outputs = tool.get('outputs', 'No outputs defined')
                        tools_text += f"Outputs: {outputs}\n\n"
                    
                    # Add separator between categories
                    tools_text += "-" * 50 + "\n\n"
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Track API call
        api_call_tracker.track_call("Captain", "_get_tools_for_platforms", execution_time)
        
        return tools_text or "No tools available for the specified platforms"

class Soldier:
    def __init__(self, model: str, prompts: Dict = None, context_id: str = None):
        self.model = model
        self.client = CancellableOpenAIClient(os.getenv("OPENAI_API_KEY"), model=model, context_id=context_id)
        self.prompts = prompts or {}
        logger.info(f"Soldier initialized with model {model}")
        if logger.isEnabledFor(logging.DEBUG) and prompts:
            logger.debug(f"Soldier loaded with {len(prompts)} prompt categories")
            
    def set_context_id(self, context_id: str):
        """Set the context ID for cancellation checks."""
        self.client.set_context_id(context_id)
    
    def format_prompt(self, prompt_name: str, prompt_vars: Dict[str, Any]) -> Dict[str, str]:
        """Format a prompt template with provided variables."""
        if not self.prompts:
            logger.warning("No prompts loaded for Soldier, using direct prompt")
            return {"system": None, "user": json.dumps(prompt_vars)}
            
        # Parse prompt name (e.g., "linear.soldier.execute" -> ["linear", "soldier", "execute"])
        parts = prompt_name.split(".")
        if len(parts) < 3:
            logger.warning(f"Invalid prompt name format: {prompt_name}")
            return {"system": None, "user": json.dumps(prompt_vars)}
            
        platform, agent_type, action = parts[0], parts[1], parts[2]
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Soldier] Format prompt called with name: {prompt_name}")
            logger.debug(f"[Soldier] Parsed parts: platform={platform}, agent_type={agent_type}, action={action}")
        
        # Look for the prompt using a single, consistent structure
        if platform in self.prompts and agent_type in self.prompts[platform] and action in self.prompts[platform][agent_type]:
            template_data = self.prompts[platform][agent_type][action]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Soldier] Found prompt template using platform.agent.action structure")
        else:
            logger.warning(f"Prompt not found: {prompt_name}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Soldier] Available prompts structure: {json.dumps({k: list(v.keys()) for k, v in self.prompts.items() if isinstance(v, dict)})}")
            return {"system": None, "user": json.dumps(prompt_vars)}
            
        system_template = template_data.get("system_template", "")
        user_template = template_data.get("user_template", "")
                
        # Format templates with variables
        formatted_user = user_template
        for var_name, var_value in prompt_vars.items():
            placeholder = f"{{{{{var_name}}}}}"
            if isinstance(var_value, (dict, list)):
                var_value = json.dumps(var_value, indent=2)
            formatted_user = formatted_user.replace(placeholder, str(var_value))
        
        return {
            "system": system_template,
            "user": formatted_user
        }
    
    async def execute(self, plan: Dict[str, Any], function_name: str, 
                     user_query: str, previous_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a tool with the appropriate parameters"""
        logger.debug(f"[Soldier] execute called for function: {function_name}")
        
        start_time = time.time()
        
        # Get the tool schema
        tool_schema = None
        platform = "unknown"
        
        if function_name in LINEAR_SCHEMAS:
            tool_schema = LINEAR_SCHEMAS[function_name]
            platform = "linear"
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Soldier] {function_name} is a Linear tool")
        elif function_name in SLACK_SCHEMAS:
            tool_schema = SLACK_SCHEMAS[function_name]
            platform = "slack"
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Soldier] {function_name} is a Slack tool")
        elif function_name in SEMANTIC_SEARCH_SCHEMAS:
            tool_schema = SEMANTIC_SEARCH_SCHEMAS[function_name]
            platform = "linear"  # Use linear prompt for semantic search
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Soldier] {function_name} is a Semantic Search tool (using linear prompts)")
        
        if not tool_schema:
            logger.error(f"No schema found for function: {function_name}")
            return {
                "error": f"Unknown function: {function_name}",
                "result": None
            }
        
        # Prepare prompt variables - only include what's used in the templates
        prompt_vars = {
            "plan": json.dumps(plan, indent=2),
            "previous_results": json.dumps(previous_results or {}, indent=2)
        }
        
        # Format the prompt using template
        prompt_name = f"{platform}.soldier.execute"
        formatted = self.format_prompt(prompt_name, prompt_vars)
        
        # For use_tool, combine system_template and user_template
        combined_prompt = formatted["system"] + "\n\n" + formatted["user"]
        
        # Log the final prompt for tool execution
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Soldier] Sending prompt to API for tool execution {function_name}:\n{combined_prompt}")
        
        try:
            if self.model.startswith("o"):
                # Get parameter values from LLM
                tool_call = self.client.use_tool(
                    prompt=combined_prompt, 
                    tools=[tool_schema],
                    reasoning_effort="high"
                )
            else:
                # Get parameter values from LLM
                tool_call = self.client.use_tool(
                    prompt=formatted["user"], 
                    system_prompt=formatted["system"],
                    tools=[tool_schema]
                )
            
            # Log the tool call response
            if logger.isEnabledFor(logging.DEBUG) and tool_call:
                logger.debug(f"[Soldier] API response for tool call {function_name}: {json.dumps(tool_call, indent=2, ensure_ascii=False)}")
            
            if not tool_call:
                # Calculate execution time for error case
                execution_time = time.time() - start_time
                
                # Track failed API call
                api_call_tracker.track_call("Soldier", f"execute_{function_name}_failed", execution_time)
                
                logger.error(f"Failed to get tool call for {function_name}")
                return {
                    "function": function_name,
                    "error": "Failed to determine parameters",
                    "result": None
                }
            
            # Import tool implementations here to avoid circular imports
            from tools.tools_declaration import linear_tools, slack_tools
            
            # Map function name to implementation
            tool_implementations = {
                # Linear tools
                "filterIssues": linear_tools.filterIssues,
                "createIssue": linear_tools.createIssue,
                "updateIssue": linear_tools.updateIssue,
                "filterComments": linear_tools.filterComments,
                "filterAttachments": linear_tools.filterAttachments,
                "getAllUsers": linear_tools.getAllUsers,
                "getAllTeams": linear_tools.getAllTeams,
                "getAllProjects": linear_tools.getAllProjects,
                "getAllCycles": linear_tools.getAllCycles,
                "getAllLabels": linear_tools.getAllLabels,
                "getAllStates": linear_tools.getAllStates,
                "filterProjects": linear_tools.filterProjects,
                "filterCycles": linear_tools.filterCycles,
                "createComment": linear_tools.createComment,
                "getCurrentUser": linear_tools.getCurrentUser,
                "semantic_search_linear": linear_tools.semantic_search_linear,
                
                # Slack tools
                "search_channel_history": slack_tools.search_channel_history,
                "get_users": slack_tools.get_users,
                "get_current_user": slack_tools.get_current_user
            }
            
            # Get the tool implementation
            tool_func = tool_implementations.get(function_name)
            if not tool_func:
                # Calculate execution time for error case
                execution_time = time.time() - start_time
                
                # Track failed API call
                api_call_tracker.track_call("Soldier", f"execute_{function_name}_not_implemented", execution_time)
                
                logger.error(f"No implementation found for function: {function_name}")
                return {
                    "function": function_name,
                    "error": f"Function {function_name} not implemented",
                    "result": None
                }
            
            # Execute the tool
            try:
                params = tool_call.get("parameters", {})
                logger.info(f"Executing {function_name} with parameters: {params}")
                
                # Check if this function requires modal approval
                linear_approval_functions = ["createIssue", "updateIssue"]
                if function_name in linear_approval_functions:
                    logger.info(f"Function {function_name} requires modal approval - deferring execution")
                    return {
                        "function": function_name,
                        "parameters": params,
                        "result": None,
                        "requires_modal_approval": True,
                        "description": f"Waiting for approval to {function_name}"
                    }
                
                # Handle async functions
                if function_name == "search_channel_history":
                    result = await tool_func(**params)
                else:
                    result = tool_func(**params)
                
                # Calculate execution time for successful case
                execution_end_time = time.time()
                execution_time = execution_end_time - start_time
                
                # Track successful API call
                api_call_tracker.track_call("Soldier", f"execute_{function_name}", execution_time)
                
                # Log the execution result (summary)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[Soldier] Execution result for {function_name}: {result}")
                
                return {
                    "function": function_name,
                    "parameters": params,
                    "result": result
                }
            except Exception as e:
                # Calculate execution time for error case
                execution_time = time.time() - start_time
                
                # Track failed API call
                api_call_tracker.track_call("Soldier", f"execute_{function_name}_error", execution_time)
                
                logger.error(f"Error executing {function_name}: {str(e)}")
                return {
                    "function": function_name,
                    "parameters": tool_call.get("parameters", {}),
                    "error": str(e),
                    "result": None
                }
        except CancellationError:
            logger.info(f"[Soldier] execute for {function_name} was cancelled")
            return {
                "function": function_name,
                "error": "Operation was cancelled",
                "result": None
            }
        except Exception as e:
            logger.error(f"Error with tool call for {function_name}: {str(e)}")
            # Calculate execution time for error case
            execution_time = time.time() - start_time
            
            # Track failed API call
            api_call_tracker.track_call("Soldier", f"execute_{function_name}_error", execution_time)
            
            logger.error(f"Error executing {function_name}: {str(e)}")
            return {
                "function": function_name,
                "error": str(e),
                "result": None
            }

# Function to get the API call tracker
def get_api_call_tracker():
    """Get the global API call tracker instance."""
    return api_call_tracker

# Function to reset the API call tracker
def reset_api_call_tracker():
    """Reset the global API call tracker instance."""
    api_call_tracker.reset()

# Function to log the API call report
def log_api_call_report(log_level=logging.DEBUG):
    """Log the API call report at the specified level."""
    api_call_tracker.log_report(log_level)

