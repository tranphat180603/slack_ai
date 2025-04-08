import os
import json
import logging
import dotenv
import yaml
import time
from typing import Dict, List, Any, Optional, Union

from llm.openai_client import OpenaiClient
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
    def __init__(self, model: str, prompts: Dict = None):
        self.model = model
        self.client = OpenaiClient(os.getenv("OPENAI_API_KEY"), model=model)
        self.prompts = prompts or {}
        logger.info(f"Commander initialized with model {model}")

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

    def assign_tasks(self, user_query: str, history: Union[List[Dict], str] = None) -> Dict[str, Any]:
        """Assign tasks between different platforms based on user query"""
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
        
        # Prepare prompt variables
        prompt_vars = {
            "user_query": user_query,
            "history": history_text
        }
        
        # Format the prompt using template
        formatted = self.format_prompt("commander.assign_tasks", prompt_vars)
        
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
            # For response_reasoning, combine system_template and user_template
            combined_prompt = formatted["system"] + "\n\n" + formatted["user"]
            
            # Log the final prompt sent to the API
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Commander] Sending prompt to API for assign_tasks:\n{combined_prompt}")
            
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
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Track API call
        api_call_tracker.track_call("Commander", "assign_tasks", execution_time)
        
        # Log the response from the API
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Commander] API response for assign_tasks: {json.dumps(json.loads(response), indent=2, ensure_ascii=False)}")
        
        # Parse JSON response
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Commander response: {e}")
            # Fallback to direct response
            return {
                "order": user_query,
                "platform": ["direct_response"]
            }

    def response(self, order: str, execution_results: Dict[str, Any]) -> str:
        """Generate a response based on execution results"""
        logger.debug(f"[Commander] response generation called with order: {order}")

        start_time = time.time()
        
        # Prepare prompt variables
        prompt_vars = {
            "order": order,
            "execution_results": json.dumps(execution_results, indent=2)
        }
        
        # Format the prompt using template
        formatted = self.format_prompt("commander.response", prompt_vars)
        
        # Create a new OpenaiClient with gpt-4o-mini model specifically for this response
        from llm.openai_client import OpenaiClient
        gpt4o_mini_client = OpenaiClient(os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
        
        # Log the final prompt sent to the API
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Commander] Sending prompt to API for response using gpt-4o-mini:\nSystem: {formatted['system']}\nUser: {formatted['user']}")
        
        # Always use the response function with the new client
        response = gpt4o_mini_client.response(
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
            
        return response

class Captain:
    def __init__(self, model: str, prompts: Dict = None):
        self.model = model
        self.client = OpenaiClient(os.getenv("OPENAI_API_KEY"), model=model)
        self.prompts = prompts or {}
        logger.info(f"Captain initialized with model {model}")
        if logger.isEnabledFor(logging.DEBUG) and prompts:
            logger.debug(f"Captain loaded with {len(prompts)} prompt categories")

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
        
        # Get response from LLM
        response = self.client.response_reasoning(
            prompt=combined_prompt,
            reasoning_effort="medium"
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
        
        # Get response from LLM
        response = self.client.response_reasoning(
            prompt=combined_prompt,
            reasoning_effort="medium"
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
                        tools_text += f"Tool: {tool['name']}\n"
                        tools_text += f"Description: {tool['description']}\n"
                        tools_text += "Inputs:\n"
                        for input_name, input_desc in tool['inputs'].items():
                            required = input_name in tool.get('required_inputs', [])
                            tools_text += f"  - {input_name}: {input_desc} {'(Required)' if required else '(Optional)'}\n"
                        tools_text += f"Outputs: {tool['outputs']}\n\n"
                    
                    # Add separator between categories
                    tools_text += "-" * 50 + "\n\n"
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Track API call
        api_call_tracker.track_call("Captain", "_get_tools_for_platforms", execution_time)
        
        return tools_text or "No tools available for the specified platforms"

class Soldier:
    def __init__(self, model: str, prompts: Dict = None):
        self.model = model
        self.client = OpenaiClient(os.getenv("OPENAI_API_KEY"), model=model)
        self.prompts = prompts or {}
        logger.info(f"Soldier initialized with model {model}")
        if logger.isEnabledFor(logging.DEBUG) and prompts:
            logger.debug(f"Soldier loaded with {len(prompts)} prompt categories")
    
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
        
        # Get parameter values from LLM
        tool_call = self.client.use_tool(
            prompt=combined_prompt, 
            tools=[tool_schema],
            reasoning_effort="low"
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
            
            # Handle async functions
            if function_name == "search_channel_history":
                result = await tool_func(**params)
            else:
                result = tool_func(**params)
            
            # Calculate execution time for successful case
            execution_time = time.time() - start_time
            
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

