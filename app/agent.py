import os
import json
import logging
import dotenv
import yaml
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
        
        # Prepare prompt variables
        prompt_vars = {
            "order": order,
            "execution_results": json.dumps(execution_results, indent=2)
        }
        
        # Format the prompt using template
        formatted = self.format_prompt("commander.response", prompt_vars)
        
        # Get response from LLM based on model type
        if self.model.startswith("gpt"):
            # Log the final prompt sent to the API
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Commander] Sending prompt to API for response:\nSystem: {formatted['system']}\nUser: {formatted['user']}")
            
            response = self.client.response(
                prompt=formatted["user"], 
                system_prompt=formatted["system"]
            )
        elif self.model.startswith("o"):
            # For response_reasoning, combine system_template and user_template
            combined_prompt = formatted["system"] + "\n\n" + formatted["user"]
            
            # Log the final prompt sent to the API
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Commander] Sending prompt to API for response:\n{combined_prompt}")
            
            response = self.client.response_reasoning(
                prompt=combined_prompt,
                reasoning_effort="low"
            )
        else:
            # Default to response method
            # Log the final prompt sent to the API
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Commander] Sending prompt to API for response:\nSystem: {formatted['system']}\nUser: {formatted['user']}")
            
            response = self.client.response(
                prompt=formatted["user"], 
                system_prompt=formatted["system"]
            )
        
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

    def plan(self, order: str, platforms: List[str]) -> Dict[str, Any]:
        """Create a plan based on available tools and user order"""
        
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
        
        # Log the response from the API
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Captain] API response for plan: {json.dumps(json.loads(response), indent=2, ensure_ascii=False)}")
        
        # Parse JSON response
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Captain plan: {e}")
            # Fallback to empty plan
            return {
                "plan_description": "Failed to create a plan",
                "functions": {
                    "ready_to_execute": [],
                    "not_ready_to_execute": []
                }
            }
    
    def evaluate(self, order: str, plan: Dict[str, Any], execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate execution results and determine if plan should continue"""
        
        # Determine the platform from the plan
        # Extract the platform from the first executed function, or default to 'linear'
        platform = 'linear'  # Default platform
        if plan and 'functions' in plan and 'ready_to_execute' in plan['functions'] and plan['functions']['ready_to_execute']:
            first_function = plan['functions']['ready_to_execute'][0]
            if first_function in LINEAR_SCHEMAS:
                platform = 'linear'
            elif first_function in SLACK_SCHEMAS:
                platform = 'slack'
            # Add more platform detections as needed
        
        # Use the same platform-based prompt naming as in the plan method
        prompt_name = f"{platform}.captain.evaluate"
        logger.debug(f"[Captain] Using prompt name: {prompt_name}")
        
        order = "This is the order for TMAI Agent to execute: " + order
        plan = "This is the current plan for TMAI Agent to follow: " + plan
        execution_results = "This is the results of the tools that have been executed so far: " + execution_results
        # Prepare prompt variables
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
        
        # Log the response from the API
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Captain] API response for evaluate: {json.dumps(json.loads(response), indent=2, ensure_ascii=False)}")
        
        # Parse JSON response
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Captain evaluate: {e}")
            # Fallback to continue with current plan
            return {
                "change_plan": False,
                "error_description": None,
                "execution_complete": False,
                "response_ready": False
            }
    
    def _get_tools_for_platforms(self, platforms: List[str]) -> str:
        """Get and format tools description for specified platforms"""
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
        
        # Prepare prompt variables
        prompt_vars = {
            "plan": json.dumps(plan, indent=2),
            "current_tool": function_name,
            "tool_schema": json.dumps(tool_schema, indent=2),
            "previous_results": json.dumps(previous_results or {}, indent=2),
            "user_query": user_query
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
            
            # Log the execution result (summary)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Soldier] Execution result for {function_name}: {result}")
            
            return {
                "function": function_name,
                "parameters": params,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error executing {function_name}: {str(e)}")
            return {
                "function": function_name,
                "parameters": tool_call.get("parameters", {}),
                "error": str(e),
                "result": None
            }
    
