#!/usr/bin/env python3
"""
Simple script to test if the tools_desc.yaml file can be loaded correctly.
"""

import yaml
import sys
import os

def test_yaml_loading(file_path):
    """
    Attempt to load a YAML file and print its contents.
    
    Args:
        file_path: Path to the YAML file
    
    Returns:
        True if the file was loaded successfully, False otherwise
    """
    try:
        print(f"Attempting to load YAML file: {file_path}")
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            
        print(f"✅ Success! File loaded correctly.")
        print(f"File structure (first level keys): {list(data.keys())}")
        
        # Count tools
        tool_count = 0
        for category in data.get('linear_tools', []):
            for tool in category.get('tools', []):
                tool_count += 1
        
        print(f"Found {tool_count} tools in the 'linear_tools' section.")
        
        return True
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML: {e}")
        if hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            print(f"Error position: line {mark.line + 1}, column {mark.column + 1}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    # Try to find the file in the current directory and in the tools directory
    possible_paths = [
        "tools_desc.yaml",
        "tools/tools_desc.yaml",
        "../tools/tools_desc.yaml",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools_desc.yaml"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "tools_desc.yaml")
    ]
    
    success = False
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found file at: {path}")
            if test_yaml_loading(path):
                success = True
                break
    
    if not success:
        print("Could not find or load the YAML file in any of the expected locations.")
        print("Checked paths:")
        for path in possible_paths:
            print(f"  - {path}")
        sys.exit(1)
    
    sys.exit(0) 