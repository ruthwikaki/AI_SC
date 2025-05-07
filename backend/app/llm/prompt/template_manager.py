# app/llm/prompt/template_manager.py

from typing import Dict, List, Any, Optional, Union
import os
import json
import yaml
from pathlib import Path
import re
from datetime import datetime

from app.utils.logger import get_logger
from app.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Templates cache
_templates_cache: Dict[str, Dict[str, Any]] = {}
_last_reload: datetime = datetime.min

class TemplateManager:
    """
    Manager for prompt templates.
    
    This class provides methods to load, manage, and render prompt templates
    for different use cases in the application.
    """
    
    @staticmethod
    def get_template(template_name: str, reload: bool = False) -> Dict[str, Any]:
        """
        Get a prompt template by name.
        
        Args:
            template_name: Name of the template
            reload: Whether to force reload from disk
            
        Returns:
            Template data dictionary
        """
        global _templates_cache, _last_reload
        
        # Check if we need to reload templates
        templates_dir = settings.templates_dir
        if templates_dir is None:
            templates_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "prompt",
                "templates"
            )
        
        # Reload templates if requested or cache is empty
        if reload or not _templates_cache or (datetime.now() - _last_reload).total_seconds() > 300:  # 5 minutes
            try:
                TemplateManager._load_templates(templates_dir)
                _last_reload = datetime.now()
            except Exception as e:
                logger.error(f"Error loading templates: {str(e)}")
                # If reload fails and cache is empty, raise error
                if not _templates_cache:
                    raise ValueError(f"Failed to load templates: {str(e)}")
        
        # Get template from cache
        if template_name not in _templates_cache:
            # Try to find a partial match (e.g., "query" matches "query_translation")
            matches = [t for t in _templates_cache.keys() if template_name in t]
            if matches:
                template_name = matches[0]
            else:
                raise ValueError(f"Template not found: {template_name}")
        
        return _templates_cache[template_name]
    
    @staticmethod
    def render_template(template: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Render a template with context variables.
        
        Args:
            template: Template dictionary
            context: Context variables
            
        Returns:
            Rendered template
        """
        # Get the template content
        content = template.get("content", "")
        
        # Render system context if provided
        system_context = template.get("system_context", "")
        if system_context:
            system_context = TemplateManager._render_text(system_context, context)
        
        # Render user message
        user_message = template.get("user_message", "")
        if not user_message:
            user_message = content  # Fall back to content if user_message not specified
        
        user_message = TemplateManager._render_text(user_message, context)
        
        # Format based on template type
        template_type = template.get("type", "standard")
        
        if template_type == "chat":
            # Chat format with system and user messages
            result = ""
            if system_context:
                result += f"System: {system_context}\n\n"
            result += f"User: {user_message}"
            return result
        
        elif template_type == "json":
            # JSON instruction format
            result = {}
            if system_context:
                result["system"] = system_context
            result["user"] = user_message
            return json.dumps(result)
        
        else:
            # Standard format
            result = ""
            if system_context:
                result += f"{system_context}\n\n"
            result += user_message
            return result
    
    @staticmethod
    def _render_text(text: str, context: Dict[str, Any]) -> str:
        """
        Render text with context variables.
        
        Args:
            text: Text with placeholders
            context: Context variables
            
        Returns:
            Rendered text
        """
        try:
            # Replace variables in text
            for key, value in context.items():
                # Handle nested keys with dot notation (e.g., schema.tables)
                if "." in key:
                    continue  # Skip complex keys, handled in format()
                
                # Replace {key} with value
                placeholder = "{" + key + "}"
                if placeholder in text:
                    # Handle different value types
                    if isinstance(value, (dict, list)):
                        # Format dicts and lists as YAML for readability
                        formatted_value = yaml.dump(value, default_flow_style=False)
                        # Indent each line to maintain formatting
                        formatted_value = "\n".join(
                            ["    " + line for line in formatted_value.split("\n")]
                        )
                        text = text.replace(placeholder, formatted_value)
                    else:
                        text = text.replace(placeholder, str(value))
            
            # Use string formatting for any remaining placeholders
            # This handles nested keys and more complex formatting
            try:
                text = text.format(**context)
            except KeyError as e:
                logger.warning(f"Missing key in template context: {e}")
            
            return text
            
        except Exception as e:
            logger.error(f"Error rendering template: {str(e)}")
            return text
    
    @staticmethod
    def _load_templates(templates_dir: str) -> None:
        """
        Load templates from directories and files.
        
        Args:
            templates_dir: Directory containing templates
        """
        global _templates_cache
        
        templates = {}
        
        # Check if directory exists
        if not os.path.exists(templates_dir):
            logger.warning(f"Templates directory not found: {templates_dir}")
            return
        
        # Get all JSON and YAML files in directory and subdirectories
        for root, _, files in os.walk(templates_dir):
            for file in files:
                if file.endswith((".json", ".yaml", ".yml", ".py")):
                    file_path = os.path.join(root, file)
                    
                    try:
                        template_name = os.path.splitext(file)[0]
                        # Add parent directory name if in subdirectory
                        rel_path = os.path.relpath(root, templates_dir)
                        if rel_path != ".":
                            template_name = f"{rel_path.replace(os.sep, '_')}_{template_name}"
                        
                        # Load template from file
                        if file.endswith(".json"):
                            with open(file_path, "r") as f:
                                template = json.load(f)
                        elif file.endswith((".yaml", ".yml")):
                            with open(file_path, "r") as f:
                                template = yaml.safe_load(f)
                        elif file.endswith(".py"):
                            # Extract templates from Python module
                            templates_from_py = TemplateManager._load_from_py(file_path)
                            for py_name, py_template in templates_from_py.items():
                                templates[py_name] = py_template
                            continue
                        
                        templates[template_name] = template
                        logger.debug(f"Loaded template: {template_name}")
                        
                    except Exception as e:
                        logger.error(f"Error loading template from {file_path}: {str(e)}")
        
        # Update cache
        _templates_cache = templates
        
    @staticmethod
    def _load_from_py(file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load templates from Python module.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Dictionary of templates
        """
        templates = {}
        
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            # Extract template dictionaries from Python file
            # Look for patterns like TEMPLATE_NAME = {...}
            template_pattern = r'([A-Z_]+)\s*=\s*({[^}]+})'
            matches = re.findall(template_pattern, content, re.DOTALL)
            
            for name, template_str in matches:
                try:
                    # Convert string representation to dict
                    template = eval(template_str)
                    templates[name.lower()] = template
                except Exception as e:
                    logger.error(f"Error parsing template {name} in {file_path}: {str(e)}")
            
            # Also look for template dictionaries as module variables
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            templates[module_name] = {"templates": templates}
            
            return templates
            
        except Exception as e:
            logger.error(f"Error loading templates from {file_path}: {str(e)}")
            return {}