"""
Input validation utilities.

This module provides functions for validating user input and data.
"""

import re
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar
from datetime import datetime, date
import uuid
from email_validator import validate_email, EmailNotValidError
from pydantic import BaseModel, ValidationError

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Type variable for generic functions
T = TypeVar('T')

def validate_email_address(email: str) -> bool:
    """
    Validate an email address.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Validate and normalize the email
        valid = validate_email(email)
        return True
    except EmailNotValidError as e:
        logger.debug(f"Invalid email address: {email} - {str(e)}")
        return False

def validate_uuid(uuid_str: str) -> bool:
    """
    Validate a UUID string.
    
    Args:
        uuid_str: UUID string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        uuid_obj = uuid.UUID(uuid_str)
        return str(uuid_obj) == uuid_str
    except (ValueError, AttributeError):
        return False

def validate_date_format(date_str: str, format_str: str = "%Y-%m-%d") -> bool:
    """
    Validate a date string against a format.
    
    Args:
        date_str: Date string to validate
        format_str: Expected date format
        
    Returns:
        True if valid, False otherwise
    """
    try:
        datetime.strptime(date_str, format_str)
        return True
    except ValueError:
        return False

def validate_iso_date(date_str: str) -> bool:
    """
    Validate an ISO format date string (YYYY-MM-DD).
    
    Args:
        date_str: Date string to validate
        
    Returns:
        True if valid, False otherwise
    """
    return validate_date_format(date_str, "%Y-%m-%d")

def validate_iso_datetime(datetime_str: str) -> bool:
    """
    Validate an ISO format datetime string.
    
    Args:
        datetime_str: Datetime string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False

def validate_phone_number(phone: str) -> bool:
    """
    Validate a phone number.
    
    Args:
        phone: Phone number to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Basic phone validation pattern
    pattern = r'^\+?[0-9]{10,15}$'
    return bool(re.match(pattern, phone))

def validate_password_strength(password: str) -> Dict[str, Any]:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": False,
        "length": False,
        "uppercase": False,
        "lowercase": False,
        "digit": False,
        "special": False,
        "errors": []
    }
    
    # Check length (minimum 8 characters)
    if len(password) >= 8:
        results["length"] = True
    else:
        results["errors"].append("Password must be at least 8 characters long")
    
    # Check for uppercase letter
    if re.search(r'[A-Z]', password):
        results["uppercase"] = True
    else:
        results["errors"].append("Password must contain at least one uppercase letter")
    
    # Check for lowercase letter
    if re.search(r'[a-z]', password):
        results["lowercase"] = True
    else:
        results["errors"].append("Password must contain at least one lowercase letter")
    
    # Check for digit
    if re.search(r'[0-9]', password):
        results["digit"] = True
    else:
        results["errors"].append("Password must contain at least one digit")
    
    # Check for special character
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        results["special"] = True
    else:
        results["errors"].append("Password must contain at least one special character")
    
    # Password is valid if all criteria are met
    results["valid"] = all([
        results["length"],
        results["uppercase"],
        results["lowercase"],
        results["digit"],
        results["special"]
    ])
    
    return results

def validate_url(url: str) -> bool:
    """
    Validate a URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Basic URL validation pattern
    pattern = r'^(https?://)?([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(/.*)?$'
    return bool(re.match(pattern, url))

def validate_sql_injection(value: str) -> bool:
    """
    Check if a string contains potential SQL injection patterns.
    
    Args:
        value: String to check
        
    Returns:
        True if safe (no injection detected), False otherwise
    """
    # List of SQL injection patterns to check for
    sql_patterns = [
        r'(\s|^)(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC|UNION\s+SELECT)(\s|$)',
        r'--',
        r'/\*',
        r';.*?$',
        r'@@[a-zA-Z0-9_]+'
    ]
    
    # Check each pattern
    for pattern in sql_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            return False
    
    return True

def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to an integer.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted integer or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to a float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted float or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_bool(value: Any, default: bool = False) -> bool:
    """
    Safely convert a value to a boolean.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted boolean or default
    """
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        value = value.lower()
        if value in ('yes', 'true', '1', 'y', 't'):
            return True
        if value in ('no', 'false', '0', 'n', 'f'):
            return False
    
    if isinstance(value, (int, float)):
        return bool(value)
    
    return default

def validate_model(model_class: type, data: Dict[str, Any]) -> Tuple[bool, Optional[BaseModel], List[str]]:
    """
    Validate data against a Pydantic model.
    
    Args:
        model_class: Pydantic model class
        data: Data to validate
        
    Returns:
        Tuple of (is_valid, model_instance, error_messages)
    """
    try:
        model_instance = model_class(**data)
        return True, model_instance, []
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field_path = " -> ".join(str(x) for x in error["loc"])
            message = error["msg"]
            errors.append(f"{field_path}: {message}")
        
        return False, None, errors

def validate_dict_keys(data: Dict[str, Any], required_keys: List[str], optional_keys: List[str] = None) -> List[str]:
    """
    Validate that a dictionary contains required keys and only allowed keys.
    
    Args:
        data: Dictionary to validate
        required_keys: List of required keys
        optional_keys: List of optional keys
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check required keys
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")
    
    # Check for unknown keys
    allowed_keys = set(required_keys)
    if optional_keys:
        allowed_keys.update(optional_keys)
    
    for key in data:
        if key not in allowed_keys:
            errors.append(f"Unknown key: {key}")
    
    return errors

def apply_validator(value: Any, validator: Callable[[Any], bool], error_message: str) -> Union[bool, str]:
    """
    Apply a validator function to a value.
    
    Args:
        value: Value to validate
        validator: Validator function
        error_message: Error message if validation fails
        
    Returns:
        True if valid, error message if invalid
    """
    if validator(value):
        return True
    return error_message

# Validation for specific business domain objects
def validate_query_parameters(
    query_params: Dict[str, Any],
    allowed_filters: List[str],
    allowed_sort_fields: List[str]
) -> List[str]:
    """
    Validate query parameters for API endpoints.
    
    Args:
        query_params: Query parameters to validate
        allowed_filters: List of allowed filter fields
        allowed_sort_fields: List of allowed sort fields
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Validate filters
    for filter_key in query_params.get("filters", {}).keys():
        if filter_key not in allowed_filters:
            errors.append(f"Invalid filter field: {filter_key}")
    
    # Validate sort field
    sort_field = query_params.get("sort_by")
    if sort_field and sort_field not in allowed_sort_fields:
        errors.append(f"Invalid sort field: {sort_field}")
    
    # Validate sort order
    sort_order = query_params.get("sort_order", "").lower()
    if sort_order and sort_order not in ("asc", "desc"):
        errors.append(f"Invalid sort order: {sort_order}. Must be 'asc' or 'desc'")
    
    # Validate pagination
    page = safe_int(query_params.get("page"), 1)
    page_size = safe_int(query_params.get("page_size"), 10)
    
    if page < 1:
        errors.append("Page must be a positive integer")
    
    if page_size < 1 or page_size > 100:
        errors.append("Page size must be between 1 and 100")
    
    return errors

def sanitize_input(value: str) -> str:
    """
    Sanitize input string to prevent injection attacks.
    
    Args:
        value: String to sanitize
        
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        return str(value)
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\';&]', '', value)
    
    # Trim whitespace
    sanitized = sanitized.strip()
    
    return sanitized