"""
Password utility functions for hashing and verification
"""
from passlib.context import CryptContext

# Create password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password to verify against
        
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password
    
    Args:
        password: Plain text password to hash
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


# Alias for compatibility
hash_password = get_password_hash


def is_strong_password(password: str) -> bool:
    """
    Check if password meets strength requirements
    
    Args:
        password: Password to check
        
    Returns:
        True if password is strong enough
    """
    if len(password) < 8:
        return False
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    
    return has_upper and has_lower and has_digit


def check_password_strength(password: str) -> dict:
    """
    Check password strength and return detailed analysis
    
    Args:
        password: Password to check
        
    Returns:
        Dictionary with strength analysis
    """
    result = {
        "length": len(password),
        "has_upper": any(c.isupper() for c in password),
        "has_lower": any(c.islower() for c in password),
        "has_digit": any(c.isdigit() for c in password),
        "has_special": any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password),
        "is_strong": False,
        "score": 0
    }
    
    # Calculate score
    if result["length"] >= 8:
        result["score"] += 1
    if result["length"] >= 12:
        result["score"] += 1
    if result["has_upper"]:
        result["score"] += 1
    if result["has_lower"]:
        result["score"] += 1
    if result["has_digit"]:
        result["score"] += 1
    if result["has_special"]:
        result["score"] += 1
    
    result["is_strong"] = result["score"] >= 4
    
    return result
