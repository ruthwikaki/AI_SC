# security/password_utils.py

"""
Password hashing and validation utilities using bcrypt
"""

import re
import secrets
import string
from typing import Optional, Tuple
from datetime import datetime, timedelta

import bcrypt
from passlib.context import CryptContext


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class PasswordValidator:
    """Password validation with configurable rules"""
    
    def __init__(
        self,
        min_length: int = 8,
        max_length: int = 128,
        require_uppercase: bool = True,
        require_lowercase: bool = True,
        require_digits: bool = True,
        require_special: bool = True,
        special_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_digits = require_digits
        self.require_special = require_special
        self.special_chars = special_chars
    
    def validate(self, password: str) -> Tuple[bool, Optional[str]]:
        """
        Validate password against rules
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not password:
            return False, "Password cannot be empty"
        
        if len(password) < self.min_length:
            return False, f"Password must be at least {self.min_length} characters long"
        
        if len(password) > self.max_length:
            return False, f"Password cannot exceed {self.max_length} characters"
        
        if self.require_uppercase and not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"
        
        if self.require_lowercase and not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter"
        
        if self.require_digits and not re.search(r"\d", password):
            return False, "Password must contain at least one digit"
        
        if self.require_special and not any(char in self.special_chars for char in password):
            return False, f"Password must contain at least one special character ({self.special_chars})"
        
        # Check for common weak patterns
        if self._is_common_pattern(password):
            return False, "Password is too common or follows a predictable pattern"
        
        return True, None
    
    def _is_common_pattern(self, password: str) -> bool:
        """Check for common weak password patterns"""
        # Common weak passwords (in production, use a more comprehensive list)
        common_passwords = {
            "password", "123456", "password123", "admin", "letmein",
            "qwerty", "111111", "123123", "abc123", "password1"
        }
        
        lower_password = password.lower()
        
        # Check against common passwords
        if lower_password in common_passwords:
            return True
        
        # Check for repeated characters (e.g., "aaaaaa")
        if len(set(password)) < len(password) // 2:
            return True
        
        # Check for sequential characters (e.g., "abcdef", "123456")
        sequences = ["abcdefghijklmnopqrstuvwxyz", "0123456789", "qwertyuiop", "asdfghjkl"]
        for seq in sequences:
            for i in range(len(seq) - 3):
                if seq[i:i+4] in lower_password or seq[i:i+4][::-1] in lower_password:
                    return True
        
        return False
    
    def get_strength(self, password: str) -> str:
        """
        Calculate password strength
        
        Returns: 'weak', 'medium', 'strong', or 'very_strong'
        """
        if not password:
            return "weak"
        
        score = 0
        
        # Length score
        if len(password) >= 8:
            score += 1
        if len(password) >= 12:
            score += 1
        if len(password) >= 16:
            score += 1
        
        # Character variety score
        if re.search(r"[a-z]", password):
            score += 1
        if re.search(r"[A-Z]", password):
            score += 1
        if re.search(r"\d", password):
            score += 1
        if any(char in self.special_chars for char in password):
            score += 1
        
        # Complexity bonus
        if len(set(password)) > len(password) * 0.7:
            score += 1
        
        if score <= 2:
            return "weak"
        elif score <= 4:
            return "medium"
        elif score <= 6:
            return "strong"
        else:
            return "very_strong"


# Default password validator instance
default_validator = PasswordValidator()


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password to compare against
        
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def validate_password(password: str, validator: Optional[PasswordValidator] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate password using default or custom validator
    
    Args:
        password: Password to validate
        validator: Optional custom validator
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if validator is None:
        validator = default_validator
    return validator.validate(password)


def get_password_strength(password: str, validator: Optional[PasswordValidator] = None) -> str:
    """
    Get password strength rating
    
    Args:
        password: Password to analyze
        validator: Optional custom validator
        
    Returns:
        Strength rating: 'weak', 'medium', 'strong', or 'very_strong'
    """
    if validator is None:
        validator = default_validator
    return validator.get_strength(password)


def generate_password(
    length: int = 16,
    uppercase: bool = True,
    lowercase: bool = True,
    digits: bool = True,
    special: bool = True,
    special_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?",
    exclude_ambiguous: bool = True
) -> str:
    """
    Generate a secure random password
    
    Args:
        length: Password length
        uppercase: Include uppercase letters
        lowercase: Include lowercase letters
        digits: Include digits
        special: Include special characters
        special_chars: Special characters to use
        exclude_ambiguous: Exclude ambiguous characters (0, O, l, 1, etc.)
        
    Returns:
        Generated password
    """
    # Build character pool
    chars = ""
    
    if lowercase:
        chars += string.ascii_lowercase
    if uppercase:
        chars += string.ascii_uppercase
    if digits:
        chars += string.digits
    if special:
        chars += special_chars
    
    if not chars:
        raise ValueError("At least one character type must be enabled")
    
    # Remove ambiguous characters if requested
    if exclude_ambiguous:
        ambiguous = "0O1lI"
        chars = "".join(c for c in chars if c not in ambiguous)
    
    # Ensure password contains at least one of each required type
    password_chars = []
    
    if lowercase and any(c in string.ascii_lowercase for c in chars):
        password_chars.append(secrets.choice([c for c in chars if c in string.ascii_lowercase]))
    if uppercase and any(c in string.ascii_uppercase for c in chars):
        password_chars.append(secrets.choice([c for c in chars if c in string.ascii_uppercase]))
    if digits and any(c in string.digits for c in chars):
        password_chars.append(secrets.choice([c for c in chars if c in string.digits]))
    if special and any(c in special_chars for c in chars):
        password_chars.append(secrets.choice([c for c in chars if c in special_chars]))
    
    # Fill remaining length with random characters
    remaining_length = length - len(password_chars)
    password_chars.extend(secrets.choice(chars) for _ in range(remaining_length))
    
    # Shuffle to avoid predictable patterns
    secrets.SystemRandom().shuffle(password_chars)
    
    return "".join(password_chars)


def generate_reset_token(length: int = 32) -> str:
    """
    Generate a secure password reset token
    
    Args:
        length: Token length
        
    Returns:
        URL-safe token string
    """
    return secrets.token_urlsafe(length)


def generate_temp_password() -> str:
    """
    Generate a temporary password for initial setup or reset
    
    Returns:
        Temporary password (12 characters, mixed case + digits)
    """
    return generate_password(
        length=12,
        uppercase=True,
        lowercase=True,
        digits=True,
        special=False,  # Easier to type
        exclude_ambiguous=True
    )


def check_password_history(
    password: str,
    password_history: list[str],
    history_limit: int = 5
) -> bool:
    """
    Check if password was recently used
    
    Args:
        password: New password to check
        password_history: List of previous password hashes
        history_limit: Number of previous passwords to check
        
    Returns:
        True if password is acceptable (not in history), False otherwise
    """
    # Check only the most recent passwords
    recent_passwords = password_history[-history_limit:] if password_history else []
    
    for old_hash in recent_passwords:
        if verify_password(password, old_hash):
            return False
    
    return True


def is_password_expired(
    last_changed: datetime,
    expiry_days: int = 90
) -> bool:
    """
    Check if password has expired
    
    Args:
        last_changed: When password was last changed
        expiry_days: Password expiry period in days
        
    Returns:
        True if password is expired, False otherwise
    """
    if expiry_days <= 0:
        return False  # No expiry
    
    expiry_date = last_changed + timedelta(days=expiry_days)
    return datetime.utcnow() > expiry_date


def get_password_age_days(last_changed: datetime) -> int:
    """
    Get password age in days
    
    Args:
        last_changed: When password was last changed
        
    Returns:
        Number of days since password was changed
    """
    age = datetime.utcnow() - last_changed
    return age.days


# Password policy configuration
class PasswordPolicy:
    """Password policy configuration"""
    
    def __init__(
        self,
        min_length: int = 8,
        max_length: int = 128,
        require_uppercase: bool = True,
        require_lowercase: bool = True,
        require_digits: bool = True,
        require_special: bool = True,
        special_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?",
        expiry_days: int = 90,
        history_limit: int = 5,
        max_attempts: int = 5,
        lockout_duration: int = 30  # minutes
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_digits = require_digits
        self.require_special = require_special
        self.special_chars = special_chars
        self.expiry_days = expiry_days
        self.history_limit = history_limit
        self.max_attempts = max_attempts
        self.lockout_duration = lockout_duration
        
        # Create validator with policy settings
        self.validator = PasswordValidator(
            min_length=min_length,
            max_length=max_length,
            require_uppercase=require_uppercase,
            require_lowercase=require_lowercase,
            require_digits=require_digits,
            require_special=require_special,
            special_chars=special_chars
        )
    
    def to_dict(self) -> dict:
        """Convert policy to dictionary for API responses"""
        return {
            "min_length": self.min_length,
            "max_length": self.max_length,
            "require_uppercase": self.require_uppercase,
            "require_lowercase": self.require_lowercase,
            "require_digits": self.require_digits,
            "require_special": self.require_special,
            "special_chars": self.special_chars,
            "expiry_days": self.expiry_days,
            "history_limit": self.history_limit,
            "max_attempts": self.max_attempts,
            "lockout_duration": self.lockout_duration
        }


# Default password policy
default_policy = PasswordPolicy()


# Utility functions for password reset
def create_reset_token_hash(token: str) -> str:
    """Create a hash of reset token for storage"""
    return hash_password(token)


def verify_reset_token(token: str, token_hash: str) -> bool:
    """Verify a reset token against its hash"""
    return verify_password(token, token_hash)