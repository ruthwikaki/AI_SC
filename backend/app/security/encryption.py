"""
Encryption utilities for securing sensitive data.

This module provides functions for hashing passwords, verifying hashed passwords,
and encrypting/decrypting sensitive data.
"""

import os
from typing import Optional, Tuple
import base64
import hashlib
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from passlib.context import CryptContext

from app.config import get_settings
from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Initialize password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password to check against
        
    Returns:
        True if password matches hash, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)

# Encryption key management
def generate_key() -> bytes:
    """
    Generate a new encryption key.
    
    Returns:
        New encryption key as bytes
    """
    return Fernet.generate_key()

def derive_key_from_password(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """
    Derive an encryption key from a password and salt.
    
    Args:
        password: Password to derive key from
        salt: Optional salt, will be generated if not provided
        
    Returns:
        Tuple of (key, salt)
    """
    if salt is None:
        salt = os.urandom(16)
        
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key, salt

def get_encryption_key() -> bytes:
    """
    Get the application encryption key.
    
    Returns:
        Encryption key as bytes
    """
    key = settings.encryption_key
    
    if not key:
        # In production, this should fail if no key is configured
        # For development, generate a key but log a warning
        if settings.environment == "production":
            raise ValueError("Encryption key not configured")
            
        logger.warning("Encryption key not configured, generating temporary key")
        key = Fernet.generate_key().decode()
        
    # Ensure key is correctly formatted
    if isinstance(key, str):
        try:
            # If it's a base64 string, decode it
            return base64.urlsafe_b64decode(key.encode())
        except Exception:
            # If it's not valid base64, hash it to create a valid key
            hashed = hashlib.sha256(key.encode()).digest()
            return base64.urlsafe_b64encode(hashed)
    
    return key

# Data encryption
def encrypt_data(data: str) -> str:
    """
    Encrypt sensitive data.
    
    Args:
        data: String data to encrypt
        
    Returns:
        Encrypted data as a base64 string
    """
    try:
        key = get_encryption_key()
        f = Fernet(key)
        encrypted_data = f.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    except Exception as e:
        logger.error(f"Encryption error: {str(e)}")
        raise

def decrypt_data(encrypted_data: str) -> str:
    """
    Decrypt encrypted data.
    
    Args:
        encrypted_data: Encrypted data as a base64 string
        
    Returns:
        Decrypted data as a string
    """
    try:
        key = get_encryption_key()
        f = Fernet(key)
        decoded_data = base64.urlsafe_b64decode(encrypted_data)
        decrypted_data = f.decrypt(decoded_data)
        return decrypted_data.decode()
    except Exception as e:
        logger.error(f"Decryption error: {str(e)}")
        raise

def generate_secure_token(length: int = 32) -> str:
    """
    Generate a secure random token.
    
    Args:
        length: Length of the token in bytes
        
    Returns:
        Secure token as a hex string
    """
    return secrets.token_hex(length)