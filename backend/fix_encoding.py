"""
Fix metadata reserved attribute name
"""
from pathlib import Path

BASE_DIR = Path(r"C:\Users\akiru\OneDrive\Desktop\AI_SC\AI_SC\backend")

def fix_metadata_attribute():
    """Rename metadata attribute to avoid reserved name conflict"""
    print("=== Fixing metadata reserved attribute ===\n")
    
    user_py_path = BASE_DIR / "app" / "models" / "user.py"
    
    with open(user_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace metadata with audit_metadata or meta_data
    replacements = 0
    
    # Find and replace metadata column definitions
    if 'metadata = Column(' in content:
        content = content.replace('metadata = Column(', 'audit_metadata = Column(')
        replacements += 1
        print("✓ Renamed metadata column to audit_metadata")
    
    # Also check for any other metadata references
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'metadata' in line and 'Base.metadata' not in line and 'audit_metadata' not in line:
            print(f"  Found metadata reference at line {i+1}: {line.strip()}")
    
    # Write back
    with open(user_py_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return replacements

def check_other_reserved_names():
    """Check for other potentially reserved attribute names"""
    print("\n=== Checking for other reserved names ===\n")
    
    user_py_path = BASE_DIR / "app" / "models" / "user.py"
    
    # SQLAlchemy reserved attribute names
    reserved = ['metadata', 'query', 'registry', '__table__', '__mapper__', '__tablename__']
    
    with open(user_py_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    found = []
    for i, line in enumerate(lines):
        for reserved_name in reserved:
            if f'{reserved_name} = Column(' in line:
                found.append((i+1, line.strip(), reserved_name))
    
    if found:
        print("Found reserved names:")
        for line_num, line, name in found:
            print(f"  Line {line_num}: {name} - {line}")
    else:
        print("✓ No other reserved names found")

def test_import_final():
    """Final import test"""
    print("\n=== Final Import Test ===\n")
    
    import sys
    import gc
    
    # Clear everything
    modules_to_remove = [m for m in list(sys.modules.keys()) if m.startswith('app')]
    for m in modules_to_remove:
        if m in sys.modules:
            del sys.modules[m]
    
    # Force garbage collection
    gc.collect()
    
    # Add path
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    
    try:
        print("Testing imports...")
        
        # Import all at once
        from app.models import (
            User, UserSession, PasswordResetToken, UserPreference,
            Role, Permission, AuditLog
        )
        
        print("✓ All model classes imported successfully!")
        
        # Try to access tables
        from app.models.user import user_roles, role_permissions
        print("✓ Association tables accessible!")
        
        # Quick sanity check - can we access the classes?
        print(f"\nClasses available:")
        print(f"  - User: {User.__tablename__}")
        print(f"  - Role: {Role.__tablename__}")
        print(f"  - AuditLog: {AuditLog.__tablename__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        
        # More detailed error info
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        
        return False

def main():
    print("=== Fix Metadata Reserved Attribute ===\n")
    
    # Fix metadata
    fix_metadata_attribute()
    
    # Check for other reserved names
    check_other_reserved_names()
    
    # Test
    if test_import_final():
        print("\n" + "="*50)
        print("✅ SUCCESS! All issues have been resolved!")
        print("="*50)
        print("\nYou can now run your application:")
        print("  python main.py")
        print("\nIf you still get errors when running main.py, they are")
        print("likely configuration issues (database connection, etc.)")
        print("not model import issues.")
    else:
        print("\n⚠️  Still having issues")

if __name__ == "__main__":
    main()