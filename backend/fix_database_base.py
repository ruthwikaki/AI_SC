#!/usr/bin/env python3
"""
Find where NaturalLanguageQuery.created_by is being referenced
"""

from pathlib import Path
import re
import ast

BACKEND_DIR = Path(r"C:\Users\akiru\OneDrive\Desktop\AI_SC\AI_SC\backend")

def find_created_by_references():
    """Search all Python files for created_by references"""
    print("=== Searching for 'created_by' references ===\n")
    
    findings = []
    
    # Search patterns
    patterns = [
        r'NaturalLanguageQuery\.created_by',
        r'foreign_keys.*created_by',
        r'order_by.*created_by',
        r'filter.*created_by',
        r'back_populates.*created_by',
        r'"created_by"',
        r"'created_by'"
    ]
    
    # Search all Python files
    for py_file in BACKEND_DIR.rglob("*.py"):
        if 'backup' in str(py_file) or '__pycache__' in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for i, line in enumerate(content.splitlines(), 1):
                for pattern in patterns:
                    if re.search(pattern, line):
                        findings.append({
                            'file': py_file.relative_to(BACKEND_DIR),
                            'line_num': i,
                            'line': line.strip(),
                            'pattern': pattern
                        })
        except Exception as e:
            pass
    
    # Display findings
    if findings:
        print(f"Found {len(findings)} references to 'created_by':\n")
        for finding in findings:
            print(f"File: {finding['file']}")
            print(f"Line {finding['line_num']}: {finding['line']}")
            print()
    else:
        print("No direct references found.")
    
    return findings

def check_relationship_definitions():
    """Check all relationship definitions in models"""
    print("\n=== Checking relationship definitions ===\n")
    
    model_files = [
        'app/models/user.py',
        'app/models/query.py',
        'app/models/visualization.py',
        'app/models/analytics.py',
        'app/models/supply_chain.py',
        'app/models/extended_models.py'
    ]
    
    for model_file in model_files:
        file_path = BACKEND_DIR / model_file
        if not file_path.exists():
            continue
            
        print(f"Checking {model_file}:")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all relationship definitions
            rel_pattern = r'(\w+)\s*=\s*relationship\s*\([^)]+\)'
            relationships = re.findall(rel_pattern, content, re.MULTILINE | re.DOTALL)
            
            if relationships:
                print(f"  Found {len(relationships)} relationships: {', '.join(relationships)}")
                
                # Check for problematic foreign_keys
                problematic = re.findall(r'foreign_keys\s*=\s*["\']([^"\']+)["\']', content)
                if problematic:
                    print(f"  ⚠️  String-based foreign_keys found: {problematic}")
            else:
                print("  No relationships found")
                
        except Exception as e:
            print(f"  Error: {e}")
        
        print()

def fix_all_foreign_keys():
    """Fix all string-based foreign_keys references"""
    print("\n=== Fixing all foreign_keys references ===\n")
    
    fixes_made = 0
    
    for py_file in BACKEND_DIR.rglob("*.py"):
        if 'backup' in str(py_file) or '__pycache__' in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            content = original_content
            
            # Fix string-based foreign_keys that reference non-existent columns
            # Pattern: foreign_keys="ModelName.column_name"
            content = re.sub(
                r'foreign_keys\s*=\s*["\']NaturalLanguageQuery\.created_by["\']',
                'foreign_keys=[user_id]',
                content
            )
            
            # Fix any other model.created_by references
            content = re.sub(
                r'foreign_keys\s*=\s*["\'](\w+)\.created_by["\']',
                r'foreign_keys=[created_by]',
                content
            )
            
            # Fix back_populates that might be wrong
            if 'NaturalLanguageQuery' in content:
                # Ensure user relationship uses correct foreign key
                content = re.sub(
                    r'user\s*=\s*relationship\([^,)]*,\s*foreign_keys=["\']NaturalLanguageQuery\.created_by["\'][^)]*\)',
                    'user = relationship("User", back_populates="queries")',
                    content
                )
            
            if content != original_content:
                # Backup and write
                backup_path = py_file.with_suffix('.py.bak')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                fixes_made += 1
                print(f"✓ Fixed {py_file.relative_to(BACKEND_DIR)}")
        
        except Exception as e:
            pass
    
    print(f"\nTotal files fixed: {fixes_made}")

def verify_model_columns():
    """Verify what columns actually exist in NaturalLanguageQuery"""
    print("\n=== Verifying NaturalLanguageQuery columns ===\n")
    
    query_path = BACKEND_DIR / 'app' / 'models' / 'query.py'
    
    if query_path.exists():
        with open(query_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find NaturalLanguageQuery class
        class_match = re.search(r'class NaturalLanguageQuery.*?(?=class|\Z)', content, re.DOTALL)
        if class_match:
            class_content = class_match.group(0)
            
            # Find all Column definitions
            columns = re.findall(r'(\w+)\s*=\s*Column\(', class_content)
            print(f"Columns in NaturalLanguageQuery: {', '.join(columns)}")
            
            if 'created_by' in columns:
                print("✓ created_by column exists")
            else:
                print("❌ created_by column does NOT exist")
                print("✓ But user_id column exists" if 'user_id' in columns else "")
            
            # Find relationships
            relationships = re.findall(r'(\w+)\s*=\s*relationship\(', class_content)
            print(f"\nRelationships: {', '.join(relationships)}")

def main():
    print("=== Find and Fix created_by Reference ===\n")
    
    # Step 1: Find where created_by is referenced
    findings = find_created_by_references()
    
    # Step 2: Check relationship definitions
    check_relationship_definitions()
    
    # Step 3: Verify model columns
    verify_model_columns()
    
    # Step 4: Fix all issues
    print("\n" + "="*50)
    response = input("\nFix all foreign_keys issues? (y/n): ")
    
    if response.lower() == 'y':
        fix_all_foreign_keys()
        
        print("\n✅ All fixes applied!")
        print("\nNow try running again:")
        print("  python main.py")
    else:
        print("\nNo fixes applied.")
        print("\nThe issue is that somewhere in your code, a relationship is trying to")
        print("reference 'NaturalLanguageQuery.created_by' which doesn't exist.")
        print("The column is actually called 'user_id' in NaturalLanguageQuery.")

if __name__ == "__main__":
    main()