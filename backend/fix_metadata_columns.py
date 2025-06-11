#!/usr/bin/env python3
"""
Script to find and fix all occurrences of 'metadata' as a column name in SQLAlchemy models.
This is necessary because 'metadata' is a reserved attribute in SQLAlchemy's declarative base.
"""

import os
import re
import sys
from pathlib import Path

def find_metadata_columns(file_path):
    """Find lines that declare a metadata column in SQLAlchemy models."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match metadata column declarations
    # This will match lines like: metadata = Column(...)
    pattern = r'^(\s*)metadata\s*=\s*Column\s*\('
    
    matches = []
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if re.match(pattern, line):
            matches.append((i + 1, line))
    
    return matches, content, lines

def fix_metadata_columns(file_path, dry_run=True):
    """Fix metadata column declarations by renaming them to meta_data."""
    matches, content, lines = find_metadata_columns(file_path)
    
    if not matches:
        return False
    
    print(f"\nFound metadata columns in {file_path}:")
    for line_num, line in matches:
        print(f"  Line {line_num}: {line.strip()}")
    
    if dry_run:
        print("  (DRY RUN - no changes made)")
        return True
    
    # Replace metadata with meta_data
    new_lines = []
    for i, line in enumerate(lines):
        if any(i + 1 == match[0] for match in matches):
            # Replace metadata = Column( with meta_data = Column(
            new_line = re.sub(r'^(\s*)metadata(\s*=\s*Column\s*\()', r'\1meta_data\2', line)
            new_lines.append(new_line)
            print(f"  Fixed line {i + 1}: {new_line.strip()}")
        else:
            new_lines.append(line)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    print(f"  ✓ Fixed {len(matches)} occurrences")
    return True

def main():
    """Main function to process all Python files in the models directory."""
    if len(sys.argv) > 1 and sys.argv[1] == '--apply':
        dry_run = False
        print("Running in APPLY mode - files will be modified!")
    else:
        dry_run = True
        print("Running in DRY RUN mode - no files will be modified.")
        print("Use --apply flag to actually make changes.")
    
    # Find all Python files in the models directory
    models_dir = Path('app/models')
    if not models_dir.exists():
        print(f"Error: {models_dir} directory not found!")
        return 1
    
    python_files = list(models_dir.glob('*.py'))
    
    print(f"\nScanning {len(python_files)} Python files in {models_dir}...")
    
    files_with_issues = 0
    for file_path in python_files:
        if file_path.name == '__init__.py':
            continue
        
        if fix_metadata_columns(file_path, dry_run):
            files_with_issues += 1
    
    if files_with_issues == 0:
        print("\n✓ No metadata column issues found!")
    else:
        print(f"\n{files_with_issues} file(s) with metadata columns found.")
        if dry_run:
            print("\nTo apply fixes, run: python fix_metadata_columns.py --apply")
    
    # Also check for any references to the metadata field that might need updating
    if not dry_run:
        print("\n⚠️  Remember to also update:")
        print("  1. Any database queries that reference the 'metadata' column")
        print("  2. Any API endpoints that expect/return 'metadata' field")
        print("  3. Any frontend code that uses the 'metadata' field")
        print("  4. Database migration to rename the column from 'metadata' to 'meta_data'")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())