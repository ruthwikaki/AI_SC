#!/usr/bin/env python3
"""
Comprehensive validation script for circular imports
"""
import ast
import os
import sys
from pathlib import Path
from collections import defaultdict, deque
import json

class CircularImportDetector:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.import_graph = defaultdict(set)
        self.file_imports = {}
        self.errors = []
        
    def extract_imports(self, file_path):
        """Extract all imports from a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(('import', alias.name, 0))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    level = node.level
                    for alias in node.names:
                        imports.append(('from', module, level, alias.name))
            
            return imports
        except Exception as e:
            self.errors.append(f"Error parsing {file_path}: {e}")
            return []
    
    def resolve_import(self, from_file, import_info):
        """Resolve an import to an actual file path"""
        from_path = Path(from_file)
        
        if import_info[0] == 'import':
            # Absolute import
            module_parts = import_info[1].split('.')
            return self.find_module(module_parts)
        else:
            # Relative import
            level = import_info[2]
            module = import_info[1]
            
            # Go up directories based on level
            current = from_path.parent
            for _ in range(level - 1):
                current = current.parent
            
            if module:
                module_parts = module.split('.')
                for part in module_parts:
                    current = current / part
            
            # Try to find the module
            if current.with_suffix('.py').exists():
                return str(current.with_suffix('.py'))
            elif (current / '__init__.py').exists():
                return str(current / '__init__.py')
            
        return None
    
    def find_module(self, module_parts):
        """Find a module file from module parts"""
        # Start from root
        current = self.root_dir
        
        for part in module_parts:
            current = current / part
            if current.with_suffix('.py').exists():
                return str(current.with_suffix('.py'))
        
        # Check for __init__.py
        if (current / '__init__.py').exists():
            return str(current / '__init__.py')
        
        return None
    
    def build_import_graph(self):
        """Build the import dependency graph"""
        py_files = list(self.root_dir.rglob("*.py"))
        
        for file_path in py_files:
            if 'venv' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            imports = self.extract_imports(file_path)
            self.file_imports[str(file_path)] = imports
            
            for import_info in imports:
                target = self.resolve_import(file_path, import_info)
                if target and Path(target).exists():
                    self.import_graph[str(file_path)].add(target)
    
    def find_cycles(self):
        """Find all circular import cycles"""
        cycles = []
        visited = set()
        
        def dfs(node, path, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.import_graph.get(node, []):
                if neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                elif neighbor not in visited:
                    dfs(neighbor, path, rec_stack)
            
            path.pop()
            rec_stack.remove(node)
        
        for node in self.import_graph:
            if node not in visited:
                dfs(node, [], set())
        
        # Remove duplicates
        unique_cycles = []
        seen = set()
        for cycle in cycles:
            normalized = tuple(sorted(cycle))
            if normalized not in seen:
                seen.add(normalized)
                unique_cycles.append(cycle)
        
        return unique_cycles
    
    def analyze(self):
        """Run the complete analysis"""
        print("Building import graph...")
        self.build_import_graph()
        
        print("Finding circular imports...")
        cycles = self.find_cycles()
        
        if self.errors:
            print(f"\nErrors encountered: {len(self.errors)}")
            for error in self.errors[:5]:
                print(f"  - {error}")
        
        if cycles:
            print(f"\nFound {len(cycles)} circular import cycles:")
            for i, cycle in enumerate(cycles[:20]):  # Show first 20
                print(f"\n{i+1}. Circular import chain:")
                for j, file in enumerate(cycle[:-1]):
                    next_file = cycle[j+1]
                    rel_file = os.path.relpath(file, self.root_dir)
                    rel_next = os.path.relpath(next_file, self.root_dir)
                    print(f"   {rel_file} -> {rel_next}")
        else:
            print("\nNo circular imports detected!")
        
        # Save detailed report
        report = {
            'total_files': len(self.file_imports),
            'total_cycles': len(cycles),
            'errors': self.errors,
            'cycles': [
                [os.path.relpath(f, self.root_dir) for f in cycle]
                for cycle in cycles
            ]
        }
        
        with open('circular_import_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to circular_import_report.json")
        
        return len(cycles) == 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "backend/app"
    
    detector = CircularImportDetector(root_dir)
    success = detector.analyze()
    
    sys.exit(0 if success else 1)
