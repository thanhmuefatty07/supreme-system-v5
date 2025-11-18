#!/usr/bin/env python3
"""
fix_all_pytorch_imports.py

Comprehensive PyTorch import refactoring script for Supreme System V5.

Applies the enhanced PyTorch import pattern to ALL files in src/ that import torch.

Features:
- Finds all .py files in src/ importing torch
- Replaces direct torch imports with try/except pattern + decorator
- Adds TORCH_AVAILABLE flag and @requires_torch decorator
- Creates backup before modifying
- Comprehensive logging and error handling
- Windows-compatible

Usage: python fix_all_pytorch_imports.py
"""

import os
import re
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pytorch_refactor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PyTorchRefactor:
    """Class to handle comprehensive PyTorch import refactoring across the codebase."""

    def __init__(self, src_dir="src"):
        self.src_dir = Path(src_dir)
        self.backup_dir = Path(f"backup_pytorch_refactor_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.stats = {
            'files_processed': 0,
            'files_modified': 0,
            'torch_imports_found': 0,
            'torch_imports_fixed': 0,
            'decorators_added': 0,
            'functions_decorated': 0
        }

    def find_pytorch_files(self) -> List[Path]:
        """Find all Python files in src/ that import torch."""
        pytorch_files = []

        if not self.src_dir.exists():
            logger.error(f"Source directory {self.src_dir} does not exist")
            return pytorch_files

        # Patterns to detect PyTorch imports
        torch_patterns = [
            r'^import torch\b',
            r'^from torch\b',
            r'\bimport torch\b',
            r'\bfrom torch\b'
        ]

        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for torch imports
                has_torch = any(re.search(pattern, content, re.MULTILINE) for pattern in torch_patterns)

                if has_torch:
                    pytorch_files.append(py_file)
                    logger.info(f"Found PyTorch usage in: {py_file}")

            except Exception as e:
                logger.error(f"Error reading {py_file}: {e}")

        return pytorch_files

    def analyze_functions_needing_torch(self, content: str) -> List[str]:
        """Analyze which functions in the file use PyTorch and need @requires_torch decorator."""
        lines = content.split('\n')
        functions_needing_torch = []
        current_function = None
        brace_count = 0

        for i, line in enumerate(lines):
            # Check for function definition
            func_match = re.match(r'^\s*def\s+(\w+)\s*\(', line)
            if func_match:
                current_function = func_match.group(1)
                brace_count = 0
                continue

            if current_function:
                # Count braces to track function scope
                brace_count += line.count('{') - line.count('}')

                # Check if line contains torch usage
                torch_usage_patterns = [
                    r'\btorch\.',
                    r'\bnn\.',
                    r'\bTensor\b',
                    r'\bParameter\b'
                ]

                has_torch_usage = any(re.search(pattern, line) for pattern in torch_usage_patterns)

                if has_torch_usage and current_function not in functions_needing_torch:
                    functions_needing_torch.append(current_function)

                # End of function
                if brace_count <= 0 and line.strip() == '':
                    current_function = None

        return functions_needing_torch

    def has_torch_available_flag(self, content: str) -> bool:
        """Check if file already has TORCH_AVAILABLE flag."""
        return 'TORCH_AVAILABLE = False' in content and 'TORCH_AVAILABLE = True' in content

    def has_requires_torch_decorator(self, content: str) -> bool:
        """Check if file already has @requires_torch decorator."""
        return 'def requires_torch(' in content

    def create_backup(self, file_path: Path) -> None:
        """Create backup of file before modifying."""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)
            logger.info(f"Created backup directory: {self.backup_dir}")

        # Create relative path structure in backup
        relative_path = file_path.relative_to(self.src_dir)
        backup_file = self.backup_dir / relative_path

        # Create backup directory structure
        backup_file.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(file_path, backup_file)
        logger.info(f"Backed up: {file_path} -> {backup_file}")

    def add_torch_infrastructure(self, content: str) -> str:
        """Add TORCH_AVAILABLE flag, imports, and @requires_torch decorator."""
        lines = content.split('\n')

        # Find where to insert the infrastructure (after imports, before first function/class)
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith(('import ', 'from ', '#', '"""', "'''")):
                insert_idx = i
                break

        # Clean, simple PyTorch infrastructure
        torch_infra = [
            '',
            'try:',
            '    import torch',
            '    import torch.nn as nn',
            '    TORCH_AVAILABLE = True',
            'except (ImportError, OSError):',
            '    torch = None',
            '    nn = None',
            '    TORCH_AVAILABLE = False',
            '',
            '',
            'def requires_torch(func):',
            '    """Decorator to mark functions requiring PyTorch"""',
            '    import functools',
            '',
            '    @functools.wraps(func)',
            '    def wrapper(*args, **kwargs):',
            '        if not TORCH_AVAILABLE:',
            '            raise ImportError("PyTorch is required but not available")',
            '        return func(*args, **kwargs)',
            '    return wrapper',
            '',
        ]

        # Insert the infrastructure
        lines[insert_idx:insert_idx] = torch_infra

        return '\n'.join(lines)

    def fix_torch_imports(self, content: str) -> str:
        """Replace direct torch imports with the new pattern."""
        # Remove existing torch imports (they'll be handled by the infrastructure)
        content = re.sub(r'^import torch\s*$', '# import torch  # Handled by TORCH_AVAILABLE infrastructure', content, flags=re.MULTILINE)
        content = re.sub(r'^from torch\s+.*$', '# from torch import ...  # Handled by TORCH_AVAILABLE infrastructure', content, flags=re.MULTILINE)
        content = re.sub(r'^import torch\.nn\s+as\s+nn\s*$', '# import torch.nn as nn  # Handled by TORCH_AVAILABLE infrastructure', content, flags=re.MULTILINE)

        return content

    def add_decorators_to_functions(self, content: str, functions_needing_torch: List[str]) -> str:
        """Add @requires_torch decorators to functions that need PyTorch."""
        lines = content.split('\n')

        for func_name in functions_needing_torch:
            # Find function definition
            for i, line in enumerate(lines):
                if re.match(rf'^\s*def\s+{re.escape(func_name)}\s*\(', line):
                    # Check if already has decorator
                    if i > 0 and '@requires_torch' in lines[i-1]:
                        continue  # Already decorated

                    # Insert decorator before function definition
                    lines.insert(i, '@requires_torch')
                    self.stats['functions_decorated'] += 1
                    logger.info(f"Added @requires_torch to function: {func_name}")
                    break

        return '\n'.join(lines)

    def process_file(self, file_path: Path) -> bool:
        """Process a single file to refactor PyTorch imports."""
        logger.info(f"Processing: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Skip if already processed
            if self.has_torch_available_flag(content) and self.has_requires_torch_decorator(content):
                logger.info(f"Already processed: {file_path}")
                return False

            # Create backup
            self.create_backup(file_path)

            # Analyze which functions need PyTorch
            functions_needing_torch = self.analyze_functions_needing_torch(content)
            logger.info(f"Functions needing @requires_torch in {file_path.name}: {functions_needing_torch}")

            # Apply transformations
            original_content = content

            # Add PyTorch infrastructure if not present
            if not self.has_torch_available_flag(content):
                content = self.add_torch_infrastructure(content)
                self.stats['decorators_added'] += 1
                logger.info("Added PyTorch infrastructure")

            # Fix torch imports
            content = self.fix_torch_imports(content)

            # Add decorators to functions
            if functions_needing_torch:
                content = self.add_decorators_to_functions(content, functions_needing_torch)

            # Write back if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.stats['files_modified'] += 1
                self.stats['torch_imports_fixed'] += 1

                logger.info(f"Successfully refactored: {file_path}")
                return True
            else:
                logger.info(f"No changes needed: {file_path}")
                return False

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False

    def run(self) -> Dict:
        """Main execution method."""
        logger.info("=== PyTorch Refactoring Started ===")
        logger.info(f"Source directory: {self.src_dir}")

        # Find PyTorch files
        pytorch_files = self.find_pytorch_files()
        self.stats['files_processed'] = len(pytorch_files)
        self.stats['torch_imports_found'] = len(pytorch_files)

        logger.info(f"Found {len(pytorch_files)} files with PyTorch usage:")

        for file_path in pytorch_files:
            logger.info(f"  - {file_path}")

        # Process each file
        logger.info("\n=== Processing Files ===")

        for file_path in pytorch_files:
            self.process_file(file_path)

        # Print summary
        self.print_summary()

        logger.info("=== PyTorch Refactoring Completed ===")
        return self.stats

    def print_summary(self) -> None:
        """Print execution summary."""
        print("\n" + "="*60)
        print("PyTorch Refactoring - Summary")
        print("="*60)

        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files modified: {self.stats['files_modified']}")
        print(f"PyTorch imports found: {self.stats['torch_imports_found']}")
        print(f"PyTorch imports fixed: {self.stats['torch_imports_fixed']}")
        print(f"Infrastructure added: {self.stats['decorators_added']}")
        print(f"Functions decorated: {self.stats['functions_decorated']}")

        if self.backup_dir.exists():
            print(f"Backup directory: {self.backup_dir}")

        print("\nNext steps:")
        print("1. Run tests to verify PyTorch-dependent functionality still works")
        print("2. Check for any import errors in modified files")
        print("3. If issues occur, restore from backup directory")

        print("\nFiles modified:")
        if self.backup_dir.exists():
            for backup_file in self.backup_dir.rglob("*.py"):
                original_file = self.src_dir / backup_file.relative_to(self.backup_dir)
                print(f"  - {original_file}")

def main():
    """Main entry point."""
    refactor = PyTorchRefactor()
    refactor.run()

if __name__ == "__main__":
    main()