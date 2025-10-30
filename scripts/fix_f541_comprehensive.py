#!/usr/bin/env python3
"""
🛠️ Supreme System V5 - Comprehensive F541 Fixer
Giải pháp tối ưu, triệt để nhất để sửa lỗi F541
"""

import re
import sys
import os
from pathlib import Path
import argparse
import time

class ComprehensiveF541Fixer:
    """Công cụ sửa F541 tối ưu và toàn diện"""
    
    def __init__(self, dry_run=True, verbose=False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.stats = {
            'files_processed': 0,
            'files_modified': 0,
            'total_fixes': 0,
            'method_stats': {'simple': 0, 'regex': 0, 'line_by_line': 0}
        }
    
    def method_1_simple_replacement(self, content):
        """Phương pháp 1: Simple string replacement - Nhanh nhất"""
        fixes = []
        modified_content = content
        
        # List các f-string patterns phổ biến không có placeholder
        common_f_strings = [
            'f"✅ Phase 2 components successfully imported"',
            'f"🧠 Neuromorphic Computing: Brain-inspired processing"',
            'f"⚡ Ultra-Low Latency: Sub-microsecond capability"',
            'f"🔧 Initializing Phase 2 breakthrough components..."',
            'f"   🧠 Initializing Neuromorphic Processor..."',
            'f"   ⚡ Initializing Ultra-Low Latency Engine..."',
            'f"✅ Phase 2 components initialized successfully"',
            'f"✅ Mock Phase 2 components initialized"',
            'f"🧪 Running Phase 2 integrated demonstration..."',
            'f"✅ Phase 2 integrated demonstration completed successfully"'
        ]
        
        for f_string in common_f_strings:
            if f_string in modified_content:
                # Remove f prefix
                regular_string = f_string[1:]  # Remove 'f' prefix
                modified_content = modified_content.replace(f_string, regular_string)
                fixes.append(f"Fixed: {f_string} -> {regular_string}")
                self.stats['method_stats']['simple'] += 1
        
        return modified_content, fixes
    
    def method_2_smart_regex(self, content):
        """Phương pháp 2: Smart regex với timeout protection"""
        fixes = []
        modified_content = content
        
        try:
            # Timeout protection cho regex
            start_time = time.time()
            timeout = 5  # 5 seconds max
            
            # Pattern đơn giản và hiệu quả - giới hạn độ dài để tránh catastrophic backtracking
            patterns = [
                (r'f"([^"]{1,100}?)"', '"{}"'),    # f"short text" - limit to 100 chars
                (r"f'([^']{1,100}?)'", "'{}'"),   # f'short text' - limit to 100 chars
            ]
            
            for pattern, replacement_template in patterns:
                if time.time() - start_time > timeout:
                    break
                    
                def replace_func(match):
                    content_inside = match.group(1)
                    if '{' not in content_inside or '}' not in content_inside:
                        # No placeholders, remove f prefix
                        return replacement_template.format(content_inside)
                    return match.group(0)  # Keep as f-string
                
                old_content = modified_content
                modified_content = re.sub(pattern, replace_func, modified_content)
                
                if modified_content != old_content:
                    # Count differences
                    count = old_content.count('f"') + old_content.count("f'") - (modified_content.count('f"') + modified_content.count("f'"))
                    if count > 0:
                        fixes.append(f"Regex method fixed {count} f-strings")
                        self.stats['method_stats']['regex'] += count
        
        except Exception as e:
            fixes.append(f"Regex method error (fallback used): {e}")
        
        return modified_content, fixes
    
    def method_3_line_by_line(self, content):
        """Phương pháp 3: Line by line - An toàn nhất"""
        fixes = []
        lines = content.split('\n')
        modified_lines = []
        
        for line_num, line in enumerate(lines, 1):
            modified_line = line
            
            # Simple check for f-strings
            if 'f"' in line:
                # Find positions of f" patterns
                pos = 0
                while True:
                    pos = line.find('f"', pos)
                    if pos == -1:
                        break
                    
                    # Find end quote
                    end_pos = line.find('"', pos + 2)
                    if end_pos != -1:
                        f_string_content = line[pos+2:end_pos]
                        if '{' not in f_string_content or '}' not in f_string_content:
                            # Replace f"content" with "content"
                            modified_line = modified_line.replace(f'f"{f_string_content}"', f'"{f_string_content}"', 1)
                            fixes.append(f"Line {line_num}: Fixed f-string without placeholder")
                            self.stats['method_stats']['line_by_line'] += 1
                    
                    pos += 2
            
            # Also check f' patterns
            if "f'" in line:
                pos = 0
                while True:
                    pos = line.find("f'", pos)
                    if pos == -1:
                        break
                    
                    end_pos = line.find("'", pos + 2)
                    if end_pos != -1:
                        f_string_content = line[pos+2:end_pos]
                        if '{' not in f_string_content or '}' not in f_string_content:
                            modified_line = modified_line.replace(f"f'{f_string_content}'", f"'{f_string_content}'", 1)
                            fixes.append(f"Line {line_num}: Fixed f-string without placeholder")
                            self.stats['method_stats']['line_by_line'] += 1
                    
                    pos += 2
            
            modified_lines.append(modified_line)
        
        return '\n'.join(modified_lines), fixes
    
    def fix_file(self, file_path):
        """Sửa file với tất cả phương pháp"""
        try:
            if self.verbose:
                print(f"\n🔧 Processing: {file_path}")
            
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Apply all methods sequentially
            content = original_content
            all_fixes = []
            
            # Method 1: Simple replacement
            content, fixes1 = self.method_1_simple_replacement(content)
            all_fixes.extend(fixes1)
            
            # Method 2: Smart regex (with timeout)
            content, fixes2 = self.method_2_smart_regex(content)
            all_fixes.extend(fixes2)
            
            # Method 3: Line by line (cleanup)
            content, fixes3 = self.method_3_line_by_line(content)
            all_fixes.extend(fixes3)
            
            # Check if any changes were made
            if content != original_content:
                self.stats['total_fixes'] += len(all_fixes)
                
                if not self.dry_run:
                    # Write modified content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.stats['files_modified'] += 1
                    print(f"✅ Fixed {file_path} - {len(all_fixes)} changes")
                else:
                    print(f"🧪 DRY RUN: Would fix {file_path} - {len(all_fixes)} changes")
                
                if self.verbose:
                    for fix in all_fixes:
                        print(f"    {fix}")
            else:
                if self.verbose:
                    print(f"✅ No F541 issues in {file_path}")
            
            self.stats['files_processed'] += 1
            return len(all_fixes) > 0
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return False
    
    def fix_multiple_files(self, file_paths):
        """Sửa nhiều files"""
        print(f"🚀 Comprehensive F541 Fixer")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print(f"Files to process: {len(file_paths)}")
        print("=" * 50)
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                self.fix_file(file_path)
            else:
                print(f"⚠️ File not found: {file_path}")
    
    def print_summary(self):
        """In summary báo cáo"""
        print("\n" + "=" * 50)
        print("📊 COMPREHENSIVE F541 FIXING SUMMARY")
        print("=" * 50)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files modified: {self.stats['files_modified']}")
        print(f"Total fixes: {self.stats['total_fixes']}")
        print("\nMethod breakdown:")
        for method, count in self.stats['method_stats'].items():
            print(f"  {method}: {count} fixes")
        
        if self.dry_run:
            print("\n🧪 This was a DRY RUN - no files were actually modified")
            print("Run with --live to apply changes")

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive F541 Fixer for Supreme System V5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fix_f541_comprehensive.py                    # Dry run on default files
  python fix_f541_comprehensive.py --live             # Apply fixes
  python fix_f541_comprehensive.py --all-py --live    # Fix all .py files
  python fix_f541_comprehensive.py phase2_main.py -v  # Fix specific file with verbose
        """
    )
    parser.add_argument('files', nargs='*', help='Files to fix')
    parser.add_argument('--live', action='store_true', help='Apply changes (default is dry-run)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--all-py', action='store_true', help='Fix all .py files in current directory')
    
    args = parser.parse_args()
    
    # Determine files to process
    if args.all_py:
        files_to_fix = [str(p) for p in Path('.').glob('*.py')]
    elif args.files:
        files_to_fix = args.files
    else:
        # Default files from Supreme System V5
        files_to_fix = ['phase2_main.py', 'main.py', 'run_backtest.py']
        files_to_fix = [f for f in files_to_fix if os.path.exists(f)]
    
    if not files_to_fix:
        print("❌ No files to process")
        return
    
    # Create fixer
    fixer = ComprehensiveF541Fixer(dry_run=not args.live, verbose=args.verbose)
    
    # Fix files
    fixer.fix_multiple_files(files_to_fix)
    fixer.print_summary()

if __name__ == "__main__":
    main()
