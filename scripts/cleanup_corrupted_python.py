#!/usr/bin/env python3
"""Remove cascading [REVIEWED] markers from corrupted Python files."""

import re
import os
import sys
from pathlib import Path

def clean_file(file_path):
    """Remove cascading # [REVIEWED] markers from start of lines."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern: Remove all cascading "# [REVIEWED] " markers at line start
    # This regex removes one or more cascading markers at the beginning of a line
    content = re.sub(r'^(# \[REVIEWED\] )+', '', content, flags=re.MULTILINE)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    python_dir = Path("/home/xing/Qallow/python")
    cleaned = 0
    
    for py_file in python_dir.rglob("*.py"):
        if clean_file(py_file):
            print(f"✓ Cleaned: {py_file.relative_to('/home/xing/Qallow')}")
            cleaned += 1
    
    print(f"\n✓ Total files cleaned: {cleaned}")

if __name__ == "__main__":
    main()
