#!/usr/bin/env python3
"""
🚀 FAST Lightning Agent - Aggressive Code Fixer
================================================

This is a HIGH-SPEED improvement loop that:
1. ⚡ Runs Qallow build ONCE
2. 🔍 Captures compiler/runtime errors
3. 🔧 DIRECTLY MODIFIES source code to fix issues
4. ✅ Validates fixes work
5. 📊 Reports improvements
6. 🔄 Loops until no more fixes

Much faster than the old system - fixes code directly instead of just configs.
"""

import os
import sys
import re
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import difflib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CompileError:
    """Compiler error with location and message."""
    file_path: str
    line_number: int
    column: int
    message: str
    full_line: str = ""


class ErrorParser:
    """Parse compiler and runtime errors to get file/line info."""
    
    @staticmethod
    def parse_gcc_error(line: str) -> Optional[CompileError]:
        """Parse GCC-format error: file.c:10:5: error: message"""
        match = re.match(r'([^:]+):(\d+):(\d+):\s*(error|warning):\s*(.+)', line)
        if match:
            file_path, line_num, col, severity, message = match.groups()
            return CompileError(
                file_path=file_path,
                line_number=int(line_num),
                column=int(col),
                message=message,
                full_line=line
            )
        return None
    
    @staticmethod
    def parse_errors_from_output(output: str) -> List[CompileError]:
        """Extract all errors from compiler output."""
        errors = []
        for line in output.split('\n'):
            error = ErrorParser.parse_gcc_error(line)
            if error:
                errors.append(error)
        return errors


class CodeFixer:
    """Applies direct code fixes to source files."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.fixes_applied = []
    
    def fix_unused_imports(self, file_path: Path) -> bool:
        """Remove unused import statements."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original = content
            
            # Remove unused Python imports
            lines = content.split('\n')
            new_lines = []
            
            for line in lines:
                # Skip if line is import and seems unused
                if re.match(r'^import |^from .* import', line):
                    # Check if import is actually used
                    import_name = re.findall(r'(?:from|import)\s+(\w+)', line)
                    if import_name:
                        name = import_name[0]
                        # Count occurrences (excluding the import line itself)
                        count = 0
                        for l in lines:
                            if l != line and name in l:
                                count += 1
                        
                        if count > 0:
                            new_lines.append(line)
                        else:
                            logger.info(f"  Removed unused import: {name}")
                else:
                    new_lines.append(line)
            
            new_content = '\n'.join(new_lines)
            
            if new_content != original:
                with open(file_path, 'w') as f:
                    f.write(new_content)
                return True
        
        except Exception as e:
            logger.error(f"Error fixing imports in {file_path}: {e}")
        
        return False
    
    def fix_undefined_reference(self, error: CompileError) -> bool:
        """Try to fix undefined reference errors."""
        try:
            file_path = self.project_root / error.file_path
            if not file_path.exists():
                return False
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Extract function/variable name from error message
            match = re.search(r"'([^']+)'", error.message)
            if not match:
                return False
            
            symbol = match.group(1)
            logger.info(f"  Attempting to fix undefined reference: {symbol}")
            
            # This is complex - would need semantic analysis
            # For now, just log it
            return False
        
        except Exception as e:
            logger.error(f"Error fixing undefined reference: {e}")
        
        return False
    
    def fix_missing_header(self, error: CompileError) -> bool:
        """Try to add missing header files."""
        try:
            file_path = self.project_root / error.file_path
            if not file_path.exists():
                return False
            
            # Extract header name
            match = re.search(r"[<\"]([^>\"]+)[>\"]", error.message)
            if not match:
                return False
            
            header = match.group(1)
            logger.info(f"  Attempting to add missing header: {header}")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check if header already included
            if f"#include" in content and header in content:
                return False
            
            # Add include at top
            lines = content.split('\n')
            insert_pos = 0
            
            # Find first non-comment, non-pragma line
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith(('#', '//')):
                    insert_pos = i
                    break
            
            # Add include
            include_line = f'#include <{header}>' if not header.startswith('"') else f'#include "{header}"'
            lines.insert(insert_pos, include_line)
            
            new_content = '\n'.join(lines)
            with open(file_path, 'w') as f:
                f.write(new_content)
            
            logger.info(f"  Added include: {include_line}")
            return True
        
        except Exception as e:
            logger.error(f"Error fixing missing header: {e}")
        
        return False
    
    def fix_type_mismatch(self, error: CompileError) -> bool:
        """Try to fix type mismatch errors."""
        try:
            file_path = self.project_root / error.file_path
            if not file_path.exists():
                return False
            
            logger.info(f"  Type mismatch in {file_path}:{error.line_number}")
            # This requires semantic analysis - complex to automate
            return False
        
        except Exception as e:
            logger.error(f"Error fixing type mismatch: {e}")
        
        return False
    
    def fix_syntax_error(self, error: CompileError) -> bool:
        """Try to fix common syntax errors."""
        try:
            file_path = self.project_root / error.file_path
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Fix off-by-one since line numbers are 1-indexed
            line_idx = error.line_number - 1
            
            if line_idx >= len(lines):
                return False
            
            line = lines[line_idx]
            original_line = line
            
            # Common syntax fixes
            
            # Fix missing semicolon
            if 'expected' in error.message and ';' in error.message:
                if not line.rstrip().endswith((';', '{', '}', ':')):
                    line = line.rstrip() + ';\n'
                    logger.info(f"  Added missing semicolon")
            
            # Fix mismatched braces
            elif 'expected' in error.message and '{' in error.message:
                line = line.rstrip() + ' {\n'
                logger.info(f"  Added opening brace")
            
            if line != original_line:
                lines[line_idx] = line
                with open(file_path, 'w') as f:
                    f.writelines(lines)
                return True
        
        except Exception as e:
            logger.error(f"Error fixing syntax: {e}")
        
        return False
    
    def apply_error_fix(self, error: CompileError) -> bool:
        """Apply appropriate fix for the error."""
        msg = error.message.lower()
        
        if 'undefined reference' in msg or 'symbol not found' in msg:
            return self.fix_undefined_reference(error)
        elif 'no such file' in msg or 'not found' in msg:
            return self.fix_missing_header(error)
        elif 'type' in msg and 'mismatch' in msg:
            return self.fix_type_mismatch(error)
        elif 'expected' in msg or 'syntax error' in msg:
            return self.fix_syntax_error(error)
        
        return False


class FastBuilder:
    """Fast build with error capture."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.last_output = ""
    
    def build(self, use_cuda: bool = True) -> Tuple[bool, str]:
        """Build project and capture output."""
        logger.info(f"Building (CUDA={use_cuda})...")
        
        try:
            # Configure
            cuda_flag = "ON" if use_cuda else "OFF"
            cmd = f"cmake -S . -B build -DQALLOW_ENABLE_CUDA={cuda_flag} 2>&1"
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=300
            )
            
            if result.returncode != 0:
                logger.warning(f"Configure failed, output:\n{result.stderr}")
                return False, result.stderr
            
            # Build
            try:
                num_cores = os.cpu_count() or 2
            except:
                num_cores = 2
            
            cmd = f"cmake --build build --parallel {num_cores} 2>&1"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=600
            )
            
            output = result.stdout + result.stderr
            self.last_output = output
            
            if result.returncode == 0:
                logger.info("✅ Build successful!")
                return True, output
            else:
                logger.warning("❌ Build failed")
                return False, output
        
        except subprocess.TimeoutExpired:
            logger.error("Build timeout")
            return False, "Build timeout"
        except Exception as e:
            logger.error(f"Build error: {e}")
            return False, str(e)


class LightningAgentFast:
    """Main fast improvement agent."""
    
    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.project_root = Path(".")
        self.builder = FastBuilder(str(self.project_root))
        self.fixer = CodeFixer(str(self.project_root))
        self.error_parser = ErrorParser()
        self.iteration = 0
        self.total_fixes = 0
    
    def run_loop(self):
        """Main improvement loop."""
        logger.info("=" * 70)
        logger.info("🚀 Lightning Agent - Fast Code Fixer Started")
        logger.info("=" * 70)
        
        for self.iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"⚡ ITERATION {self.iteration}/{self.max_iterations}")
            logger.info(f"{'='*70}")
            
            # Build
            success, output = self.builder.build(use_cuda=False)  # Use CPU for faster builds
            
            if success:
                logger.info("✅ Build successful! Running tests...")
                self.run_tests()
                break
            
            # Parse errors
            errors = self.error_parser.parse_errors_from_output(output)
            
            if not errors:
                logger.info("❌ Build failed but no errors found")
                break
            
            logger.info(f"🔍 Found {len(errors)} errors:")
            
            # Apply fixes
            fixed_count = 0
            for i, error in enumerate(errors[:5], 1):  # Fix top 5 errors
                logger.info(f"  {i}. {error.file_path}:{error.line_number}: {error.message}")
                
                if self.fixer.apply_error_fix(error):
                    fixed_count += 1
                    self.total_fixes += 1
            
            if fixed_count == 0:
                logger.warning("⚠️  No fixes could be applied. Stopping.")
                break
            
            logger.info(f"✅ Applied {fixed_count} fixes in this iteration")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🏁 Agent finished after {self.iteration} iterations")
        logger.info(f"📊 Total fixes applied: {self.total_fixes}")
        logger.info(f"{'='*70}\n")
    
    def run_tests(self):
        """Run quick tests to verify everything works."""
        logger.info("Running tests...")
        try:
            result = subprocess.run(
                "ctest --test-dir build --output-on-failure",
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info("✅ All tests passed!")
            else:
                logger.warning("⚠️  Some tests failed")
        
        except Exception as e:
            logger.error(f"Test error: {e}")


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='⚡ Fast Lightning Agent - Aggressive Code Fixer'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,
        help='Maximum iterations (default: 10)'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run continuously until Ctrl+C'
    )
    
    args = parser.parse_args()
    
    agent = LightningAgentFast(max_iterations=args.max_iterations)
    
    if args.daemon:
        iteration = 0
        try:
            while True:
                iteration += 1
                logger.info(f"\n🔄 Daemon run {iteration}")
                agent.run_loop()
                logger.info("Waiting 60 seconds before next run...")
                import time
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("\n✋ Daemon stopped by user")
    else:
        agent.run_loop()


if __name__ == '__main__':
    main()
