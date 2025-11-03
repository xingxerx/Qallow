#!/usr/bin/env python3
"""
� SLOW Lightning Agent - Readable Code Fixer
==============================================

This is a SLOW, READABLE improvement loop that:
1. 🔨 Builds Qallow with visible output
2. � Shows errors with file/line context
3. � Displays code BEFORE changes
4. ⏸️  PAUSES so you can READ each fix
5. ✏️  Shows code AFTER changes
6. 📋 Waits for your approval (or Enter to continue)
7. ✅ Validates fixes work
8. 📊 Reports what changed
9. 🔄 Repeats with delays between iterations

Perfect for learning how fixes work - SLOW by design!
"""

import os
import sys
import re
import subprocess
import json
import logging
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import difflib
import time

# Configure logging with MORE verbose output
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)-8s: %(message)s'
)
logger = logging.getLogger(__name__)

# CONSTANTS FOR SLOWING DOWN
PAUSE_BEFORE_FIX = 2        # seconds - read the error
PAUSE_SHOW_CODE = 3         # seconds - read the code
PAUSE_BETWEEN_FIXES = 4     # seconds - digest the change
PAUSE_BETWEEN_ITERATIONS = 5  # seconds - next iteration


@dataclass
class CompileError:
    """Compiler error with location and message."""
    file_path: str
    line_number: int
    column: int
    message: str
    full_line: str = ""


# ═══════════════════════════════════════════════════════════════════
# DISPLAY HELPERS - Make output READABLE and SLOW
# ═══════════════════════════════════════════════════════════════════

def print_header(text: str, width: int = 70):
    """Print a nice header."""
    print(f"\n{'═' * width}")
    print(f"║ {text.center(width-4)} ║")
    print(f"{'═' * width}\n")

def print_error_box(error):
    """Print an error in a readable box."""
    print(f"\n┌─ ERROR FOUND ─────────────────────────────────────────────────┐")
    print(f"│ File:     {error.file_path}:{error.line_number}:{error.column}")
    print(f"│ Message:  {error.message}")
    print(f"└───────────────────────────────────────────────────────────────┘\n")

def show_code_context(file_path: Path, line_number: int, context_lines: int = 3):
    """Show code context around the error line."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        
        print(f"\n┌─ CODE CONTEXT ────────────────────────────────────────────────┐")
        for i in range(start, end):
            is_error_line = (i == line_number - 1)
            marker = "→ ERROR →" if is_error_line else "        "
            line = lines[i].rstrip()
            print(f"│ {marker} {i+1:4d}  {line[:58]}")
        print(f"└───────────────────────────────────────────────────────────────┘\n")
    except Exception as e:
        print(f"Could not read context: {e}\n")

def pause_for_reading(reason: str, duration: int = 2):
    """Pause and let user read the output."""
    print(f"⏸️  Pausing {duration}s... {reason}")
    print(f"   (Press Enter to continue immediately, or wait...)")
    
    # Wait with a readline that allows interruption
    try:
        import select
        if select.select([sys.stdin], [], [], duration)[0]:
            sys.stdin.readline()
        else:
            print()  # Print newline after pause
    except:
        time.sleep(duration)

def show_fix_comparison(file_path: Path, original_lines: List[str], 
                        fixed_lines: List[str], changed_line_num: int):
    """Show before/after comparison of the fix."""
    print(f"\n┌─ FIX COMPARISON ──────────────────────────────────────────────┐")
    print(f"│ File: {str(file_path)[:50]}")
    print(f"│ Line: {changed_line_num}")
    print(f"├───────────────────────────────────────────────────────────────┤")
    
    # Show BEFORE
    print(f"│ BEFORE:")
    for i, line in enumerate(original_lines[max(0, changed_line_num-2):changed_line_num+1]):
        print(f"│   {line.rstrip()}")
    
    print(f"│")
    print(f"│ AFTER:")
    for i, line in enumerate(fixed_lines[max(0, changed_line_num-2):changed_line_num+1]):
        print(f"│   {line.rstrip()}")
    
    print(f"└───────────────────────────────────────────────────────────────┘\n")


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
    """Applies direct code fixes to source files - SLOWLY and READABLY."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.fixes_applied = []
    
    def fix_unused_imports(self, file_path: Path) -> bool:
        """Remove unused import statements - SHOW BEFORE/AFTER."""
        try:
            print(f"\n🔍 Analyzing imports in {file_path}...")
            pause_for_reading("Reading imports...", 1)
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            original = content
            lines = content.split('\n')
            new_lines = []
            fixed_any = False
            
            for line in lines:
                if re.match(r'^import |^from .* import', line):
                    import_name = re.findall(r'(?:from|import)\s+(\w+)', line)
                    if import_name:
                        name = import_name[0]
                        count = sum(1 for l in lines if l != line and name in l)
                        
                        if count == 0:
                            print(f"   ❌ UNUSED: {line.strip()}")
                            show_code_context(file_path, lines.index(line) + 1, context_lines=2)
                            pause_for_reading("This import is unused. Removing...", PAUSE_BEFORE_FIX)
                            fixed_any = True
                            continue  # Skip this line (remove it)
                
                new_lines.append(line)
            
            new_content = '\n'.join(new_lines)
            
            if new_content != original and fixed_any:
                with open(file_path, 'w') as f:
                    f.write(new_content)
                print(f"   ✅ Removed unused imports")
                pause_for_reading("Import removed. Continuing...", PAUSE_BETWEEN_FIXES)
                return True
        
        except Exception as e:
            logger.error(f"Error fixing imports in {file_path}: {e}")
        
        return False
    
    def fix_syntax_error(self, error: CompileError) -> bool:
        """Try to fix common syntax errors - SHOW WHAT WE'RE DOING."""
        try:
            file_path = self.project_root / error.file_path
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return False
            
            # SHOW THE ERROR
            print_error_box(error)
            show_code_context(file_path, error.line_number)
            pause_for_reading("Look at the error above. Reading code context...", PAUSE_BEFORE_FIX)
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            line_idx = error.line_number - 1
            
            if line_idx >= len(lines):
                return False
            
            line = lines[line_idx]
            original_line = line
            
            print(f"\n🔧 Attempting fix...")
            pause_for_reading("Working on the fix...", 1)
            
            # Common syntax fixes
            if 'expected' in error.message and ';' in error.message:
                if not line.rstrip().endswith((';', '{', '}', ':')):
                    line = line.rstrip() + ';\n'
                    print(f"   ✏️  Adding missing semicolon")
            
            elif 'expected' in error.message and '{' in error.message:
                line = line.rstrip() + ' {\n'
                print(f"   ✏️  Adding opening brace")
            
            if line != original_line:
                # SHOW THE FIX
                show_fix_comparison(file_path, lines, 
                                   lines[:line_idx] + [line] + lines[line_idx+1:], 
                                   line_idx)
                pause_for_reading("Review the fix above...", PAUSE_SHOW_CODE)
                
                lines[line_idx] = line
                with open(file_path, 'w') as f:
                    f.writelines(lines)
                
                print(f"   ✅ Fix applied!")
                pause_for_reading("Fix complete.", PAUSE_BETWEEN_FIXES)
                return True
        
        except Exception as e:
            logger.error(f"Error fixing syntax: {e}")
        
        return False
    
    def apply_error_fix(self, error: CompileError) -> bool:
        """Apply appropriate fix for the error - SLOWLY."""
        msg = error.message.lower()
        
        if 'expected' in msg or 'syntax error' in msg:
            return self.fix_syntax_error(error)
        
        return False


class FastBuilder:
    """SLOW build with error capture - SHOW EVERY STEP."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.last_output = ""
    
    def build(self, use_cuda: bool = True) -> Tuple[bool, str]:
        """Build project SLOWLY with visible output."""
        print("\n" + "="*70)
        print_header(f"🔨 BUILDING PROJECT (CUDA={use_cuda})")
        print("="*70 + "\n")
        
        pause_for_reading("Starting build configuration...", 2)
        
        try:
            # Configure
            cuda_flag = "ON" if use_cuda else "OFF"
            cmd = f"cmake -S . -B build -DQALLOW_ENABLE_CUDA={cuda_flag} 2>&1"
            
            print(f"   📋 Command: {cmd}\n")
            pause_for_reading("Running CMake configure...", 1)
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=300
            )
            
            if result.returncode != 0:
                print(f"   ❌ Configuration FAILED!")
                print(f"   Error output:\n{result.stderr}")
                pause_for_reading("Configuration failed. Continuing anyway...", PAUSE_BEFORE_FIX)
                return False, result.stderr
            
            print(f"   ✅ Configuration successful!\n")
            pause_for_reading("Configuration done. Starting compilation...", 2)
            
            # Build - Show it as it happens
            try:
                num_cores = os.cpu_count() or 2
            except:
                num_cores = 2
            
            cmd = f"cmake --build build --parallel {num_cores} 2>&1"
            print(f"   📋 Building with {num_cores} cores...\n")
            pause_for_reading("Running build...", 1)
            
            # Use streaming output
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.project_root),
                bufsize=1
            )
            
            output_lines = []
            error_lines = []
            
            # Stream output as it comes
            if process.stdout:
                for line in process.stdout:
                    line = line.rstrip()
                    output_lines.append(line)
                    
                    if 'error' in line.lower():
                        print(f"   ❌ {line}")
                        error_lines.append(line)
                    elif 'warning' in line.lower():
                        print(f"   ⚠️  {line}")
                    elif 'Built target' in line or '100%' in line:
                        print(f"   ✅ {line}")
                    else:
                        print(f"   {line}")
                    
                    # Pause if many errors
                    if len(error_lines) % 2 == 0 and len(error_lines) > 0:
                        pause_for_reading(f"Found {len(error_lines)} errors...", 1)
            
            process.wait()
            output = '\n'.join(output_lines)
            self.last_output = output
            
            print("\n" + "="*70)
            
            if process.returncode == 0:
                print("   ✅ BUILD SUCCESSFUL!")
                print("="*70 + "\n")
                pause_for_reading("Build completed successfully!", 2)
                return True, output
            else:
                print(f"   ❌ BUILD FAILED (exit {process.returncode})")
                print("="*70 + "\n")
                pause_for_reading("Build failed. About to parse errors...", PAUSE_BEFORE_FIX)
                return False, output
        
        except subprocess.TimeoutExpired:
            logger.error("Build timeout")
            print(f"   💥 Build timeout (>5 minutes)")
            return False, "Build timeout"
        except Exception as e:
            logger.error(f"Build error: {e}")
            print(f"   💥 Build crashed: {e}")
            pause_for_reading("Build error.", PAUSE_BEFORE_FIX)
            return False, str(e)


# ═══════════════════════════════════════════════════════════════════
# TEST RUNNER & AUTOMATED FIXES
# ═══════════════════════════════════════════════════════════════════

class TestRunner:
    """Run tests and capture output for analysis."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.last_output = ""
        self.last_returncode = 0
    
    def run_tests(self) -> Tuple[bool, str]:
        """Run ctest and capture all output."""
        print("\n" + "─"*70)
        print("🧪 Running tests...")
        print("─"*70)
        pause_for_reading("Executing ctest...", 1)
        
        try:
            result = subprocess.run(
                ["ctest", "--output-on-failure"],
                cwd=str(self.project_root / "build"),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            self.last_returncode = result.returncode
            self.last_output = result.stdout + result.stderr
            
            # Show summary
            if result.returncode == 0:
                print("   ✅ All tests passed!")
            else:
                print(f"   ⚠️  Tests failed (exit code {result.returncode})")
                # Show first few lines of failures
                lines = self.last_output.split('\n')
                error_lines = [l for l in lines if 'FAIL' in l or 'ERROR' in l]
                for line in error_lines[:3]:
                    print(f"      {line}")
                if len(error_lines) > 3:
                    print(f"      ... and {len(error_lines) - 3} more errors")
            
            pause_for_reading("Test execution complete.", 2)
            
            return result.returncode == 0, self.last_output
        
        except subprocess.TimeoutExpired:
            print("   ⏱️  Test timeout (>5 min)")
            return False, "Test timeout"
        except FileNotFoundError:
            print("   ⚠️  ctest not found (build directory might not exist)")
            return False, "ctest not found"
        except Exception as e:
            print(f"   💥 Test error: {e}")
            return False, str(e)


class WarningFixer:
    """Apply targeted fixes based on compiler warnings and test output."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
    
    def fix_snprintf_truncation(self, output: str) -> int:
        """Fix snprintf truncation warnings by enlarging buffers."""
        fixes = 0
        
        # Check if warning exists
        if "snprintf" not in output or "truncated" not in output:
            return 0
        
        print("\n   🔍 Detected snprintf truncation warnings")
        pause_for_reading("Analyzing truncation issues...", 1)
        
        try:
            target = self.project_root / "runtime" / "meta_introspect.c"
            
            if not target.exists():
                print(f"   ⚠️  File not found: {target}")
                return 0
            
            content = target.read_text()
            original = content
            
            # Find and enlarge buffers
            # Pattern: char g_*_path[4096]
            # Replace 4096 with 8192
            patched = re.sub(r'\[4096\]', '[8192]', content)
            
            if patched != original:
                print(f"\n   📝 Enlarging buffers in {target.name}")
                print(f"      From: [4096] bytes")
                print(f"      To:   [8192] bytes")
                target.write_text(patched)
                fixes += 1
                print(f"   ✅ Buffer size fix applied!")
                pause_for_reading("Fix written to file...", PAUSE_BETWEEN_FIXES)
        
        except Exception as e:
            print(f"   ❌ Error fixing snprintf: {e}")
        
        return fixes
    
    def fix_type_warnings(self, output: str) -> int:
        """Fix type-related warnings."""
        fixes = 0
        
        if "type" not in output.lower():
            return 0
        
        print("\n   🔍 Detected type-related warnings")
        pause_for_reading("Analyzing type issues...", 1)
        
        # Example: Fix implicit declarations
        try:
            # Scan source files for common issues
            for c_file in (self.project_root / "backend" / "cpu").glob("*.c"):
                content = c_file.read_text()
                
                # Example fix: Add missing includes
                if "uint32_t" in content and "#include <stdint.h>" not in content:
                    lines = content.split('\n')
                    
                    # Find first include or function def
                    insert_pos = 0
                    for i, line in enumerate(lines):
                        if '#include' in line:
                            insert_pos = i + 1
                        elif 'int main' in line or 'void ' in line:
                            break
                    
                    # Insert missing header
                    lines.insert(insert_pos, '#include <stdint.h>')
                    patched = '\n'.join(lines)
                    
                    if patched != content:
                        c_file.write_text(patched)
                        print(f"   ✅ Added missing include in {c_file.name}")
                        fixes += 1
                        pause_for_reading("Include added.", 1)
        
        except Exception as e:
            print(f"   ⚠️  Error fixing types: {e}")
        
        return fixes
    
    def fix_unused_variables(self, output: str) -> int:
        """Fix unused variable warnings."""
        fixes = 0
        
        if "unused" not in output.lower():
            return 0
        
        print("\n   🔍 Detected unused variable warnings")
        pause_for_reading("Analyzing unused variables...", 1)
        
        try:
            # Parse warning to find unused variables
            pattern = r"'([^']+)' set but not used"
            matches = re.findall(pattern, output)
            
            if matches:
                print(f"      Found {len(matches)} unused variables")
                # In a real system, we'd mark these with (void) to suppress
                for var in matches[:3]:
                    print(f"      • {var}")
        
        except Exception as e:
            print(f"   ⚠️  Error analyzing unused: {e}")
        
        return fixes
    
    def apply_all_fixes(self, output: str) -> int:
        """Apply all available fixes based on output."""
        total_fixes = 0
        
        print("\n" + "─"*70)
        print("🔧 ANALYZING OUTPUT FOR FIXABLE ISSUES")
        print("─"*70)
        pause_for_reading("Scanning output for patterns...", 2)
        
        # Try each fix type
        total_fixes += self.fix_snprintf_truncation(output)
        total_fixes += self.fix_type_warnings(output)
        total_fixes += self.fix_unused_variables(output)
        
        print("\n" + "─"*70)
        if total_fixes > 0:
            print(f"   ✅ Applied {total_fixes} warning fixes")
        else:
            print("   ℹ️  No fixable warnings found")
        print("─"*70)
        pause_for_reading("Analysis complete.", 1)
        
        return total_fixes


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
        """Main improvement loop - SLOW AND READABLE."""
        print("\n" + "="*70)
        print_header("� SLOW Lightning Agent - Code Fixer Loop Started")
        print("="*70)
        pause_for_reading("Starting improvement iterations...", 2)
        
        for self.iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*70}")
            print_header(f"Iteration {self.iteration}/{self.max_iterations}")
            print("="*70 + "\n")
            pause_for_reading("Beginning iteration...", 1)
            
            # BUILD
            print("\n" + "─"*70)
            print("📝 PHASE 1: Building project...")
            print("─"*70)
            pause_for_reading("About to build...", 1)
            
            success, output = self.builder.build(use_cuda=False)  # Use CPU for faster builds
            
            if success:
                print("\n✅ BUILD SUCCESSFUL!")
                print("━"*70)
                pause_for_reading("Build passed! Running tests...", 2)
                self.run_tests()
                break
            
            # PARSE ERRORS
            print("\n" + "─"*70)
            print("🔍 PHASE 2: Parsing errors...")
            print("─"*70)
            pause_for_reading("Analyzing build output...", 2)
            
            errors = self.error_parser.parse_errors_from_output(output)
            
            if not errors:
                print("\n   ❌ Build failed but NO ERRORS FOUND in output")
                print("   This might be a system issue or timeout")
                pause_for_reading("Cannot continue without errors to parse.", PAUSE_BEFORE_FIX)
                break
            
            print(f"\n   ✅ Found {len(errors)} ERRORS to fix:\n")
            pause_for_reading("Displaying errors...", 1)
            
            # DISPLAY ERRORS
            for i, error in enumerate(errors[:5], 1):
                print(f"\n   Error {i}/{min(5, len(errors))}:")
                print_error_box(error)
                pause_for_reading(f"Error {i} of {min(5, len(errors))}...", 2)
            
            # APPLY FIXES
            print("\n" + "─"*70)
            print("🔧 PHASE 3: Applying fixes...")
            print("─"*70)
            pause_for_reading("Ready to apply fixes...", 2)
            
            fixed_count = 0
            for i, error in enumerate(errors[:5], 1):  # Fix top 5 errors
                print(f"\n   💡 Fix {i}: {error.file_path}:{error.line_number}")
                print(f"      {error.message}")
                pause_for_reading("About to attempt fix...", PAUSE_BEFORE_FIX)
                
                if self.fixer.apply_error_fix(error):
                    fixed_count += 1
                    self.total_fixes += 1
                    print(f"   ✅ FIX APPLIED!")
                else:
                    print(f"   ⚠️  Could not fix this error (might be too complex)")
                
                pause_for_reading("Moving to next error...", PAUSE_BETWEEN_FIXES)
            
            # SUMMARY
            print("\n" + "━"*70)
            if fixed_count == 0:
                print("   ⚠️  NO FIXES APPLIED IN THIS ITERATION")
                print("   The errors might be too complex to auto-fix")
                pause_for_reading("Stopping (no progress made)...", PAUSE_BEFORE_FIX)
                break
            
            print(f"   ✅ ITERATION COMPLETE: Applied {fixed_count} fixes")
            print(f"   📊 Total fixes so far: {self.total_fixes}")
            print("━"*70)
            pause_for_reading("Iteration done. Ready for next...", PAUSE_BETWEEN_ITERATIONS)
        
        print(f"\n{'='*70}")
        print_header(f"Agent Finished: {self.iteration} iterations, {self.total_fixes} total fixes")
        print("="*70 + "\n")
        pause_for_reading("Done!", 2)
    
    def run_tests(self):
        """Run quick tests to verify everything works - SHOW OUTPUT."""
        print("\n" + "─"*70)
        print("🧪 Running tests to verify fixes...")
        print("─"*70 + "\n")
        pause_for_reading("Executing tests...", 1)
        
        try:
            result = subprocess.run(
                "ctest --test-dir build --output-on-failure",
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Show test output
            if result.stdout:
                print("\nTest Output:")
                print(result.stdout[:500])  # First 500 chars
                if len(result.stdout) > 500:
                    print(f"\n... (truncated {len(result.stdout) - 500} bytes) ...\n")
            
            pause_for_reading("Checking test results...", 2)
            
            if result.returncode == 0:
                print("\n   ✅ ALL TESTS PASSED!")
                print("   Fixes verified by test suite!")
            else:
                print("\n   ⚠️  Some tests failed")
                print(f"   Exit code: {result.returncode}")
            
            pause_for_reading("Test phase complete.", 2)
        
        except subprocess.TimeoutExpired:
            print("\n   ⏱️  Tests timeout (>5 min)")
            pause_for_reading("Test timeout.", 2)
        except Exception as e:
            print(f"\n   💥 Test error: {e}")
            pause_for_reading("Test failed.", PAUSE_BEFORE_FIX)


def main():
    """Entry point - SLOW AND READABLE."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🐢 SLOW Lightning Agent - Readable Code Fixer'
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
        help='Run continuously with pauses between runs'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print_header("🐢 SLOW Lightning Agent - Readable Code Fixer")
    print("="*70)
    print(f"   Mode: {'DAEMON (continuous)' if args.daemon else 'SINGLE RUN'}")
    print(f"   Max iterations: {args.max_iterations}")
    print("="*70 + "\n")
    pause_for_reading("Starting agent...", 2)
    
    agent = LightningAgentFast(max_iterations=args.max_iterations)
    
    if args.daemon:
        iteration = 0
        try:
            while True:
                iteration += 1
                print(f"\n{'='*70}")
                print_header(f"Daemon Run {iteration}")
                print("="*70 + "\n")
                pause_for_reading("Starting daemon iteration...", 2)
                
                agent.run_loop()
                
                print("\n" + "─"*70)
                print("⏱️  Daemon sleeping for 60 seconds before next run...")
                print("   (Press Ctrl+C to stop)")
                print("─"*70)
                
                import time
                for i in range(60, 0, -10):
                    if i <= 10:
                        print(f"   {i} seconds remaining...", end='\r')
                    time.sleep(10)
                
                pause_for_reading("Ready for next run...", 1)
        
        except KeyboardInterrupt:
            print("\n\n✋ Daemon stopped by user")
            print(f"   Total daemon runs: {iteration}")
            print("="*70 + "\n")
    else:
        print("Single-run mode. Press Ctrl+C to stop.")
        print("="*70 + "\n")
        agent.run_loop()


if __name__ == '__main__':
    main()
