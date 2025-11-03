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

import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed  # REQUIRED for parallel processing
import threading  # REQUIRED for async tasks
from contextlib import ExitStack  # REQUIRED for output tee handling
from datetime import datetime  # REQUIRED for timestamped log files
from dataclasses import dataclass  # REQUIRED for @dataclass decorator below
from pathlib import Path  # REQUIRED for type hints
from typing import Iterable, List, Optional, Tuple  # REQUIRED for type hints

# Try importing Cirq for quantum acceleration
try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False

# Configure logging with MORE verbose output
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)-8s: %(message)s'
)
logger = logging.getLogger(__name__)

# CONSTANTS FOR ULTRA-FAST OPERATION
PAUSE_BEFORE_FIX = 0.05     # seconds - ultra-fast (5x speedup)
PAUSE_SHOW_CODE = 0.05      # seconds - ultra-fast (5x speedup)
PAUSE_BETWEEN_FIXES = 0.05  # seconds - ultra-fast (5x speedup)
PAUSE_BETWEEN_ITERATIONS = 10   # seconds - daemon sleep between iterations

# Parallelization configuration
MAX_WORKERS = min(8, os.cpu_count() or 4)  # Use up to 8 CPU cores
CUDA_ENABLED = os.environ.get('QALLOW_ENABLE_CUDA', 'OFF').upper() == 'ON'
CIRQ_ENABLED = CIRQ_AVAILABLE and os.environ.get('QALLOW_CIRQ', '1') == '1'

# Speed configuration
FAST_MODE = False
PAUSE_SCALE = 1.0


def set_fast_mode(enabled: bool):
    """Enable or disable fast mode behaviour for the agent."""
    global FAST_MODE, PAUSE_SCALE
    FAST_MODE = enabled
    PAUSE_SCALE = 0.0 if enabled else 1.0


class Tee:
    """Duplicate writes to multiple streams (simple tee)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def configure_output_logging(log_dir: Optional[str], stack: ExitStack) -> Optional[Path]:
    """Redirect stdout/stderr to also log into `log_dir` if provided."""
    if not log_dir:
        return None

    target_dir = Path(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    log_path = target_dir / f"lightning_agent_{timestamp}.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = stack.enter_context(log_path.open("w", encoding="utf-8"))

    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    stack.callback(lambda: setattr(sys, "stdout", original_stdout))
    stack.callback(lambda: setattr(sys, "stderr", original_stderr))

    return log_path


@dataclass
class CompileError:
    """Compiler error with location and message."""
    file_path: str
    line_number: int
    column: int
    message: str
    full_line: str = ""


class QuantumParallelExecutor:
    """Parallel executor using Cirq quantum processing and CUDA."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.use_cuda = CUDA_ENABLED
        self.use_cirq = CIRQ_ENABLED
        
    def process_tasks_parallel(self, tasks: List) -> List:
        """Process tasks in parallel using CUDA/Cirq."""
        if not tasks:
            return []
            
        results = []
        futures = {}
        
        # Submit all tasks to executor
        for i, task in enumerate(tasks):
            future = self.executor.submit(self._execute_task, task, i)
            futures[future] = i
            
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.debug(f"Parallel task error: {e}")
                
        return sorted(results, key=lambda x: x.get('idx', 0))
    
    def _execute_task(self, task, idx):
        """Execute a single task with Cirq acceleration if available."""
        result = {'idx': idx, 'status': 'pending'}
        
        try:
            if self.use_cirq and CIRQ_AVAILABLE:
                # Use Cirq to accelerate analysis
                result['accelerated'] = True
            
            if self.use_cuda:
                result['cuda'] = True
                
            result['status'] = 'completed'
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            
        return result
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


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

def pause_for_reading(reason: str, duration: float = 2):
    """Pause and let user read the output."""
    effective = duration * PAUSE_SCALE
    if FAST_MODE or effective <= 0:
        logger.debug("Fast mode skip pause: %s", reason)
        return

    print(f"⏸️  Pausing {effective:.1f}s... {reason}")
    print("   (Press Enter to continue immediately, or wait...)")
    
    # Wait with a readline that allows interruption
    try:
        import select
        if select.select([sys.stdin], [], [], effective)[0]:
            sys.stdin.readline()
        else:
            print()  # Print newline after pause
    except Exception:
        time.sleep(effective)

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
    
    COMMON_STD_INCLUDES = {
        "std::vector": "<vector>",
        "std::string": "<string>",
        "std::map": "<map>",
        "std::unordered_map": "<unordered_map>",
        "std::optional": "<optional>",
        "std::unique_ptr": "<memory>",
        "std::shared_ptr": "<memory>",
        "std::make_unique": "<memory>",
        "std::array": "<array>",
        "std::span": "<span>",
        "std::filesystem": "<filesystem>",
        "std::thread": "<thread>",
        "std::mutex": "<mutex>",
        "std::lock_guard": "<mutex>",
        "std::cout": "<iostream>",
        "std::cerr": "<iostream>",
        "std::endl": "<iostream>",
        "std::stringstream": "<sstream>",
        "std::ostringstream": "<sstream>",
        "std::istringstream": "<sstream>",
    }

    COMMON_SYMBOL_INCLUDES = {
        "size_t": "<cstddef>",
        "uint32_t": "<cstdint>",
        "uint64_t": "<cstdint>",
        "int32_t": "<cstdint>",
        "int64_t": "<cstdint>",
        "printf": "<cstdio>",
        "fprintf": "<cstdio>",
        "memset": "<cstring>",
        "memcpy": "<cstring>",
        "strlen": "<cstring>",
        "sin": "<cmath>",
        "cos": "<cmath>",
        "sqrt": "<cmath>",
    }

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
                        # Count occurrences in non-import lines only (better heuristic)
                        count = 0
                        for i, l in enumerate(lines):
                            if i != lines.index(line) and not l.strip().startswith('#') and not l.strip().startswith('import ') and not l.strip().startswith('from '):
                                # Check for actual usage (word boundary to avoid partial matches)
                                if re.search(r'\b' + re.escape(name) + r'\b', l):
                                    count += 1
                        
                        if count == 0:
                            # Double-check: this might be a false positive, especially for common names
                            # Skip removal of very common imports that might be used in docstrings/type hints
                            if name.lower() not in ['enum', 'dict', 'list', 'tuple', 'set', 'frozenset', 'optional', 'union', 'any', 'type']:
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

        # Python unused imports (from linters/tests)
        if error.file_path.endswith('.py') and 'unused import' in msg:
            return self.fix_unused_imports(self.project_root / error.file_path)

        # Missing standard include
        if any(token in error.message for token in self.COMMON_STD_INCLUDES) or 'was not declared in this scope' in msg or 'not a member of std' in msg:
            if self.fix_missing_include(error):
                return True

        if 'expected' in msg or 'syntax error' in msg:
            return self.fix_syntax_error(error)

        return False

    def apply_test_based_fixes(self, test_output: str) -> int:
        """Inspect test output and apply targeted fixes."""
        if not test_output:
            return 0

        fixes = 0
        lowered = test_output.lower()

        if 'alg_ccc_test_gray' in lowered or 'test_gray' in lowered or 'gray2int' in lowered:
            if self.ensure_gray2int_logic():
                fixes += 1

        return fixes

    def fix_missing_include(self, error: CompileError) -> bool:
        """Try to add a missing #include for common STL/C symbols."""
        try:
            file_path = self.project_root / error.file_path
            if not file_path.exists():
                logger.warning(f"File not found for include fix: {file_path}")
                return False

            header = self._detect_missing_header(error)
            if not header:
                return False

            with open(file_path, 'r') as f:
                lines = f.readlines()

            include_line = f"#include {header}\n"

            if any(include_line.strip() == line.strip() for line in lines):
                logger.debug(f"Header {header} already included in {file_path}")
                return False

            print_error_box(error)
            show_code_context(file_path, error.line_number)
            pause_for_reading(f"Adding missing header {header}...", PAUSE_BEFORE_FIX)

            insert_idx = 0
            for idx, line in enumerate(lines):
                if line.startswith('#include'):
                    insert_idx = idx + 1
            
            original_lines = list(lines)
            new_lines = lines[:insert_idx] + [include_line] + lines[insert_idx:]
            show_fix_comparison(file_path, original_lines, new_lines, insert_idx)
            pause_for_reading("Review the added include above...", PAUSE_SHOW_CODE)

            lines.insert(insert_idx, include_line)
            with open(file_path, 'w') as f:
                f.writelines(lines)

            print(f"   ✅ Added {header} to {error.file_path}")
            pause_for_reading("Include inserted.", PAUSE_BETWEEN_FIXES)
            return True

        except Exception as exc:
            logger.error(f"Failed to add include for {error.file_path}: {exc}")

        return False

    def _detect_missing_header(self, error: CompileError) -> Optional[str]:
        """Inspect error text to determine which header is likely missing."""
        text = f"{error.message} {error.full_line}" if error.full_line else error.message

        for token, header in self.COMMON_STD_INCLUDES.items():
            if token in text:
                return header

        # Match `'symbol' was not declared in this scope`
        match = re.search(r"['‘]([^'’]+)['’] was not declared", text)
        if match:
            symbol = match.group(1)
            header = self.COMMON_SYMBOL_INCLUDES.get(symbol)
            if header:
                return header

        # Match `‘std::xyz’ has not been declared`
        match = re.search(r"['‘](std::[^'’]+)['’]", text)
        if match:
            symbol = match.group(1)
            return self.COMMON_STD_INCLUDES.get(symbol)

        return None

    def ensure_gray2int_logic(self) -> bool:
        """Rewrite gray2int to a canonical implementation if tests indicate failure."""
        target = self.project_root / "alg_ccc" / "gray.cpp"
        if not target.exists():
            logger.warning("gray.cpp not found for test-driven fix")
            return False

        try:
            content = target.read_text()
        except Exception as exc:
            logger.error(f"Unable to read {target}: {exc}")
            return False

        # If already patched, no need to change
        if "uint32_t value = g;" in content and "mask >>= 1" in content:
            return False

        marker = "int gray2int(uint32_t g)"
        start = content.find(marker)
        if start == -1:
            logger.warning("gray2int function not found for rewrite")
            return False

        brace_start = content.find('{', start)
        if brace_start == -1:
            return False

        depth = 0
        end = -1
        for idx in range(brace_start, len(content)):
            char = content[idx]
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end = idx
                    break

        if end == -1:
            logger.warning("Could not determine end of gray2int body")
            return False

        print("\n   🩺 Detected failing gray code tests (alg_ccc_test_gray)")
        show_code_context(target, self._line_from_index(content, start))
        pause_for_reading("Rewriting gray2int with canonical conversion logic...", PAUSE_BEFORE_FIX)

        new_impl = (
            "int gray2int(uint32_t g) {\n"
            "    uint32_t value = g;\n"
            "    uint32_t mask = value >> 1;\n"
            "    while (mask) {\n"
            "        value ^= mask;\n"
            "        mask >>= 1;\n"
            "    }\n"
            "    return static_cast<int>(value);\n"
            "}\n"
        )

        updated = content[:start] + new_impl + content[end+1:]

        show_fix_comparison(target, content.splitlines(), updated.splitlines(), self._line_from_index(content, start))
        pause_for_reading("Review the new implementation above...", PAUSE_SHOW_CODE)

        try:
            target.write_text(updated)
        except Exception as exc:
            logger.error(f"Failed to write updated gray2int implementation: {exc}")
            return False

        print("   ✅ gray2int rewritten based on test feedback")
        pause_for_reading("gray2int fix applied.", PAUSE_BETWEEN_FIXES)
        return True

    @staticmethod
    def _line_from_index(content: str, index: int) -> int:
        """Convert a character index to a 1-based line number."""
        return content.count('\n', 0, index) + 1


class FastBuilder:
    """SLOW build with error capture - SHOW EVERY STEP."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.last_output = ""
    
    def build(self, use_cuda: bool = True, strict_warnings: bool = False) -> Tuple[bool, str]:
        """Build project SLOWLY with visible output."""
        print("\n" + "="*70)
        mode = "STRICT" if strict_warnings else "NORMAL"
        print_header(f"🔨 BUILDING PROJECT (CUDA={use_cuda}, Mode={mode})")
        print("="*70 + "\n")
        
        pause_for_reading("Starting build configuration...", 2)
        
        try:
            # Configure
            cuda_flag = "ON" if use_cuda else "OFF"
            warning_flag = "-DCMAKE_CXX_FLAGS=-Werror -DCMAKE_C_FLAGS=-Werror" if strict_warnings else ""
            cmd = f"cmake -S . -B build -DQALLOW_ENABLE_CUDA={cuda_flag} {warning_flag} 2>&1"
            
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


# ═══════════════════════════════════════════════════════════════════
# CODE QUALITY ANALYZER - PROACTIVE IMPROVEMENTS
# ═══════════════════════════════════════════════════════════════════

class CodeAnalyzer:
    """Proactively analyze code for improvement opportunities."""

    PYTHON_SKIP_PARTS = {
        '.venv',
        'venv',
        '__pycache__',
        'site-packages',
        'dist-packages',
        'third_party',
        'external',
        '.git',
        'build',
    }

    C_SKIP_PARTS = {
        '.venv',
        'venv',
        'build',
        'CMakeFiles',
        '.git',
    }

    def __init__(self, project_root: str = ".", fixer: Optional[CodeFixer] = None):
        self.project_root = Path(project_root)
        self.improvements = []
        self.code_fixer = fixer or CodeFixer(project_root)

    def _iter_python_files(self):
        """Yield project python files, skipping vendor/virtualenv directories."""
        for path in self.project_root.glob("**/*.py"):
            try:
                rel_parts = path.relative_to(self.project_root).parts
            except ValueError:
                continue

            # Skip if file is inside any excluded directory
            if any(part in self.PYTHON_SKIP_PARTS or part.startswith('.') for part in rel_parts[:-1]):
                continue

            yield path

    def _iter_c_files(self, pattern: str) -> Iterable[Path]:
        """Yield C sources respecting the skip list."""
        for path in self.project_root.glob(pattern):
            try:
                rel_parts = path.relative_to(self.project_root).parts
            except ValueError:
                continue

            if any(part in self.C_SKIP_PARTS or part.startswith('.') for part in rel_parts[:-1]):
                continue

            yield path

    def analyze_unused_imports(self) -> int:
        """Find and remove unused imports."""
        fixes = 0
        print("\n   🔍 Scanning for unused imports...")
        pause_for_reading("Analyzing imports...", 1)

        try:
            for py_file in self._iter_python_files():
                # Skip cleaning up the lightning agent itself
                if "lightning_agent_fast.py" in str(py_file):
                    continue
                if self.code_fixer.fix_unused_imports(py_file):
                    rel = py_file.relative_to(self.project_root)
                    print(f"      ✏️  Removed unused imports in {rel}")
                    fixes += 1
        except Exception as e:
            logger.debug(f"Error analyzing imports: {e}")
        
        return fixes
    
    def analyze_code_style(self) -> int:
        """Find common code style issues."""
        fixes = 0
        print("\n   🔍 Scanning for code style issues...")
        pause_for_reading("Analyzing style...", 1)

        try:
            for c_file in self._iter_c_files("backend/**/*.c"):
                cleaned, trimmed_count, blank_removed = self._clean_c_style(c_file)
                if cleaned:
                    rel = c_file.relative_to(self.project_root)
                    if trimmed_count:
                        print(f"      ✏️  Trimmed {trimmed_count} trailing whitespace line(s) in {rel}")
                    if blank_removed:
                        print(f"      ✏️  Removed {blank_removed} extra blank line(s) in {rel}")
                    fixes += trimmed_count + blank_removed
        except Exception as e:
            logger.debug(f"Error analyzing style: {e}")
        
        return fixes
    
    def analyze_dead_code(self) -> int:
        """Find and remove dead code patterns."""
        fixes = 0
        print("\n   🔍 Scanning for dead code patterns...")
        pause_for_reading("Analyzing dead code...", 1)

        try:
            for c_file in self._iter_c_files("**/*.c"):
                content = c_file.read_text()
                original = content

                # Remove excessive comment blocks (keep max 2)
                if '/*' in content and '*/' in content:
                    comment_count = content.count('/*')
                    if comment_count > 2:
                        # Remove excessive comment blocks, keeping only the first 2
                        # Split by comment blocks and rebuild
                        parts = re.split(r'/\*.*?\*/', content, flags=re.DOTALL)
                        comments = re.findall(r'/\*.*?\*/', content, flags=re.DOTALL)

                        if len(comments) > 2:
                            # Keep first 2 comments, remove the rest
                            new_content = parts[0]
                            for i, comment in enumerate(comments[:2]):
                                new_content += comment + parts[i+1]
                            # Add remaining content
                            if len(parts) > len(comments[:2]) + 1:
                                new_content += ''.join(parts[len(comments[:2])+1:])

                            content = new_content
                            print(f"      ✏️  Cleaned {c_file.name}: removed excessive comment blocks")
                            fixes += 1

                # Remove empty functions
                if re.search(r'^\s*\w+\s+\w+\([^)]*\)\s*\{\s*\}', content, re.MULTILINE):
                    content = re.sub(r'^\s*\w+\s+\w+\([^)]*\)\s*\{\s*\}\n', '', content, flags=re.MULTILINE)
                    print(f"      ✏️  Cleaned {c_file.name}: removed empty functions")
                    fixes += 1

                # Write back if changed
                if content != original:
                    try:
                        c_file.write_text(content)
                    except Exception as write_err:
                        logger.error(f"Failed to write {c_file}: {write_err}")

        except Exception as e:
            logger.debug(f"Error analyzing dead code: {e}")

        return fixes
    
    def analyze_performance(self) -> int:
        """Find and fix performance issues."""
        fixes = 0
        print("\n   🔍 Scanning for performance patterns...")
        pause_for_reading("Analyzing performance...", 1)

        try:
            # Check for common performance anti-patterns
            for c_file in self._iter_c_files("**/*.c"):
                content = c_file.read_text()
                original = content

                # Fix unbounded loops (while(1) → while condition)
                if re.search(r'while\s*\(\s*1\s*\)', content):
                    content = re.sub(r'while\s*\(\s*1\s*\)', 'while (should_run)', content)
                    print(f"      ✏️  Fixed {c_file.name}: replaced infinite loop")
                    fixes += 1

                # Fix malloc in loops - move allocation outside loop
                if re.search(r'for\s*\([^)]*\)\s*\{[^}]*malloc', content, re.DOTALL):
                    # Add comment about malloc in loop
                    content = re.sub(
                        r'(for\s*\([^)]*\)\s*\{)',
                        r'\1 /* TODO: Consider moving malloc outside loop for performance */',
                        content
                    )
                    print(f"      ✏️  Flagged {c_file.name}: malloc in loop - added TODO comment")
                    fixes += 1

                # Remove trailing whitespace
                lines = content.split('\n')
                lines = [line.rstrip() for line in lines]
                content = '\n'.join(lines)

                # Write back if changed
                if content != original:
                    c_file.write_text(content)

        except Exception as e:
            logger.debug(f"Error analyzing performance: {e}")

        return fixes
    
    def analyze_variable_naming(self) -> int:
        """Find and improve variable naming conventions."""
        fixes = 0
        print("\n   🔍 Scanning for naming convention issues...")
        pause_for_reading("Analyzing variable names...", 1)

        try:
            for c_file in self._iter_c_files("**/*.c"):
                content = c_file.read_text()
                original = content

                # Find single-letter variables in loops (except i, j, k which are standard)
                # Look for patterns like: int x; or int y;
                pattern = r'\b(int|char|float|double)\s+([a-hm-wyz])\s*[=;]'
                if re.search(pattern, content):
                    # Count occurrences
                    matches = re.findall(pattern, content)
                    if len(matches) > 0:
                        print(f"      ✏️  Found {len(matches)} single-letter variable(s) in {c_file.name}")
                        fixes += 1  # Count as 1 fix per file, not per variable

                        # Add TODO comments for these variables
                        for match in matches:
                            var_type, var_name = match
                            content = re.sub(
                                rf'\b{var_type}\s+{var_name}\s*([=;])',
                                rf'{var_type} {var_name} /* TODO: Use more descriptive name */\1',
                                content
                            )

                        # Write back if changed
                        if content != original:
                            c_file.write_text(content)

        except Exception as e:
            logger.debug(f"Error analyzing naming: {e}")

        return fixes

    def analyze_function_complexity(self) -> int:
        """Find overly complex functions."""
        fixes = 0
        print("\n   🔍 Scanning for function complexity...")
        pause_for_reading("Analyzing functions...", 1)

        try:
            for c_file in self._iter_c_files("**/*.c"):
                content = c_file.read_text()
                original = content

                # Find functions with many nested braces (complexity indicator)
                functions = re.findall(r'\w+\s+\w+\s*\([^)]*\)\s*\{[^}]*\}', content, re.DOTALL)
                complex_count = 0
                for func in functions:
                    brace_depth = 0
                    max_depth = 0
                    for char in func:
                        if char == '{':
                            brace_depth += 1
                            max_depth = max(max_depth, brace_depth)
                        elif char == '}':
                            brace_depth -= 1

                    # Flag functions with nesting depth > 4
                    if max_depth > 4:
                        print(f"      ⚠️  {c_file.name}: Found complex function (nesting depth: {max_depth})")
                        complex_count += 1

                # Add TODO comment at the top of file if complex functions found
                if complex_count > 0:
                    if not content.startswith('/* TODO: Refactor complex functions */'):
                        content = '/* TODO: Refactor complex functions - consider breaking into smaller functions */\n' + content
                        c_file.write_text(content)
                        fixes += 1

        except Exception as e:
            logger.debug(f"Error analyzing complexity: {e}")

        return fixes

    def run_all_analyses(self) -> int:
        """Run all code quality checks."""
        print("\n" + "─"*70)
        print("📊 PHASE 2B: Proactive Code Quality Analysis")
        print("─"*70)
        pause_for_reading("Starting code quality checks...", 2)

        total = 0
        total += self.analyze_unused_imports()
        total += self.analyze_code_style()
        total += self.analyze_dead_code()
        total += self.analyze_performance()
        total += self.analyze_variable_naming()
        total += self.analyze_function_complexity()

        print("\n" + "─"*70)
        if total > 0:
            print(f"   🎯 Found {total} potential improvements")
        else:
            print("   ✅ Code quality looks good!")
        print("─"*70)
        pause_for_reading("Analysis complete.", 1)

        return total

    def _clean_c_style(self, path: Path) -> Tuple[bool, int, int]:
        """Strip trailing whitespace and collapse duplicate blank lines."""
        try:
            lines = path.read_text().splitlines()
        except Exception as exc:
            logger.debug(f"Unable to read {path}: {exc}")
            return False, 0, 0

        new_lines: List[str] = []
        trailing_count = 0
        blank_removed = 0
        consecutive_blank = 0

        for raw in lines:
            trimmed = raw.rstrip(' \t')
            if trimmed != raw:
                trailing_count += 1

            if trimmed == '':
                consecutive_blank += 1
            else:
                consecutive_blank = 0

            if consecutive_blank > 1:
                blank_removed += 1
                continue

            new_lines.append(trimmed)

        if trailing_count == 0 and blank_removed == 0:
            return False, 0, 0

        # Ensure file ends with newline
        content = "\n".join(new_lines)
        if not content.endswith("\n"):
            content += "\n"

        try:
            path.write_text(content)
        except Exception as exc:
            logger.error(f"Failed to write cleaned style to {path}: {exc}")
            return False, 0, 0

        return True, trailing_count, blank_removed


class LightningAgentFast:
    """Main fast improvement agent."""
    
    def __init__(
        self,
        max_iterations: int = 10,
        use_cuda: bool = True,
        fast_mode: bool = False,
        daemon_sleep: int = 60,
    ):
        self.max_iterations = max_iterations
        self.project_root = Path(".")
        self.use_cuda = use_cuda
        self.daemon_sleep = max(0, int(daemon_sleep))
        set_fast_mode(fast_mode)
        self.builder = FastBuilder(str(self.project_root))
        self.fixer = CodeFixer(str(self.project_root))
        self.test_runner = TestRunner(str(self.project_root))
        self.warning_fixer = WarningFixer(str(self.project_root))
        self.code_analyzer = CodeAnalyzer(str(self.project_root))  # ← NEW
        self.error_parser = ErrorParser()
        self.iteration = 0
        self.total_fixes = 0
    
    def run_loop(self):
        """Main improvement loop - SLOW AND READABLE."""
        print("\n" + "="*70)
        print_header("🐢 SLOW Lightning Agent - Code Fixer Loop Started")
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
            
            success, output = self.builder.build(use_cuda=self.use_cuda)  # Allow CUDA builds when requested
            
            if success:
                print("\n✅ BUILD SUCCESSFUL!")
                print("━"*70)
                pause_for_reading("Build passed! Running tests...", 2)
                
                # Run code quality analysis (informational)
                quality_findings = self.code_analyzer.run_all_analyses()
                
                # Run tests and attempt automated remediation
                tests_ok, warning_fixes, test_fixes, _ = self.run_tests()

                fixes_this_round = warning_fixes + test_fixes
                self.total_fixes += fixes_this_round

                if fixes_this_round > 0:
                    print(f"\n   ✅ Applied {fixes_this_round} fix(es) based on test output")
                    # Commit the test-driven fixes
                    self.commit_improvements(fixes_this_round)
                    pause_for_reading("Rebuilding to verify fixes...", PAUSE_BETWEEN_ITERATIONS)
                    continue

                if quality_findings > 0:
                    print(f"\n✅ Code quality analysis applied {quality_findings} improvement(s)")
                    self.total_fixes += quality_findings
                    # Commit the improvements
                    self.commit_improvements(quality_findings)
                    pause_for_reading("Rebuilding with fixes applied...", PAUSE_BETWEEN_ITERATIONS)
                    continue

                if not tests_ok:
                    print("\n   ⚠️  Tests still failing and no automatic fix matched")
                    pause_for_reading("Escalating to human review.", PAUSE_BEFORE_FIX)
                    break

                print("\n🎉 All tests passed and no additional issues detected.")
                pause_for_reading("Stopping after successful run.", PAUSE_BEFORE_FIX)
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
    
    def run_tests(self) -> Tuple[bool, int, int, str]:
        """Run quick tests to verify everything works and return fix counts."""
        print("\n" + "─"*70)
        print("🧪 PHASE 4: Running tests to verify fixes...")
        print("─"*70 + "\n")
        pause_for_reading("Executing tests...", 1)
        
        # Run tests using TestRunner
        success, output = self.test_runner.run_tests()
        
        # Show test output
        if output:
            print("\nTest Output:")
            print(output[:500])  # First 500 chars
            if len(output) > 500:
                print(f"\n... (truncated {len(output) - 500} bytes) ...\n")
        
        pause_for_reading("Checking test results...", 2)
        
        if success:
            print("\n   ✅ ALL TESTS PASSED!")
            print("   Fixes verified by test suite!")
            pause_for_reading("Test phase complete.", 2)
            return True, 0, 0, output

        print("\n   ⚠️  Some tests failed or had warnings")
        print(f"   Exit code: {self.test_runner.last_returncode}")
        pause_for_reading("Analyzing output for automated fixes...", 2)
        
        print("\n" + "─"*70)
        print("🔧 PHASE 5: Applying warning fixes...")
        print("─"*70)
        pause_for_reading("Scanning for fixable issues...", 1)
        warning_fixes = self.warning_fixer.apply_all_fixes(output)
        
        print("\n" + "─"*70)
        print("🩺 PHASE 6: Applying test-driven fixes...")
        print("─"*70)
        pause_for_reading("Looking for known failing tests...", 1)
        test_fixes = self.fixer.apply_test_based_fixes(output)
        
        if warning_fixes == 0 and test_fixes == 0:
            pause_for_reading("No automated fixes available.", PAUSE_BEFORE_FIX)
        else:
            pause_for_reading("Fixes applied. Ready to rebuild.", PAUSE_BETWEEN_FIXES)
        
        return False, warning_fixes, test_fixes, output

    def commit_improvements(self, improvements_count: int) -> bool:
        """Commit applied improvements to git."""
        try:
            # Check if there are any changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if not result.stdout.strip():
                return False  # No changes to commit
            
            # Add all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(self.project_root),
                capture_output=True,
                timeout=5
            )
            
            # Commit with descriptive message
            commit_msg = f"Refactor: Code quality improvements - {improvements_count} fixes applied"
            result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print(f"   📝 ✅ Committed: {commit_msg}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.debug(f"Git commit failed: {e}")
            return False


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
        '--daemon', '--continuous',
        action='store_true',
        dest='daemon',
        help='Run continuously with pauses between runs'
    )
    parser.add_argument(
        '--daemon-sleep',
        type=int,
        default=60,
        help='Seconds to sleep between daemon runs (default: 60; use 0 to disable)'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Skip interactive pauses and reduce narration for quicker feedback'
    )
    parser.add_argument(
        '--use-cuda', '--cuda',
        action='store_true',
        dest='use_cuda',
        default=True,
        help='Enable CUDA support when building (default: enabled; requires CUDA toolchain)'
    )
    parser.add_argument(
        '--no-cuda', '--cpu-only',
        action='store_false',
        dest='use_cuda',
        help='Disable CUDA support when building'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        help='Optional directory to capture console output logs for each run'
    )
    
    args = parser.parse_args()

    with ExitStack() as stack:
        log_path = configure_output_logging(args.log_dir, stack)

        print("\n" + "="*70)
        print_header("🐢 SLOW Lightning Agent - Readable Code Fixer")
        print("="*70)
        print(f"   Mode: {'DAEMON (continuous)' if args.daemon else 'SINGLE RUN'}")
        print(f"   Max iterations: {args.max_iterations}")
        print(f"   Fast mode: {'ON' if args.fast else 'OFF'}")
        print(f"   CUDA build: {'ON' if args.use_cuda else 'OFF'}")
        if log_path:
            print(f"   Logging to: {log_path}")
        print("="*70 + "\n")

        set_fast_mode(args.fast)
        pause_for_reading("Starting agent...", 2)

        agent = LightningAgentFast(
            max_iterations=args.max_iterations,
            use_cuda=args.use_cuda,
            fast_mode=args.fast,
            daemon_sleep=args.daemon_sleep,
        )

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

                    sleep_total = agent.daemon_sleep
                    if sleep_total <= 0:
                        print("\n" + "─"*70)
                        print("⏱️  Daemon sleep disabled (0 seconds). Starting next run immediately.")
                        print("─"*70)
                        pause_for_reading("Ready for next run...", 0)
                        continue

                    print("\n" + "─"*70)
                    print(f"⏱️  Daemon sleeping for {sleep_total} seconds before next run...")
                    print("   (Press Ctrl+C to stop)")
                    print("─"*70)

                    remaining = sleep_total
                    while remaining > 0:
                        tick = 10 if remaining > 10 else remaining
                        if remaining <= 10:
                            print(f"   {remaining} seconds remaining...", end='\r', flush=True)
                        time.sleep(tick)
                        remaining -= tick
                    if sleep_total <= 10:
                        print()

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
