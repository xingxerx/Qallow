#!/usr/bin/env python3
"""
Advanced Error Fixer for Qallow
================================

Detects specific error patterns and applies targeted fixes.
"""





from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = 1  # Build/run fails
    HIGH = 2      # Functional issues
    MEDIUM = 3    # Performance issues
    LOW = 4       # Warnings only


@dataclass
class ErrorInfo:
    """Information about an error."""
    category: str
    severity: ErrorSeverity
    error_text: str
    file_path: Optional[str]
    line_number: Optional[int]
    suggested_fix: str


class ErrorDetector:
    """Detects and categorizes errors."""
    
    PATTERNS = {
        'cuda_not_found': {
            'pattern': r'(CUDA|nvcc|cudaError|GPU.*not found)',
            'severity': ErrorSeverity.CRITICAL,
            'fix': 'install_cuda_toolkit'
        },
        'cuda_version_mismatch': {
            'pattern': r'(CUDA version.*mismatch|incompatible.*compute capability)',
            'severity': ErrorSeverity.HIGH,
            'fix': 'update_cuda_version'
        },
        'memory_error': {
            'pattern': r'(segmentation fault|SIGSEGV|out of memory|malloc failed)',
            'severity': ErrorSeverity.CRITICAL,
            'fix': 'increase_memory'
        },
        'undefined_reference': {
            'pattern': r'(undefined reference|symbol not found)',
            'severity': ErrorSeverity.CRITICAL,
            'fix': 'fix_linker_flags'
        },
        'missing_header': {
            'pattern': r'(No such file or directory|fatal error.*file not found)',
            'severity': ErrorSeverity.HIGH,
            'fix': 'install_missing_headers'
        },
        'compilation_error': {
            'pattern': r'(error:|syntax error|expected|token)',
            'severity': ErrorSeverity.CRITICAL,
            'fix': 'fix_compilation_error'
        },
        'runtime_assertion': {
            'pattern': r'(Assertion.*failed|assert)',
            'severity': ErrorSeverity.HIGH,
            'fix': 'investigate_assertion'
        },
        'phase_convergence': {
            'pattern': r'(phase.*convergence|failed to converge)',
            'severity': ErrorSeverity.MEDIUM,
            'fix': 'increase_phase_ticks'
        },
        'ethics_calculation': {
            'pattern': r'(ethics.*error|sustainability.*failed)',
            'severity': ErrorSeverity.MEDIUM,
            'fix': 'fix_ethics_calc'
        },
    }
    
    def detect(self, output: str) -> List[ErrorInfo]:
        """Detect errors in output."""
        errors = []
        
        for line in output.split('\n'):
            for error_type, config in self.PATTERNS.items():
                if re.search(config['pattern'], line, re.IGNORECASE):
                    error_info = ErrorInfo(
                        category=error_type,
                        severity=config['severity'],
                        error_text=line.strip(),
                        file_path=self._extract_file_path(line),
                        line_number=self._extract_line_number(line),
                        suggested_fix=config['fix']
                    )
                    errors.append(error_info)
        
        return errors
    
    def _extract_file_path(self, line: str) -> Optional[str]:
        """Extract file path from error line."""
        match = re.search(r'(\S+\.[chp]+):', line)
        return match.group(1) if match else None
    
    def _extract_line_number(self, line: str) -> Optional[int]:
        """Extract line number from error line."""
        match = re.search(r':(\d+):', line)
        return int(match.group(1)) if match else None


class ErrorFixer:
    """Applies fixes for detected errors."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.detector = ErrorDetector()
    
    def fix_error(self, error_info: ErrorInfo) -> bool:
        """Apply fix for error."""
        fix_method = getattr(self, f'_fix_{error_info.suggested_fix}', None)
        
        if not fix_method:
            logger.warning(f"No fix method for {error_info.suggested_fix}")
            return False
        
        try:
            logger.info(f"Applying fix: {error_info.suggested_fix}")
            return fix_method(error_info)
        except Exception as e:
            logger.error(f"Fix failed: {e}")
            return False
    
    def _run_command(self, cmd: str, cwd: Optional[Path] = None) -> Tuple[bool, str]:
        """Run a shell command."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd or self.project_root,
                timeout=300
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    # Fix methods
    
    def _fix_install_cuda_toolkit(self, error: ErrorInfo) -> bool:
        """Install CUDA toolkit."""
        logger.info("Attempting to install CUDA toolkit...")
        
        # Check if nvcc is available
        success, _ = self._run_command("which nvcc")
        if success:
            logger.info("CUDA toolkit already installed")
            return True
        
        # Ubuntu/Debian
        success, _ = self._run_command("apt-cache search cuda-toolkit", Path("/"))
        if success:
            logger.info("CUDA available in apt. Install with: sudo apt install nvidia-cuda-toolkit")
            return False
        
        return False
    
    def _fix_update_cuda_version(self, error: ErrorInfo) -> bool:
        """Update CUDA version."""
        logger.info("Checking CUDA version compatibility...")
        
        # Get current CUDA version
        success, output = self._run_command("nvcc --version")
        if success:
            logger.info(f"Current CUDA: {output}")
            # Recommend compatible version
            logger.info("Recommendation: Update CMake CUDA_ARCHITECTURES flags")
            return True
        
        return False
    
    def _fix_increase_memory(self, error: ErrorInfo) -> bool:
        """Increase memory limits."""
        logger.info("Applying memory fix...")
        
        # Set memory limit environment variables
        os.environ['QALLOW_MEMORY_LIMIT'] = '16384'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        # Rebuild with memory flags
        success, _ = self._run_command(
            "cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS_RELWITHDEBINFO='-O2 -g'"
        )
        
        return success
    
    def _fix_fix_linker_flags(self, error: ErrorInfo) -> bool:
        """Fix linker flags."""
        logger.info("Fixing linker flags...")
        
        # Common linker fixes
        cmds = [
            "rm -rf build",
            "cmake -S . -B build -DCMAKE_EXE_LINKER_FLAGS=-Wl,--as-needed",
            "cmake --build build --parallel",
        ]
        
        for cmd in cmds:
            success, output = self._run_command(cmd)
            if not success:
                logger.error(f"Command failed: {cmd}")
                return False
        
        return True
    
    def _fix_install_missing_headers(self, error: ErrorInfo) -> bool:
        """Install missing header files."""
        logger.info("Installing missing headers...")
        
        # Try to install common dev packages
        packages = ['libssl-dev', 'libcurl4-openssl-dev', 'libsqlite3-dev']
        
        for pkg in packages:
            self._run_command(f"apt-get install -y {pkg}", Path("/"))
        
        return True
    
    def _fix_fix_compilation_error(self, error: ErrorInfo) -> bool:
        """Fix compilation errors."""
        logger.info(f"Fixing compilation error in {error.file_path}...")
        
        if error.file_path and error.line_number:
            logger.info(f"Error at {error.file_path}:{error.line_number}")
            logger.info(f"Error text: {error.error_text}")
        
        # Generic compilation fix: rebuild
        return self._rebuild()
    
    def _fix_investigate_assertion(self, error: ErrorInfo) -> bool:
        """Investigate assertion failure."""
        logger.info("Assertion failed - enabling debug mode...")
        
# [REVIEWED] # [REVIEWED] # [REVIEWED]         os.environ['QALLOW_ASSERT_DEBUG'] = '1'
# [REVIEWED] # [REVIEWED] # [REVIEWED]         os.environ['QALLOW_DEBUG_LEVEL'] = '2'
        
        # Rebuild with debug symbols
        success, _ = self._run_command(
            "cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug"
        )
        
        return success
    
    def _fix_increase_phase_ticks(self, error: ErrorInfo) -> bool:
        """Increase phase ticks for convergence."""
        logger.info("Increasing phase ticks for convergence...")
        
        # Will be applied at runtime
        logger.info("Recommendation: Run with --integrate-ticks=200 or higher")
        
        return True
    
    def _fix_fix_ethics_calc(self, error: ErrorInfo) -> bool:
        """Fix ethics calculation."""
        logger.info("Enabling ethics debug mode...")
        
# [REVIEWED] # [REVIEWED] # [REVIEWED]         os.environ['QALLOW_ETHICS_DEBUG'] = '1'
        os.environ['QALLOW_LOG_ETHICS'] = '1'
        
        # Rebuild
        return self._rebuild()
    
    def _rebuild(self) -> bool:
        """Generic rebuild."""
        cmds = [
            "cmake -S . -B build",
            "cmake --build build --parallel $(nproc)",
        ]
        
        for cmd in cmds:
            success, output = self._run_command(cmd)
            if not success:
                logger.error(f"Rebuild failed: {cmd}")
                return False
        
        return True


class ProactiveOptimizer:
    """Proactively optimizes settings based on metrics."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
    
    def optimize_for_metrics(self, metrics: Dict[str, float]) -> List[str]:
        """Suggest optimizations based on current metrics."""
        suggestions = []
        
        coherence = metrics.get('avg_coherence', 0)
        ethics = metrics.get('ethics_total', 0)
        stability = metrics.get('stability', 0.5)
        
        # Low coherence
        if coherence < 0.7:
            suggestions.append("--integrate-phase13-ticks=200")
            suggestions.append("--integrate-phase14-k=0.002")
        
        # Low ethics
        if ethics < 2.0:
            suggestions.append("--integrate-ethics-weight=1.5")
        
        # Low stability
        if stability < 0.6:
            suggestions.append("--integrate-elasticity-damping=0.8")
        
        return suggestions


# For use in dataclass
