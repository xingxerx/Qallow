#!/usr/bin/env python3
"""
Recursive Improvement Loop Engine for Qallow
Feeds build output back into code improvements for continuous learning
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re

class BuildAnalyzer:
    """Analyzes build output for errors, warnings, and metrics"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.build_dir = self.workspace_root / "build"
        self.logs_dir = self.workspace_root / "data" / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def run_build(self, build_type: str = "unified") -> Tuple[int, str, str]:
        """Execute build and capture output"""
        print(f"[BUILD] Starting {build_type} build...")
        
        cmd = [
            "cmake",
            "--build", str(self.build_dir),
            "--parallel", "4",
            "--verbose"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=600
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Build timeout after 600 seconds"
    
    def parse_build_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse build output for errors, warnings, and metrics"""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "errors": [],
            "warnings": [],
            "metrics": {
                "total_files": 0,
                "compiled_files": 0,
                "failed_files": 0,
                "build_time": 0
            },
            "issues": []
        }
        
        # Extract errors
        error_pattern = r"error:\s*(.+?)(?:\n|$)"
        analysis["errors"] = re.findall(error_pattern, stderr + stdout)
        
        # Extract warnings
        warning_pattern = r"warning:\s*(.+?)(?:\n|$)"
        analysis["warnings"] = re.findall(warning_pattern, stderr + stdout)
        
        # Extract compilation stats
        compile_pattern = r"\[(\d+)%\]"
        matches = re.findall(compile_pattern, stdout)
        if matches:
            analysis["metrics"]["compiled_files"] = len(matches)
        
        # Categorize issues
        for error in analysis["errors"]:
            analysis["issues"].append({
                "type": "error",
                "message": error,
                "severity": "high"
            })
        
        for warning in analysis["warnings"]:
            analysis["issues"].append({
                "type": "warning",
                "message": warning,
                "severity": "medium"
            })
        
        return analysis
    
    def save_analysis(self, analysis: Dict) -> Path:
        """Save analysis to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.logs_dir / f"build_analysis_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"[ANALYSIS] Saved to {report_file}")
        return report_file


class CodeImprover:
    """Generates code improvements based on build analysis"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.improvements_dir = self.workspace_root / "improvement_reports"
        self.improvements_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_improvements(self, analysis: Dict) -> List[Dict]:
        """Generate improvement suggestions from analysis"""
        improvements = []
        
        # Analyze error patterns
        for issue in analysis.get("issues", []):
            if issue["type"] == "error":
                improvement = self._create_improvement(issue)
                if improvement:
                    improvements.append(improvement)
        
        return improvements
    
    def _create_improvement(self, issue: Dict) -> Dict:
        """Create specific improvement from issue"""
        message = issue["message"]
        
        # Pattern matching for common issues
        if "undefined reference" in message:
            return {
                "type": "linker_error",
                "issue": message,
                "action": "Check CMakeLists.txt dependencies",
                "priority": "high"
            }
        elif "implicit declaration" in message:
            return {
                "type": "missing_include",
                "issue": message,
                "action": "Add missing header file",
                "priority": "high"
            }
        elif "unused variable" in message:
            return {
                "type": "code_cleanup",
                "issue": message,
                "action": "Remove unused variable",
                "priority": "low"
            }
        
        return None
    
    def save_improvements(self, improvements: List[Dict]) -> Path:
        """Save improvement report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.improvements_dir / f"improvements_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Recursive Improvement Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            for i, imp in enumerate(improvements, 1):
                f.write(f"## Improvement {i}\n")
                f.write(f"- **Type**: {imp.get('type', 'unknown')}\n")
                f.write(f"- **Priority**: {imp.get('priority', 'medium')}\n")
                f.write(f"- **Issue**: {imp.get('issue', 'N/A')}\n")
                f.write(f"- **Action**: {imp.get('action', 'N/A')}\n\n")
        
        print(f"[IMPROVEMENTS] Saved to {report_file}")
        return report_file


class RecursiveImprovementLoop:
    """Main orchestrator for recursive improvement"""
    
    def __init__(self, workspace_root: str, max_iterations: int = 5):
        self.workspace_root = Path(workspace_root)
        self.max_iterations = max_iterations
        self.analyzer = BuildAnalyzer(str(self.workspace_root))
        self.improver = CodeImprover(str(self.workspace_root))
        self.iteration = 0
        self.history = []
    
    def run(self) -> Dict:
        """Execute recursive improvement loop"""
        print("\n" + "="*60)
        print("RECURSIVE IMPROVEMENT LOOP STARTED")
        print("="*60 + "\n")
        
        while self.iteration < self.max_iterations:
            self.iteration += 1
            print(f"\n[ITERATION {self.iteration}/{self.max_iterations}]")
            
            # Step 1: Build
            returncode, stdout, stderr = self.analyzer.run_build()
            
            # Step 2: Analyze
            analysis = self.analyzer.parse_build_output(stdout, stderr)
            analysis["iteration"] = self.iteration
            analysis["build_success"] = returncode == 0
            
            self.history.append(analysis)
            
            # Step 3: Report
            print(f"  Errors: {len(analysis['errors'])}")
            print(f"  Warnings: {len(analysis['warnings'])}")
            print(f"  Build Success: {analysis['build_success']}")
            
            # Step 4: Generate improvements
            improvements = self.improver.generate_improvements(analysis)
            if improvements:
                self.improver.save_improvements(improvements)
                print(f"  Generated {len(improvements)} improvements")
            
            # Step 5: Check if we should continue
            if analysis["build_success"] and len(analysis["errors"]) == 0:
                print("\n✓ Build successful with no errors!")
                break
            
            if len(analysis["errors"]) == 0 and len(analysis["warnings"]) < 5:
                print("\n✓ Build quality acceptable!")
                break
        
        return self._generate_summary()
    
    def _generate_summary(self) -> Dict:
        """Generate final summary"""
        summary = {
            "total_iterations": self.iteration,
            "final_status": "success" if self.history[-1]["build_success"] else "incomplete",
            "initial_errors": len(self.history[0]["errors"]),
            "final_errors": len(self.history[-1]["errors"]),
            "initial_warnings": len(self.history[0]["warnings"]),
            "final_warnings": len(self.history[-1]["warnings"]),
            "history": self.history
        }
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Iterations: {summary['total_iterations']}")
        print(f"Errors reduced: {summary['initial_errors']} → {summary['final_errors']}")
        print(f"Warnings reduced: {summary['initial_warnings']} → {summary['final_warnings']}")
        print("="*60 + "\n")
        
        return summary


if __name__ == "__main__":
    workspace = os.environ.get("QALLOW_ROOT", "/home/xing/Qallow")
    max_iter = int(os.environ.get("QALLOW_MAX_ITERATIONS", "5"))
    
    loop = RecursiveImprovementLoop(workspace, max_iterations=max_iter)
    result = loop.run()
    
    sys.exit(0 if result["final_status"] == "success" else 1)

