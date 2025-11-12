#!/usr/bin/env python3
"""
Advanced Code Improvement Engine
Analyzes code patterns and generates targeted improvements
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class CodePatternAnalyzer:
    """Analyzes code patterns for improvements"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict:
        """Load improvement patterns"""
        return {
            "unused_variables": {
                "pattern": r"unused variable",
                "fix_type": "removal",
                "priority": "low"
            },
            "missing_includes": {
                "pattern": r"implicit declaration|undefined reference",
                "fix_type": "add_header",
                "priority": "high"
            },
            "memory_issues": {
                "pattern": r"memory leak|buffer overflow|use-after-free",
                "fix_type": "memory_fix",
                "priority": "critical"
            },
            "type_mismatches": {
                "pattern": r"incompatible types|type mismatch",
                "fix_type": "type_cast",
                "priority": "high"
            },
            "performance": {
                "pattern": r"inefficient|slow|O\\(n\\^2\\)",
                "fix_type": "optimization",
                "priority": "medium"
            }
        }
    
    def analyze_file(self, filepath: Path) -> List[Dict]:
        """Analyze single file for improvement opportunities"""
        issues = []
        
        if not filepath.exists():
            return issues
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            return issues
        
        # Check for common patterns
        issues.extend(self._check_unused_variables(filepath, lines))
        issues.extend(self._check_missing_headers(filepath, lines))
        issues.extend(self._check_code_quality(filepath, lines))
        
        return issues
    
    def _check_unused_variables(self, filepath: Path, lines: List[str]) -> List[Dict]:
        """Check for unused variables"""
        issues = []
        
        # Simple pattern: variable declared but never used
        var_pattern = r'^\s*(int|char|float|double|void\*|struct\s+\w+)\s+(\w+)\s*[=;]'
        
        for i, line in enumerate(lines, 1):
            match = re.search(var_pattern, line)
            if match:
                var_name = match.group(2)
                # Check if variable is used later
                remaining = '\n'.join(lines[i:])
                if var_name not in remaining:
                    issues.append({
                        "file": str(filepath),
                        "line": i,
                        "type": "unused_variable",
                        "variable": var_name,
                        "severity": "low",
                        "suggestion": f"Remove unused variable '{var_name}'"
                    })
        
        return issues
    
    def _check_missing_headers(self, filepath: Path, lines: List[str]) -> List[Dict]:
        """Check for potentially missing headers"""
        issues = []
        
        # Check for common function calls without includes
        function_includes = {
            "printf": "stdio.h",
            "malloc": "stdlib.h",
            "strlen": "string.h",
            "sqrt": "math.h",
            "pthread_create": "pthread.h"
        }
        
        includes = set()
        for line in lines:
            if line.strip().startswith("#include"):
                includes.add(line)
        
        for func, header in function_includes.items():
            for i, line in enumerate(lines, 1):
                if func in line and f"#include <{header}>" not in includes:
                    issues.append({
                        "file": str(filepath),
                        "line": i,
                        "type": "missing_include",
                        "function": func,
                        "header": header,
                        "severity": "high",
                        "suggestion": f"Add '#include <{header}>' for {func}()"
                    })
        
        return issues
    
    def _check_code_quality(self, filepath: Path, lines: List[str]) -> List[Dict]:
        """Check for code quality issues"""
        issues = []
        
        for i, line in enumerate(lines, 1):
            # Check for very long lines
            if len(line) > 120:
                issues.append({
                    "file": str(filepath),
                    "line": i,
                    "type": "long_line",
                    "severity": "low",
                    "suggestion": "Line exceeds 120 characters, consider breaking it up"
                })
            
            # Check for multiple statements on one line
            if ';' in line and line.count(';') > 2:
                issues.append({
                    "file": str(filepath),
                    "line": i,
                    "type": "multiple_statements",
                    "severity": "low",
                    "suggestion": "Multiple statements on one line, improve readability"
                })
        
        return issues
    
    def analyze_directory(self, directory: Path, extensions: List[str] = None) -> List[Dict]:
        """Analyze all files in directory"""
        if extensions is None:
            extensions = ['.c', '.h', '.cpp', '.py']
        
        all_issues = []
        
        for ext in extensions:
            for filepath in directory.rglob(f'*{ext}'):
                # Skip build and vendor directories
                if 'build' in filepath.parts or 'venv' in filepath.parts:
                    continue
                
                issues = self.analyze_file(filepath)
                all_issues.extend(issues)
        
        return all_issues


class ImprovementRecommender:
    """Recommends specific code improvements"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.analyzer = CodePatternAnalyzer(workspace_root)
    
    def generate_recommendations(self, issues: List[Dict]) -> Dict:
        """Generate improvement recommendations"""
        recommendations = {
            "timestamp": datetime.now().isoformat(),
            "total_issues": len(issues),
            "by_severity": {},
            "by_type": {},
            "recommendations": []
        }
        
        # Categorize issues
        for issue in issues:
            severity = issue.get("severity", "medium")
            issue_type = issue.get("type", "unknown")
            
            recommendations["by_severity"][severity] = recommendations["by_severity"].get(severity, 0) + 1
            recommendations["by_type"][issue_type] = recommendations["by_type"].get(issue_type, 0) + 1
        
        # Sort by severity
        sorted_issues = sorted(issues, key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x.get("severity"), 4))
        
        recommendations["recommendations"] = sorted_issues[:50]  # Top 50
        
        return recommendations
    
    def save_recommendations(self, recommendations: Dict) -> Path:
        """Save recommendations to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.workspace_root / "improvement_reports" / f"recommendations_{timestamp}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        return report_file


if __name__ == "__main__":
    workspace = os.environ.get("QALLOW_ROOT", "/home/xing/Qallow")
    
    analyzer = CodePatternAnalyzer(workspace)
    recommender = ImprovementRecommender(workspace)
    
    # Analyze source directories
    src_dirs = [
        Path(workspace) / "src",
        Path(workspace) / "backend",
        Path(workspace) / "interface",
        Path(workspace) / "python"
    ]
    
    all_issues = []
    for src_dir in src_dirs:
        if src_dir.exists():
            print(f"Analyzing {src_dir}...")
            issues = analyzer.analyze_directory(src_dir)
            all_issues.extend(issues)
    
    # Generate recommendations
    recommendations = recommender.generate_recommendations(all_issues)
    report_file = recommender.save_recommendations(recommendations)
    
    print(f"\nAnalysis complete!")
    print(f"Total issues found: {recommendations['total_issues']}")
    print(f"Report saved to: {report_file}")
    print(f"\nBy Severity: {recommendations['by_severity']}")
    print(f"By Type: {recommendations['by_type']}")

