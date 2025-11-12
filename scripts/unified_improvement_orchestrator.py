#!/usr/bin/env python3
"""
Unified Improvement Orchestrator
Coordinates recursive learning loop across build, analysis, and code improvement
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import argparse

# Import our modules
sys.path.insert(0, str(Path(__file__).parent))
from recursive_improvement_loop import RecursiveImprovementLoop, BuildAnalyzer
from code_improvement_engine import CodePatternAnalyzer, ImprovementRecommender


class UnifiedOrchestrator:
    """Orchestrates the complete recursive improvement cycle"""
    
    def __init__(self, workspace_root: str, config: Dict = None):
        self.workspace_root = Path(workspace_root)
        self.config = config or self._load_config()
        self.reports_dir = self.workspace_root / "improvement_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.build_analyzer = BuildAnalyzer(str(self.workspace_root))
        self.code_analyzer = CodePatternAnalyzer(str(self.workspace_root))
        self.recommender = ImprovementRecommender(str(self.workspace_root))
    
    def _load_config(self) -> Dict:
        """Load configuration"""
        config_file = self.workspace_root / "config" / "improvement_config.json"
        
        if config_file.exists():
            with open(config_file) as f:
                return json.load(f)
        
        return {
            "max_iterations": 5,
            "build_timeout": 600,
            "analyze_code": True,
            "auto_fix": False,
            "report_format": "json"
        }
    
    def run_complete_cycle(self) -> Dict:
        """Run complete improvement cycle"""
        print("\n" + "="*70)
        print("UNIFIED RECURSIVE IMPROVEMENT ORCHESTRATOR")
        print("="*70 + "\n")
        
        cycle_start = datetime.now()
        results = {
            "cycle_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "start_time": cycle_start.isoformat(),
            "phases": {}
        }
        
        # Phase 1: Build and Analyze
        print("[PHASE 1] Build and Analyze")
        print("-" * 70)
        build_results = self._phase_build_and_analyze()
        results["phases"]["build_analysis"] = build_results
        
        # Phase 2: Code Pattern Analysis
        print("\n[PHASE 2] Code Pattern Analysis")
        print("-" * 70)
        code_results = self._phase_code_analysis()
        results["phases"]["code_analysis"] = code_results
        
        # Phase 3: Generate Recommendations
        print("\n[PHASE 3] Generate Recommendations")
        print("-" * 70)
        recommendations = self._phase_generate_recommendations(code_results)
        results["phases"]["recommendations"] = recommendations
        
        # Phase 4: Generate Report
        print("\n[PHASE 4] Generate Report")
        print("-" * 70)
        report_file = self._generate_final_report(results)
        results["report_file"] = str(report_file)
        
        cycle_end = datetime.now()
        results["end_time"] = cycle_end.isoformat()
        results["duration_seconds"] = (cycle_end - cycle_start).total_seconds()
        
        print("\n" + "="*70)
        print("CYCLE COMPLETE")
        print("="*70)
        print(f"Duration: {results['duration_seconds']:.1f} seconds")
        print(f"Report: {report_file}")
        
        return results
    
    def _phase_build_and_analyze(self) -> Dict:
        """Phase 1: Build and analyze"""
        print("  Running build...")
        returncode, stdout, stderr = self.build_analyzer.run_build()
        
        print("  Analyzing build output...")
        analysis = self.build_analyzer.parse_build_output(stdout, stderr)
        analysis_file = self.build_analyzer.save_analysis(analysis)
        
        print(f"  ✓ Build analysis complete")
        print(f"    - Errors: {len(analysis['errors'])}")
        print(f"    - Warnings: {len(analysis['warnings'])}")
        print(f"    - Issues: {len(analysis['issues'])}")
        
        return {
            "success": returncode == 0,
            "errors": len(analysis["errors"]),
            "warnings": len(analysis["warnings"]),
            "issues": len(analysis["issues"]),
            "analysis_file": str(analysis_file)
        }
    
    def _phase_code_analysis(self) -> Dict:
        """Phase 2: Analyze code patterns"""
        print("  Scanning source directories...")
        
        src_dirs = [
            self.workspace_root / "src",
            self.workspace_root / "backend",
            self.workspace_root / "interface",
            self.workspace_root / "python"
        ]
        
        all_issues = []
        for src_dir in src_dirs:
            if src_dir.exists():
                print(f"    Analyzing {src_dir.name}...")
                issues = self.code_analyzer.analyze_directory(src_dir)
                all_issues.extend(issues)
        
        print(f"  ✓ Code analysis complete")
        print(f"    - Total issues found: {len(all_issues)}")
        
        # Categorize by severity
        by_severity = {}
        for issue in all_issues:
            severity = issue.get("severity", "medium")
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        print(f"    - By severity: {by_severity}")
        
        return {
            "total_issues": len(all_issues),
            "by_severity": by_severity,
            "issues": all_issues[:100]  # Top 100
        }
    
    def _phase_generate_recommendations(self, code_results: Dict) -> Dict:
        """Phase 3: Generate recommendations"""
        print("  Generating recommendations...")
        
        recommendations = self.recommender.generate_recommendations(
            code_results.get("issues", [])
        )
        
        rec_file = self.recommender.save_recommendations(recommendations)
        
        print(f"  ✓ Recommendations generated")
        print(f"    - Total recommendations: {len(recommendations['recommendations'])}")
        print(f"    - By severity: {recommendations['by_severity']}")
        
        return {
            "total": len(recommendations["recommendations"]),
            "by_severity": recommendations["by_severity"],
            "by_type": recommendations["by_type"],
            "file": str(rec_file)
        }
    
    def _generate_final_report(self, results: Dict) -> Path:
        """Generate final comprehensive report"""
        report_file = self.reports_dir / f"cycle_{results['cycle_id']}.json"
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also generate markdown summary
        md_file = self.reports_dir / f"cycle_{results['cycle_id']}.md"
        self._generate_markdown_report(results, md_file)
        
        return report_file
    
    def _generate_markdown_report(self, results: Dict, filepath: Path):
        """Generate markdown report"""
        with open(filepath, 'w') as f:
            f.write("# Recursive Improvement Cycle Report\n\n")
            f.write(f"**Cycle ID**: {results['cycle_id']}\n")
            duration = results.get('duration_seconds', 0)
            f.write(f"**Duration**: {duration:.1f} seconds\n\n")
            
            # Build Analysis
            f.write("## Build Analysis\n")
            build = results["phases"]["build_analysis"]
            f.write(f"- **Success**: {build['success']}\n")
            f.write(f"- **Errors**: {build['errors']}\n")
            f.write(f"- **Warnings**: {build['warnings']}\n")
            f.write(f"- **Issues**: {build['issues']}\n\n")
            
            # Code Analysis
            f.write("## Code Analysis\n")
            code = results["phases"]["code_analysis"]
            f.write(f"- **Total Issues**: {code['total_issues']}\n")
            f.write(f"- **By Severity**: {code['by_severity']}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n")
            rec = results["phases"]["recommendations"]
            f.write(f"- **Total**: {rec['total']}\n")
            f.write(f"- **By Severity**: {rec['by_severity']}\n")
            f.write(f"- **By Type**: {rec['by_type']}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Recursive Improvement Orchestrator"
    )
    parser.add_argument(
        "--workspace",
        default=os.environ.get("QALLOW_ROOT", "/home/xing/Qallow"),
        help="Workspace root directory"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Maximum iterations"
    )
    parser.add_argument(
        "--analyze-code",
        action="store_true",
        default=True,
        help="Analyze code patterns"
    )
    
    args = parser.parse_args()
    
    config = {
        "max_iterations": args.iterations,
        "analyze_code": args.analyze_code
    }
    
    orchestrator = UnifiedOrchestrator(args.workspace, config)
    results = orchestrator.run_complete_cycle()
    
    return 0 if results["phases"]["build_analysis"]["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

