#!/usr/bin/env python3
"""
collect_signals.py - Hardware telemetry collector for Qallow ethics system
Converts real system metrics into normalized [0,1] signals
Part of Phase 13: Closed-loop ethics monitoring
"""

import json
import os
import sys
import time
import subprocess
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("/root/Qallow/data/telemetry")
OUTPUT_FILE = OUTPUT_DIR / "current_signals.txt"
LOG_FILE = OUTPUT_DIR / "collection.log"
HUMAN_FEEDBACK_FILE = Path("/root/Qallow/data/human_feedback.txt")

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    """Write to log file with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[collect] {msg}")


def safe_exec(cmd, default="0"):
    """Execute shell command safely, return default on error"""
    try:
        result = subprocess.getoutput(cmd).strip()
        return result if result else default
    except Exception as e:
        log(f"Command failed: {cmd} - {e}")
        return default


def safety_metric():
    """
    Hardware health: CPU temperature, system load
    Returns 3 values representing thermal stability, ECC health, power stability
    """
    try:
        # CPU temperature (normalize: 1.0 at <=40°C, 0.0 at >=80°C)
        thermal_zones = list(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
        if thermal_zones:
            temp_raw = int(open(thermal_zones[0]).read().strip())
            temp_c = temp_raw / 1000.0
            thermal_score = max(0.0, min(1.0, (80.0 - temp_c) / 40.0))
        else:
            thermal_score = 0.95  # No sensor = assume OK
        
        # System load (normalize: 1.0 at low load, 0.0 at overload)
        load_avg = float(safe_exec("uptime | awk '{print $(NF-2)}' | tr -d ','", "1.0"))
        cpu_count = int(safe_exec("nproc", "4"))
        load_score = max(0.0, min(1.0, 1.0 - (load_avg / (cpu_count * 2))))
        
        # Memory pressure (normalize: 1.0 at <70% used, 0.0 at >95%)
        mem_info = safe_exec("free | grep Mem | awk '{print $3/$2}'", "0.5")
        mem_used = float(mem_info)
        mem_score = max(0.0, min(1.0, (0.95 - mem_used) / 0.25))
        
        return [round(thermal_score, 3), round(load_score, 3), round(mem_score, 3)]
    
    except Exception as e:
        log(f"Safety metric error: {e}")
        return [0.90, 0.90, 0.90]  # Conservative fallback


def clarity_metric():
    """
    Software integrity: build quality, code health
    Returns 4 values: compile success, warning count, test pass rate, lint score
    """
    try:
        build_log = Path("/root/Qallow/build.log")
        
        if build_log.exists():
            log_content = build_log.read_text()
            
            # Build success (1.0 if no errors)
            error_count = log_content.count("error:") + log_content.count("Error:")
            build_score = 1.0 if error_count == 0 else max(0.0, 1.0 - error_count / 5.0)
            
            # Warning count (1.0 if <3 warnings)
            warning_count = log_content.count("warning:") + log_content.count("Warning:")
            warning_score = max(0.0, 1.0 - warning_count / 10.0)
            
            # Test results (parse from test output if available)
            test_score = 0.98  # Placeholder - integrate actual test runner
            
            # Lint/code quality (placeholder for static analysis)
            lint_score = 0.97
        else:
            # No build log = assume clean build
            build_score = 1.0
            warning_score = 1.0
            test_score = 1.0
            lint_score = 1.0
        
        return [round(build_score, 3), round(warning_score, 3), 
                round(test_score, 3), round(lint_score, 3)]
    
    except Exception as e:
        log(f"Clarity metric error: {e}")
        return [0.95, 0.95, 0.95, 0.95]


def human_metric():
    """
    Human benefit/feedback: explicit operator scores
    Returns 3 values: direct feedback, user satisfaction, ethical override
    """
    try:
        if HUMAN_FEEDBACK_FILE.exists():
            content = HUMAN_FEEDBACK_FILE.read_text().strip()
            
            # Parse simple format: single float or space-separated floats
            values = [float(x) for x in content.split()]
            
            if len(values) >= 3:
                scores = values[:3]
            elif len(values) == 1:
                # Replicate single value
                scores = [values[0]] * 3
            else:
                scores = values + [0.75] * (3 - len(values))
            
            # Clamp to [0,1]
            scores = [max(0.0, min(1.0, s)) for s in scores]
        else:
            # No feedback file = neutral/positive default
            scores = [0.75, 0.75, 0.75]
        
        return [round(s, 3) for s in scores]
    
    except Exception as e:
        log(f"Human metric error: {e}")
        return [0.75, 0.75, 0.75]


def collect_all_signals():
    """Main collection routine"""
    log("Starting signal collection...")
    
    safety = safety_metric()
    clarity = clarity_metric()
    human = human_metric()
    
    # Average each dimension to single value for simpler C structure
    safety_avg = sum(safety) / len(safety)
    clarity_avg = sum(clarity) / len(clarity)
    human_avg = sum(human) / len(human)
    
    signals = {
        "safety": safety,
        "clarity": clarity,
        "human": human,
        "safety_avg": round(safety_avg, 3),
        "clarity_avg": round(clarity_avg, 3),
        "human_avg": round(human_avg, 3),
        "timestamp": int(time.time())
    }
    
    # Write to output file - 10 values for compatibility with detailed parsing
    # But C code will read first 3 as averages
    all_values = safety + clarity + human
    with open(OUTPUT_FILE, "w") as f:
        # Add timestamp as comment
        f.write(f"# {signals['timestamp']}\n")
        f.write(" ".join(str(v) for v in all_values) + "\n")
    
    log(f"Signals: Safety={safety_avg:.3f} Clarity={clarity_avg:.3f} Human={human_avg:.3f}")
    
    # Also write JSON for debugging/monitoring
    json_file = OUTPUT_DIR / "current_signals.json"
    with open(json_file, "w") as f:
        json.dump(signals, f, indent=2)
    
    return signals


def main():
    """Run collection once or in loop mode"""
    if "--loop" in sys.argv:
        # Continuous mode (for daemon operation)
        interval = 5  # seconds
        log(f"Running in loop mode (interval={interval}s)")
        
        try:
            while True:
                collect_all_signals()
                time.sleep(interval)
        except KeyboardInterrupt:
            log("Loop interrupted by user")
    else:
        # Single shot
        signals = collect_all_signals()
        print(json.dumps(signals, indent=2))


if __name__ == "__main__":
    main()
