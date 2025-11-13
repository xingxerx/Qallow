#!/usr/bin/env python3
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_log(path: Path):
    ticks = []
    vals = []
    pat = re.compile(r"\[PHASE14\]\[(\d+)\]\s+fidelity=(\d+\.\d+)")
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            ticks.append(int(m.group(1)))
            vals.append(float(m.group(2)))
    return ticks, vals

def main():
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/logs/phase14_stress_10k.log')
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('data/plots/phase14_fidelity_10k.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    t, v = parse_log(log)
    if not t:
        print(f"No fidelity points found in {log}")
        sys.exit(1)
    plt.figure(figsize=(9,4))
    plt.plot(t, v, label='Fidelity', color='#1f77b4')
    plt.xlabel('Tick')
    plt.ylabel('Fidelity')
    plt.title('Phase 14 Fidelity over Time')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.94, 0.982)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

if __name__ == '__main__':
    main()
