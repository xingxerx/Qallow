#!/usr/bin/env python3
import re
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

if __name__ == '__main__':
    log10k = Path('data/logs/phase14_stress_10k.log')
    log50k = Path('data/logs/phase14_stress_50k.log')
    out = Path('data/plots/phase14_fidelity_overlay.png')
    out.parent.mkdir(parents=True, exist_ok=True)

    t10, v10 = parse_log(log10k)
    t50, v50 = parse_log(log50k)

    plt.figure(figsize=(10,4))
    if t10:
        plt.plot(t10, v10, label='10k ticks', color='#1f77b4')
    if t50:
        plt.plot(t50, v50, label='50k ticks', color='#ff7f0e', alpha=0.8)
    plt.xlabel('Tick')
    plt.ylabel('Fidelity')
    plt.title('Phase 14 Fidelity: 10k vs 50k')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.94, 0.982)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
