#!/usr/bin/env python3
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_csv(path: Path):
    t, coh, dec = [], [], []
    with path.open() as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            try:
                t.append(int(row[0]))
                coh.append(float(row[1]))
                dec.append(float(row[3]))
            except Exception:
                continue
    return t, coh, dec

if __name__ == '__main__':
    f10 = Path('data/logs/phase12_10k.csv')
    f50 = Path('data/logs/phase12_50k.csv')
    out = Path('data/plots/phase12_coherence_overlay.png')
    out.parent.mkdir(parents=True, exist_ok=True)

    t10, c10, d10 = read_csv(f10)
    t50, c50, d50 = read_csv(f50)

    fig, ax1 = plt.subplots(figsize=(10,4))
    if t10:
        ax1.plot(t10, c10, label='Coherence (10k)', color='#2ca02c')
    if t50:
        ax1.plot(t50, c50, label='Coherence (50k)', color='#1f77b4', alpha=0.8)
    ax1.set_xlabel('Tick')
    ax1.set_ylabel('Coherence')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.994, 1.001)

    ax2 = ax1.twinx()
    if t10:
        ax2.plot(t10, d10, label='Deco (10k)', color='#d62728', alpha=0.5)
    if t50:
        ax2.plot(t50, d50, label='Deco (50k)', color='#9467bd', alpha=0.5)
    ax2.set_ylabel('Decoherence')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.suptitle('Phase 12: Coherence and Decoherence (10k vs 50k)')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
