#!/usr/bin/env python3
from pathlib import Path
import csv
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_csv(path: Path):
    t, coh, dec = [], [], []
    with path.open() as f:
        r = csv.reader(f)
        header = next(r, None)
        # assuming columns: tick,coherence,entropy,decoherence,...
        for row in r:
            try:
                t.append(int(row[0]))
                coh.append(float(row[1]))
                dec.append(float(row[3]))
            except Exception:
                continue
    return t, coh, dec

def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/logs/phase12.csv')
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('data/plots/phase12_coherence.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    t, coh, dec = read_csv(src)
    if not t:
        print(f"No rows found in {src}")
        sys.exit(1)
    fig, ax1 = plt.subplots(figsize=(9,4))
    ax1.plot(t, coh, color='#2ca02c', label='Coherence')
    ax1.set_xlabel('Tick')
    ax1.set_ylabel('Coherence', color='#2ca02c')
    ax1.tick_params(axis='y', labelcolor='#2ca02c')
    ax2 = ax1.twinx()
    ax2.plot(t, dec, color='#d62728', alpha=0.7, label='Decoherence')
    ax2.set_ylabel('Decoherence', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    fig.suptitle('Phase 12: Coherence and Decoherence')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

if __name__ == '__main__':
    main()
