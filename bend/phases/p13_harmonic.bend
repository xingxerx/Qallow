# Simplified harmonic propagation model for Phase 13.

import "../util.bend"

def initial_mean(nodes):
  PI() * (to_float(nodes) - 1.0) / to_float(nodes)

def initial_drift(nodes, i, m, acc):
  if i == nodes then acc / to_float(nodes)
  else
    let phase = 2.0 * PI() * to_float(i) / to_float(nodes)
    initial_drift(nodes, i + 1, m, acc + abs(phase - m))

def loop_csv(t, max, drift, k, acc):
  if t > max then acc
  else
    let coherence = 1.0 / (1.0 + drift * 1000.0)
    let line = str(t) ++ "," ++ strf(coherence, 6) ++ "," ++ strf(drift, 6) ++ "\n"
    let next_drift = drift * abs(1.0 - k)
    loop_csv(t + 1, max, next_drift, k, acc ++ line)

def phase13_csv(nodes, ticks, k):
  let m = initial_mean(nodes)
  let start_drift = initial_drift(nodes, 0, m, 0.0)
  "tick,avg_coherence,phase_drift\n" ++ loop_csv(1, ticks, start_drift, k, "")
