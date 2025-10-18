from ../util import (
  PI,
  abs,
  str,
  strf,
  to_float,
)

def initial_mean(nodes):
  return PI() * (to_float(nodes) - 1.0) / to_float(nodes)

def initial_drift(nodes, i, mean, acc):
  if i == nodes:
    return acc / to_float(nodes)
  else:
    phase = 2.0 * PI() * to_float(i) / to_float(nodes)
    return initial_drift(nodes, i + 1, mean, acc + abs(phase - mean))

def loop_csv(t, max, drift, k, acc):
  if t > max:
    return acc
  else:
    coherence = 1.0 / (1.0 + drift * 1000.0)
    line = str(t) ++ "," ++ strf(coherence, 6) ++ "," ++ strf(drift, 6) ++ "\n"
    next_drift = drift * abs(1.0 - k)
    return loop_csv(t + 1, max, next_drift, k, acc ++ line)

def phase13_csv(nodes, ticks, k):
  mean = initial_mean(nodes)
  start_drift = initial_drift(nodes, 0, mean, 0.0)
  return "tick,avg_coherence,phase_drift\n" ++ loop_csv(1, ticks, start_drift, k, "")
