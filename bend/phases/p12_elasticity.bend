from ../util import (
  clamp_entropy,
  clamp_coherence,
  clamp_decoherence,
  str,
  strf,
)

def step(state, eps):
  entropy = clamp_entropy(state.entropy - 0.000001 + eps * 0.0000002)
  coherence = clamp_coherence(1.0 - entropy * 0.2)
  deco = clamp_decoherence(state.deco * (1.0 - 0.0005) + eps * 0.0000001)
  return { coherence: coherence, entropy: entropy, deco: deco }

def loop_csv(t, max, state, eps, acc):
  if t > max:
    return acc
  else:
    next = step(state, eps)
    line = str(t) ++ "," ++ strf(next.coherence, 6) ++ "," ++ strf(next.entropy, 6) ++ "," ++ strf(next.deco, 6) ++ "\n"
    return loop_csv(t + 1, max, next, eps, acc ++ line)

def phase12_csv(ticks, eps):
  init = { coherence: 0.99990, entropy: 0.00070, deco: 0.000009 }
  return "tick,coherence,entropy,decoherence\n" ++ loop_csv(1, ticks, init, eps, "")
