# Bend translation of Phase 12 elasticity loop.

import "../util.bend"

def step(state, eps):
  let entropy = clamp_entropy(state.entropy - 0.000001 + eps * 0.0000002)
  let coherence = clamp_coherence(1.0 - entropy * 0.2)
  let deco = clamp_decoherence(state.deco * (1.0 - 0.0005) + eps * 0.0000001)
  { coherence: coherence, entropy: entropy, deco: deco }

def loop_csv(t, max, state, eps, acc):
  if t > max then acc
  else
    let next = step(state, eps)
    let line = str(t) ++ "," ++ strf(next.coherence, 6) ++ "," ++ strf(next.entropy, 6) ++ "," ++ strf(next.deco, 6) ++ "\n"
    loop_csv(t + 1, max, next, eps, acc ++ line)

def phase12_csv(ticks, eps):
  let init = { coherence: 0.99990, entropy: 0.00070, deco: 0.000009 }
  "tick,coherence,entropy,decoherence\n" ++ loop_csv(1, ticks, init, eps, "")
