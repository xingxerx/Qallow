# Qallow Phase 12 - Correct Bend Syntax

def main:
  Phase12(100, 0.0001)

# Phase 12 simulation
Phase12 ticks eps =
  Loop(1, ticks, 0.99990, 0.00070, 0.000009, eps)

# Simulation loop
Loop tick max_tick coherence entropy deco eps =
  match tick > max_tick:
    case True:
      tick
    case False:
      let new_entropy = Clamp(entropy - 0.000001 + eps * 0.0000002, 0.0, 0.001)
      let new_coherence = Clamp(1.0 - new_entropy * 0.2, 0.0, 1.0)
      let new_deco = Clamp(deco * (1.0 - 0.0005) + eps * 0.0000001, 0.0, 0.001)
      Loop(tick + 1, max_tick, new_coherence, new_entropy, new_deco, eps)

# Clamp function
Clamp x lo hi =
  match x < lo:
    case True:
      lo
    case False:
      match x > hi:
        case True:
          hi
        case False:
          x
