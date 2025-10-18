# Qallow Phase 12 - Working Bend Implementation

def main:
  return (Phase12 100 0.0001)

# Phase 12 simulation
def Phase12(ticks eps):
  return (Loop 1 ticks 0.99990 0.00070 0.000009 eps)

# Simulation loop
def Loop(tick max_tick coherence entropy deco eps):
  switch (> tick max_tick):
    case 0:
      let new_entropy = (Clamp (+ (- entropy 0.000001) (* eps 0.0000002)) 0.0 0.001)
      let new_coherence = (Clamp (- 1.0 (* new_entropy 0.2)) 0.0 1.0)
      let new_deco = (Clamp (+ (* deco (- 1.0 0.0005)) (* eps 0.0000001)) 0.0 0.001)
      return (Loop (+ tick 1) max_tick new_coherence new_entropy new_deco eps)
    case _:
      return tick

# Clamp function
def Clamp(x lo hi):
  switch (< x lo):
    case 0:
      switch (> x hi):
        case 0:
          return x
        case _:
          return hi
    case _:
      return lo
