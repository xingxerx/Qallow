# Pure elasticity loop. No side effects except final print.
def clamp(x, lo, hi) = if x < lo then lo else if x > hi then hi else x

def step(state, eps):
  let entropy = clamp(state.entropy - 0.000001 + eps * 0.0000002, 0.0, 0.001)
  let coherence = clamp(1.0 - entropy * 0.2, 0.0, 1.0)
  let deco = clamp(state.deco * (1.0 - 0.0005) + eps * 0.0000001, 0.0, 0.001)
  { coherence, entropy, deco }

def loop(t, max, s, eps, acc):
  if t > max then acc
  else
    let s2 = step(s, eps)
    loop(t + 1, max, s2, eps, acc ++ [(t, s2.coherence, s2.entropy, s2.deco)])

def phase12(ticks, eps):
  let s0 = { coherence: 0.99990, entropy: 0.00070, deco: 0.000009 }
  loop(1, ticks, s0, eps, [])

def main(args):
  # args: ["ticks","eps"]
  let ticks = to_int(args[0])
  let eps = to_float(args[1])
  let rows = phase12(ticks, eps)
  print("tick,coherence,entropy,decoherence")
  for r in rows:
    print(str(r.0) ++ "," ++ strf(r.1, 6) ++ "," ++ strf(r.2, 6) ++ "," ++ strf(r.3, 6))
