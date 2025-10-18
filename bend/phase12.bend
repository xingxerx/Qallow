# Pure elasticity loop. No side effects except final print.
def clamp(x, lo, hi):
  switch x < lo:
    case 0:
      switch x > hi:
        case 0: return x
        case _: return hi
    case _: return lo

def step(state, eps):
  let entropy = clamp(state["entropy"] - 0.000001 + eps * 0.0000002, 0.0, 0.001)
  let coherence = clamp(1.0 - entropy * 0.2, 0.0, 1.0)
  let deco = clamp(state["deco"] * (1.0 - 0.0005) + eps * 0.0000001, 0.0, 0.001)
  return {"coherence": coherence, "entropy": entropy, "deco": deco}

def loop(t, max, s, eps, acc):
  switch t > max:
    case 0:
      let s2 = step(s, eps)
      return loop(t + 1, max, s2, eps, String/concat(acc, String/concat(String/from_num(t), ",")))
    case _:
      return acc

def phase12(ticks, eps):
  let s0 = {"coherence": 0.99990, "entropy": 0.00070, "deco": 0.000009}
  return loop(1, ticks, s0, eps, "")

def main():
  with IO:
    * <- IO/print("tick,coherence,entropy,decoherence\n")
    let result = phase12(100, 0.0001)
    * <- IO/print(result)
    * <- IO/print("\n")
    return wrap(0)
