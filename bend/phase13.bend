def pi() = 3.141592653589793

def init_phases(n, i = 0):
  if i == n then []
  else [2.0 * pi() * to_float(i) / to_float(n)] ++ init_phases(n, i + 1)

def avg(xs) = sum(xs) / to_float(len(xs))
def abs(x) = if x < 0.0 then -x else x

def step(phases, k):
  let m = avg(phases)
  map(p -> p + k * (m - p), phases)

def drift(phases):
  let m = avg(phases)
  avg(map(p -> abs(m - p), phases))

def loop(t, T, phases, k, acc):
  if t > T then acc
  else
    let d = drift(phases)
    let coh = 1.0 / (1.0 + d * 1000.0)
    let phases2 = step(phases, k)
    loop(t + 1, T, phases2, k, acc ++ [(t, coh, d)])

def main(args):
  let nodes = to_int(args[0])
  let ticks = to_int(args[1])
  let k = to_float(args[2])
  let p0 = init_phases(nodes)
  let rows = loop(1, ticks, p0, k, [])
  print("tick,avg_coherence,phase_drift")
  for r in rows:
    print(str(r.0) ++ "," ++ strf(r.1, 6) ++ "," ++ strf(r.2, 6))
