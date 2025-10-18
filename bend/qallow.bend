import "util.bend"
import "phases/p12_elasticity.bend"
import "phases/p13_harmonic.bend"

def run_auto():
  let csv12 = phase12_csv(100, 0.0001)
  let csv13 = phase13_csv(16, 500, 0.001)
  csv12 ++ "\n" ++ csv13

def main(_args) -> IO(u24):
  with IO:
    * <- IO/print("[QALLOW] Bend auto-mode (phase12 + phase13)\n")
    let log = run_auto()
    * <- IO/print(log)
    return wrap(0)
