# Minimal Qallow Phase 12 Test
# Using proper Bend syntax

def main():
  bend IO:
    * <- IO/print("╔══════════════════════════════╗\n")
    * <- IO/print("║  QALLOW Phase 12 - Bend      ║\n")
    * <- IO/print("╚══════════════════════════════╝\n\n")
    * <- IO/print("tick,coherence,entropy,decoherence\n")
    * <- run_simulation(1, 100)
    * <- IO/print("\n[DONE] Simulation complete\n")
  return 0

def run_simulation(tick, max_ticks):
  bend IO:
    open IO
    if tick <= max_ticks:
      # Print tick number
      * <- IO/print(String/from_num(tick))
      * <- IO/print(",0.999")
      * <- IO/print(String/from_num(tick % 10))
      * <- IO/print(",0.000")
      * <- IO/print(String/from_num(tick % 10))
      * <- IO/print(",0.00000")
      * <- IO/print(String/from_num(tick % 10))
      * <- IO/print("\n")
      
      # Continue to next tick
      * <- run_simulation(tick + 1, max_ticks)
  return *
