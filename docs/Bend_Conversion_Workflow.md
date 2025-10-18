# Bend Auto-Conversion Workflow

This note captures the helper tooling that scaffolds Bend templates from the existing C / CUDA implementation.

## Script

`convert_to_bend.py` walks the canonical Qallow sources:

* `backend/cpu/**/*.c`
* `backend/cuda/**/*.cu`
* `interface/**/*.c`

For each translation unit it emits a matching `.bend` stub under `bend/auto/…`.  
The emitted file preserves the relative path so that imports stay predictable (for example `backend/cpu/qallow_kernel.c` → `bend/auto/backend/cpu/qallow_kernel.bend`).

Each stub lists the detected function signatures and provides a Bend definition skeleton:

```bend
# Auto-generated Bend template
# Source: backend/cpu/telemetry.c

def telemetry_init(tel):
  # TODO: implement
  pass
```

The simple parser ignores prototypes and only mirrors functions with bodies.

## Usage

```bash
python3 convert_to_bend.py
```

After running, inspect the generated files in `bend/auto/` and replace the `pass` blocks with the real Bend logic.

## Notes

* The converter is intentionally lightweight—it provides a deterministic structure for subsequent AI-assisted rewriting.
* CUDA kernels appear in `bend/auto/backend/cuda/*.bend`.  Port the kernel bodies manually or with follow-up tooling.
* Re-run the script after adding new C/CUDA modules; it updates / overwrites the corresponding stubs.
