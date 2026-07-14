use std::path::Path;

fn main() {
    // Workspace root is one level up from this crate.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");

    let sources = [
        "backend/cpu/phase1_elasticity.c",
        "backend/cpu/phase2_harmonic.c",
        "backend/cpu/phase3_coherence.c",
        "backend/cpu/phase4_convergence.c",
        "src/runtime/telemetry_outputs.c",
        "src/runtime/meta_introspect.c",
        "qallow_cli/src/phase34_driver.c",
    ];

    let mut build = cc::Build::new();
    build
        .include(root.join("include"))
        .include(root.join("core/include"))
        .warnings(false)
        .flag_if_supported("/std:c11")
        .flag_if_supported("-std=c11")
        .flag_if_supported("/utf-8");

    for src in &sources {
        let path = root.join(src);
        println!("cargo:rerun-if-changed={}", path.display());
        build.file(path);
    }

    // Headers the C sources depend on.
    for header in [
        "include/qallow/ethics_axiom.h",
        "include/qallow/telemetry_outputs.h",
        "include/meta_introspect.h",
        "core/include/qallow_kernel.h",
        "core/include/phase3.h",
        "core/include/phase4.h",
        "core/include/qallow_phase1.h",
        "core/include/qallow_phase2.h",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(header).display());
    }

    build.compile("qallow_phases");
}
