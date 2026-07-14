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
        "src/mind/persist_lmdb.c",
        "third_party/lmdb/mdb.c",
        "third_party/lmdb/midl.c",
    ];

    let mut build = cc::Build::new();
    build
        .include(root.join("include"))
        .include(root.join("core/include"))
        .include(root.join("third_party/lmdb"))
        // Match qsw_envelope's QSW_MAX_KEY_LEN (sync_wire.h); LMDB's
        // own default (511) is smaller and would reject valid envelopes.
        .define("MDB_MAXKEYSIZE", "1024")
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
        "include/qallow/sync_wire.h",
        "include/qallow/persist_lmdb.h",
        "third_party/lmdb/lmdb.h",
        "third_party/lmdb/midl.h",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(header).display());
    }

    build.compile("qallow_phases");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        // LMDB's Windows lock-file setup uses Win32 security-descriptor APIs.
        println!("cargo:rustc-link-lib=advapi32");
    }
}
