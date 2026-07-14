//! `qallow` CLI: runs the C phase engines (elasticity, harmonic, coherence,
//! convergence) individually or as the unified 1->2->3->4 pipeline.
//!
//! Contract (matches native_app's launcher):
//!   qallow run --phase=N [--ticks=T]   (N in 1..=4)
//!   qallow run unified [--ticks=T]

use std::ffi::CString;
use std::os::raw::{c_char, c_float, c_int};
use std::process::ExitCode;

extern "C" {
    fn run_phase1_elasticity(
        audit_tag: *const c_char,
        requested_log_path: *const c_char,
        ticks: c_int,
        eps: c_float,
    ) -> c_int;

    fn run_phase2_harmonic(
        audit_tag: *const c_char,
        requested_log_path: *const c_char,
        pockets: c_int,
        ticks: c_int,
        coupling: c_float,
    ) -> c_int;

    fn qallow_run_phase3(ticks: c_int) -> c_int;
    fn qallow_run_phase4(ticks: c_int, audit_unified: c_int) -> c_int;
}

const DEFAULT_TICKS: i32 = 1000;
const DEFAULT_EPS: f32 = 0.1;
const DEFAULT_POCKETS: i32 = 32; // C engine caps at QALLOW_PHASE2_MAX_POCKETS = 32
const DEFAULT_COUPLING: f32 = 0.5;

fn usage() -> String {
    "Usage:\n  qallow run --phase=N [--ticks=T]   (N in 1..4)\n  qallow run unified [--ticks=T]"
        .to_string()
}

fn run_phase(phase: i32, ticks: i32, audit_unified: bool) -> Result<(), String> {
    println!("[QALLOW] Running phase {} (ticks={})...", phase, ticks);
    let tag = CString::new("cli").expect("static tag");
    let rc = match phase {
        1 => unsafe {
            run_phase1_elasticity(tag.as_ptr(), std::ptr::null(), ticks, DEFAULT_EPS)
        },
        2 => unsafe {
            run_phase2_harmonic(
                tag.as_ptr(),
                std::ptr::null(),
                DEFAULT_POCKETS,
                ticks,
                DEFAULT_COUPLING,
            )
        },
        3 => unsafe { qallow_run_phase3(ticks) },
        4 => unsafe { qallow_run_phase4(ticks, if audit_unified { 1 } else { 0 }) },
        _ => return Err(format!("Unknown phase: {}", phase)),
    };
    if rc != 0 {
        return Err(format!("Phase {} failed with code {}", phase, rc));
    }
    println!("[QALLOW] Phase {} complete.", phase);
    Ok(())
}

fn run_unified(ticks: i32) -> Result<(), String> {
    println!("[QALLOW] Unified pipeline: phases 1 -> 2 -> 3 -> 4 (ticks={})", ticks);
    run_phase(1, ticks, false)?;
    run_phase(2, ticks, false)?;
    run_phase(3, ticks, false)?;
    run_phase(4, ticks, true)?;
    println!("[QALLOW] Unified pipeline complete.");
    Ok(())
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        return Err(usage());
    }
    if args[0] != "run" {
        return Err(format!("Unknown command: {}\n{}", args[0], usage()));
    }

    let mut phase: Option<i32> = None;
    let mut unified = false;
    let mut ticks = DEFAULT_TICKS;

    for arg in &args[1..] {
        if arg == "unified" {
            unified = true;
        } else if let Some(value) = arg.strip_prefix("--phase=") {
            let n: i32 = value
                .parse()
                .map_err(|_| format!("Invalid phase number: {}", value))?;
            if !(1..=4).contains(&n) {
                return Err(format!("Phase must be 1..4, got {}", n));
            }
            phase = Some(n);
        } else if let Some(value) = arg.strip_prefix("--ticks=") {
            let t: i32 = value
                .parse()
                .map_err(|_| format!("Invalid tick count: {}", value))?;
            if t <= 0 {
                return Err(format!("Ticks must be positive, got {}", t));
            }
            ticks = t;
        } else {
            return Err(format!("Unknown argument: {}\n{}", arg, usage()));
        }
    }

    match (unified, phase) {
        (true, None) => run_unified(ticks),
        (false, Some(n)) => run_phase(n, ticks, false),
        (true, Some(_)) => Err(format!(
            "Cannot combine 'unified' with --phase\n{}",
            usage()
        )),
        (false, None) => Err(format!("Specify --phase=N or unified\n{}", usage())),
    }
}

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("{}", msg);
            ExitCode::FAILURE
        }
    }
}
