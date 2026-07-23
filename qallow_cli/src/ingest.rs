//! `qallow ingest <store_dir> <frame_file>`: the real half of DUCTEI's
//! Qallow ingestion seam (see ductei-qallow/src/ingest.rs's `QallowSink`
//! trait on the DUCTEI side). Reads a byte stream of QSW frames written
//! by ductei-qallow-relay, decodes each with the C wire decoder
//! (qsw_decode, sync_wire.c -- the same decoder the conformance job
//! checks byte-for-byte against DUCTEI's Rust encoder), and merges
//! ENVELOPE frames into the real LMDB-backed store via
//! ql_persist_merge_blob().
//!
//! Reject-before-write (Qallow governance): a malformed frame anywhere
//! in the stream aborts the whole ingest with a plain-English error and
//! a nonzero exit before any further merge is attempted. Frames already
//! merged earlier in the same call stay merged -- last-writer-wins is
//! defined per-key, not per-batch, so there is nothing to roll back.
use std::ffi::CString;
use std::fs;
use std::os::raw::{c_char, c_void};
use std::path::Path;

const QSW_F_ENVELOPE: u8 = 3;
const QSW_NEED_MORE: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct QswHello {
    magic: u32,
    proto_ver: u16,
    caps: u16,
    node_id: [u8; 16],
    lamport: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QswEnvelope {
    node_id: [u8; 16],
    lamport: u64,
    schema_ver: u16,
    flags: u16,
    key_len: u32,
    blob_len: u32,
    key: *const c_void,
    blob: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
union QswFrameUnion {
    hello: QswHello,
    env: QswEnvelope,
    lamport: u64,
}

#[repr(C)]
struct QswFrame {
    frame_type: u8,
    u: QswFrameUnion,
}

#[repr(C)]
struct QlPersistStore {
    _private: [u8; 0],
}

extern "C" {
    fn qsw_decode(
        buf: *const u8,
        len: usize,
        frame: *mut QswFrame,
        consumed: *mut usize,
    ) -> i32;

    fn ql_persist_open(dir_path: *const c_char, out: *mut *mut QlPersistStore) -> i32;
    fn ql_persist_close(store: *mut QlPersistStore);
    fn ql_persist_merge_blob(
        store: *mut QlPersistStore,
        env: *const QswEnvelope,
        out_applied: *mut bool,
    ) -> i32;
    fn ql_persist_get(
        store: *mut QlPersistStore,
        key: *const c_void,
        key_len: u32,
        out_val: *mut c_void,
        cap: u32,
        out_len: *mut u32,
    ) -> i32;
}

const QLP_E_IO: i32 = -4;

/// Best-effort readable rendering of a borrowed key/blob pointer pair
/// for the witness log -- falls back to a byte count if not valid UTF-8.
unsafe fn describe(ptr: *const c_void, len: u32) -> String {
    let bytes = std::slice::from_raw_parts(ptr as *const u8, len as usize);
    match std::str::from_utf8(bytes) {
        Ok(s) => format!("\"{s}\""),
        Err(_) => format!("<{len} bytes>"),
    }
}

fn open_store(store_dir: &str) -> Result<*mut QlPersistStore, String> {
    fs::create_dir_all(store_dir)
        .map_err(|e| format!("could not create store dir {store_dir}: {e}"))?;
    let store_dir_c = CString::new(store_dir)
        .map_err(|_| "store_dir contains an interior NUL byte".to_string())?;
    let mut store: *mut QlPersistStore = std::ptr::null_mut();
    let rc = unsafe { ql_persist_open(store_dir_c.as_ptr(), &mut store) };
    if rc != 0 {
        return Err(format!("ql_persist_open({store_dir}) failed: qlp_status={rc}"));
    }
    Ok(store)
}

pub fn run(store_dir: &str, frame_file: &str) -> Result<(), String> {
    let store = open_store(store_dir)?;
    let result = ingest_frames(store, frame_file);
    unsafe { ql_persist_close(store) };
    result
}

/// `qallow get <store_dir> <key>`: reads a key back out of the real LMDB
/// store. Prints `FOUND:<value>` (value rendered as UTF-8 if valid, else
/// hex) or `NOT_FOUND` -- both are successful, informational outcomes;
/// only a store-open failure is an error.
pub fn run_get(store_dir: &str, key: &str) -> Result<(), String> {
    let store = open_store(store_dir)?;
    let key_bytes = key.as_bytes();
    let mut buf = vec![0u8; 1 << 20];
    let mut out_len: u32 = 0;
    let rc = unsafe {
        ql_persist_get(
            store,
            key_bytes.as_ptr() as *const c_void,
            key_bytes.len() as u32,
            buf.as_mut_ptr() as *mut c_void,
            buf.len() as u32,
            &mut out_len,
        )
    };
    unsafe { ql_persist_close(store) };

    if rc == QLP_E_IO {
        println!("NOT_FOUND");
        return Ok(());
    }
    if rc != 0 {
        return Err(format!("ql_persist_get failed: qlp_status={rc}"));
    }
    buf.truncate(out_len as usize);
    match std::str::from_utf8(&buf) {
        Ok(s) => println!("FOUND:{s}"),
        Err(_) => println!("FOUND:hex:{}", buf.iter().map(|b| format!("{b:02x}")).collect::<String>()),
    }
    Ok(())
}

fn ingest_frames(store: *mut QlPersistStore, frame_file: &str) -> Result<(), String> {
    let path = Path::new(frame_file);
    let buf = fs::read(path).map_err(|e| format!("could not read {frame_file}: {e}"))?;

    let mut offset = 0usize;
    let mut merged = 0u32;
    let mut skipped = 0u32;

    while offset < buf.len() {
        let mut frame: QswFrame = unsafe { std::mem::zeroed() };
        let mut consumed: usize = 0;
        let status = unsafe {
            qsw_decode(
                buf[offset..].as_ptr(),
                buf.len() - offset,
                &mut frame,
                &mut consumed,
            )
        };

        if status == QSW_NEED_MORE {
            return Err(format!(
                "truncated frame at byte offset {offset} ({} bytes remaining, not enough for a full frame) -- nothing further merged",
                buf.len() - offset
            ));
        }
        if status < 0 {
            return Err(format!(
                "malformed frame at byte offset {offset}: qsw_status={status} -- nothing further merged"
            ));
        }
        if consumed == 0 {
            // Decoder made no progress on an OK status: refuse to spin.
            return Err(format!("decoder made no progress at byte offset {offset}"));
        }

        if frame.frame_type == QSW_F_ENVELOPE {
            let env = unsafe { frame.u.env };
            let mut applied = false;
            let rc = unsafe { ql_persist_merge_blob(store, &env, &mut applied) };
            if rc != 0 {
                return Err(format!(
                    "ql_persist_merge_blob failed at byte offset {offset}: qlp_status={rc} -- nothing further merged"
                ));
            }
            let key_desc = unsafe { describe(env.key, env.key_len) };
            println!(
                "ingest: key={key_desc} lamport={} applied={applied}",
                env.lamport
            );
            merged += 1;
        } else {
            println!("ingest: skipping non-envelope frame type {}", frame.frame_type);
            skipped += 1;
        }

        offset += consumed;
    }

    println!("ingest: {merged} envelope(s) merged, {skipped} frame(s) skipped, {offset} bytes read");
    Ok(())
}
