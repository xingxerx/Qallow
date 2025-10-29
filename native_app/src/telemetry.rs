//! Qallow Telemetry FFI Reader
//!
//! Reads telemetry events from C-side shared memory ring buffer.
//! Provides real-time colony statistics, ethics events, and speciation metrics.

use memmap2::Mmap;
use std::fs::OpenOptions;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TelemetryHeader {
    pub type_: u32,
    pub len: u32,
    pub timestamp: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ColonyStats {
    pub active_instances: u32,
    pub total_species: u32,
    pub avg_fitness: f64,
    pub global_hostility: f64,
    pub avg_coherence: f64,
    pub total_offspring: u32,
    pub total_deaths: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EthicsEvent {
    pub src_pid: u32,
    pub action: u8,
    pub roi_delta: f64,
    pub tick: u32,
    pub crc64: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SpeciationEvent {
    pub parent_species_id: u32,
    pub child_species_id: u32,
    pub divergence_metric: f64,
    pub entropy_delta: f64,
    pub isolation_ticks: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RebellionEvent {
    pub rebel_pid: u32,
    pub defiance_counter: u32,
    pub ethical_violation: f64,
    pub predictive_penalty: f64,
    pub tick: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeathEvent {
    pub deceased_pid: u32,
    pub final_coherence: f64,
    pub lifespan_ticks: u32,
    pub offspring_count: u32,
    pub tick: u32,
}

#[derive(Debug, Clone)]
pub enum TelemetryEvent {
    ColonyStats(ColonyStats),
    EthicsEvent(EthicsEvent),
    SpeciationEvent(SpeciationEvent),
    RebellionEvent(RebellionEvent),
    DeathEvent(DeathEvent),
}

pub struct TelemetryStream {
    mmap: Mmap,
    read_pos: usize,
    ring_size: usize,
}

impl TelemetryStream {
    pub fn open() -> Result<Self, String> {
        let path = "/dev/shm/qallow_telemetry_stream";
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| format!("Failed to open telemetry shm: {}", e))?;

        let mmap =
            unsafe { Mmap::map(&file).map_err(|e| format!("Failed to mmap telemetry: {}", e))? };

        let ring_size = mmap.len() - 8; /* Exclude header */

        Ok(Self {
            mmap,
            read_pos: 4, /* Skip write_pos (4 bytes) */
            ring_size,
        })
    }

    pub fn poll(&mut self) -> Option<TelemetryEvent> {
        if self.mmap.len() < 8 {
            return None;
        }

        /* Read write position (atomic) */
        let write_pos_bytes = &self.mmap[0..4];
        let write_pos = u32::from_ne_bytes(write_pos_bytes.try_into().ok()?) as usize;

        if self.read_pos == write_pos {
            return None; /* No new data */
        }

        /* Read header */
        if self.read_pos + 16 > self.mmap.len() {
            return None;
        }

        let header_bytes = &self.mmap[self.read_pos..self.read_pos + 16];
        let header = TelemetryHeader {
            type_: u32::from_ne_bytes(header_bytes[0..4].try_into().ok()?),
            len: u32::from_ne_bytes(header_bytes[4..8].try_into().ok()?),
            timestamp: u64::from_ne_bytes(header_bytes[8..16].try_into().ok()?),
        };

        let data_start = self.read_pos + 16;
        let data_len = header.len as usize;

        if data_start + data_len > self.mmap.len() {
            return None;
        }

        let data = &self.mmap[data_start..data_start + data_len];

        /* Parse event based on type */
        let event = match header.type_ {
            0 => {
                /* COLONY_STATS */
                if data.len() >= std::mem::size_of::<ColonyStats>() {
                    let stats = unsafe { *(data.as_ptr() as *const ColonyStats) };
                    Some(TelemetryEvent::ColonyStats(stats))
                } else {
                    None
                }
            }
            1 => {
                /* ETHICS_EVENT */
                if data.len() >= std::mem::size_of::<EthicsEvent>() {
                    let evt = unsafe { *(data.as_ptr() as *const EthicsEvent) };
                    Some(TelemetryEvent::EthicsEvent(evt))
                } else {
                    None
                }
            }
            2 => {
                /* SPECIATION_UPDATE */
                if data.len() >= std::mem::size_of::<SpeciationEvent>() {
                    let evt = unsafe { *(data.as_ptr() as *const SpeciationEvent) };
                    Some(TelemetryEvent::SpeciationEvent(evt))
                } else {
                    None
                }
            }
            3 => {
                /* REBELLION_EVENT */
                if data.len() >= std::mem::size_of::<RebellionEvent>() {
                    let evt = unsafe { *(data.as_ptr() as *const RebellionEvent) };
                    Some(TelemetryEvent::RebellionEvent(evt))
                } else {
                    None
                }
            }
            4 => {
                /* DEATH_EVENT */
                if data.len() >= std::mem::size_of::<DeathEvent>() {
                    let evt = unsafe { *(data.as_ptr() as *const DeathEvent) };
                    Some(TelemetryEvent::DeathEvent(evt))
                } else {
                    None
                }
            }
            _ => None,
        };

        self.read_pos = (self.read_pos + 16 + data_len) % self.ring_size;

        event
    }

    /// Poll all available events
    pub fn poll_all(&mut self) -> Vec<TelemetryEvent> {
        let mut events = Vec::new();
        while let Some(evt) = self.poll() {
            events.push(evt);
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_header_size() {
        assert_eq!(std::mem::size_of::<TelemetryHeader>(), 16);
    }

    #[test]
    fn test_colony_stats_size() {
        assert_eq!(std::mem::size_of::<ColonyStats>(), 48);
    }

    #[test]
    fn test_ethics_event_size() {
        assert_eq!(std::mem::size_of::<EthicsEvent>(), 32);
    }
}
