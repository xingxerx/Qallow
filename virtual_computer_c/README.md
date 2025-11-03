# Virtual Computer System - C Implementation

A high-performance simulation of CUDA GPU, neuromorphic, and photonic processors for Lightning Agent optimization.

## Architecture Overview

### Components

1. **CUDA GPU Simulator** (`cuda_simulator.c/h`)
   - Device memory management
   - Kernel execution simulation
   - PCIe transfer modeling
   - Performance telemetry

2. **Neuromorphic Processor** (`neuromorphic_simulator.c/h`)
   - Spiking Neural Networks (SNNs)
   - Leaky Integrate-and-Fire (LIF) neurons
   - Synaptic plasticity (STDP)
   - Event-based processing

3. **Photonic Processor** (`photonic_simulator.c/h`)
   - Optical waveguides
   - Photonic gates (Mach-Zehnder, beam splitters, etc.)
   - Photon propagation and detection
   - Wavelength multiplexing

4. **Virtual Computer Orchestrator** (`virtual_computer.c/h`)
   - Unified workload scheduling
   - Multi-processor coordination
   - System-wide metrics

## Building

```bash
cd virtual_computer_c
mkdir build
cd build
cmake ..
make
```

## Running the Demo

```bash
./virtual_computer_demo
```

Output shows:
- System initialization
- Workload creation and execution
- Processor performance metrics
- System throughput and energy consumption

## API Reference

### CUDA GPU Simulator

```c
/* Create GPU device */
virtual_gpu_t* gpu_create(uint32_t device_id, uint32_t device_memory_mb);

/* Allocate device memory */
uint64_t gpu_malloc(virtual_gpu_t *gpu, size_t size, const char *data_type);

/* Free device memory */
bool gpu_free(virtual_gpu_t *gpu, uint64_t address);

/* Transfer data host->device */
bool gpu_memcpy_to_device(virtual_gpu_t *gpu, uint64_t address, size_t size,
                          double *transfer_time_ms);

/* Transfer data device->host */
bool gpu_memcpy_to_host(virtual_gpu_t *gpu, uint64_t address, size_t size,
                        double *transfer_time_ms);

/* Launch CUDA kernel */
uint32_t gpu_launch_kernel(virtual_gpu_t *gpu, const char *kernel_name,
                          uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                          uint32_t block_x, uint32_t block_y, uint32_t block_z,
                          size_t shared_memory);

/* Execute kernel */
bool gpu_execute_kernel(virtual_gpu_t *gpu, uint32_t kernel_id,
                       uint64_t compute_ops, double *execution_time_ms);

/* Get statistics */
void gpu_get_device_properties(virtual_gpu_t *gpu, char *buffer, size_t buffer_size);
void gpu_get_memory_stats(virtual_gpu_t *gpu, char *buffer, size_t buffer_size);
void gpu_get_kernel_stats(virtual_gpu_t *gpu, char *buffer, size_t buffer_size);
```

### Neuromorphic Processor

```c
/* Create processor */
neuromorphic_processor_t* nm_create(uint32_t num_neurons, uint32_t num_layers);

/* Inject spikes */
void nm_inject_spikes(neuromorphic_processor_t *nm, const uint32_t *neuron_ids,
                     uint32_t count, double current_time);

/* Simulate one time step */
void nm_simulate_step(neuromorphic_processor_t *nm, double current_time,
                     bool inject_input);

/* Get statistics */
void nm_get_stats(neuromorphic_processor_t *nm, char *buffer, size_t buffer_size);
void nm_get_connectivity_stats(neuromorphic_processor_t *nm, char *buffer,
                              size_t buffer_size);
```

### Photonic Processor

```c
/* Create processor */
photonic_processor_t* pp_create(uint32_t num_waveguides, uint32_t num_gates);

/* Inject photons */
uint32_t pp_inject_photons(photonic_processor_t *pp, uint32_t count,
                          double power_dbm, double wavelength_nm,
                          uint32_t *photon_ids, uint32_t max_ids);

/* Apply gate operation */
void pp_apply_gate_operation(photonic_processor_t *pp, const uint32_t *photon_ids,
                            uint32_t count, uint32_t gate_id);

/* Detect photons */
uint32_t pp_detect_photons(photonic_processor_t *pp, const uint32_t *photon_ids,
                          uint32_t count, uint32_t *detected_ids,
                          uint32_t max_detected);

/* Get statistics */
void pp_get_processor_stats(photonic_processor_t *pp, char *buffer, size_t buffer_size);
void pp_get_gate_stats(photonic_processor_t *pp, char *buffer, size_t buffer_size);
void pp_get_waveguide_stats(photonic_processor_t *pp, char *buffer, size_t buffer_size);
```

### Virtual Computer

```c
/* Create system */
virtual_computer_t* vc_create(void);

/* Create workload */
uint32_t vc_create_workload(virtual_computer_t *vc, workload_type_t type,
                           uint32_t priority, size_t data_size_mb,
                           uint64_t compute_ops);

/* Execute workloads */
void vc_run_scheduled_workloads(virtual_computer_t *vc);

/* Get status */
void vc_get_system_status(virtual_computer_t *vc, char *buffer, size_t buffer_size);
void vc_print_system_status(virtual_computer_t *vc);
```

## Performance Metrics

### GPU Metrics
- Memory utilization (%)
- Kernel execution time (ms)
- PCIe bandwidth (GB/s)
- Peak memory usage (MB)
- Occupancy (%)

### Neuromorphic Metrics
- Spike rate (Hz)
- Energy consumption (µJ)
- Network connectivity ratio
- Synaptic weights distribution
- Simulation speed (steps/sec)

### Photonic Metrics
- Detection efficiency (%)
- Insertion loss (dB)
- Photon throughput (photons/sec)
- Wavelength utilization (%)
- Gate switching latency (ns)

## System Specifications

**CUDA GPU**
- Device Memory: 8192 MB
- Bandwidth: 288 GB/s (Ampere)
- SMs: 80
- Cores/SM: 128
- Total Cores: 10,240

**Neuromorphic Processor**
- Neurons: 1000
- Layers: 4
- Connectivity: 10%
- Neuron Type: LIF
- Tau Membrane: 20 ms

**Photonic Processor**
- Waveguides: 64
- Gates: 256
- Wavelength: 1550 nm
- Propagation Loss: 0.2 dB/km
- Quantum Efficiency: 95%

## Integration with Lightning Agent

The Virtual Computer provides optimization targets for Lightning Agent:

1. **Memory Optimization** - Reduce GPU memory fragmentation
2. **Kernel Performance** - Improve kernel execution time
3. **Neuromorphic Efficiency** - Optimize spike rate and energy
4. **Photonic Fidelity** - Improve detection efficiency
5. **Hybrid Coordination** - Optimize multi-processor scheduling

## File Structure

```
virtual_computer_c/
├── CMakeLists.txt               # Build configuration
├── cuda_simulator.h/c           # GPU simulation (~500 lines)
├── neuromorphic_simulator.h/c   # Neural network simulation (~600 lines)
├── photonic_simulator.h/c       # Optical processor simulation (~550 lines)
├── virtual_computer.h/c         # Orchestration layer (~400 lines)
├── main.c                       # Demo program (~200 lines)
└── README.md                    # This file
```

## Performance Characteristics

- **Memory Usage**: ~50-100 MB for full system
- **Simulation Speed**: 1-10 ms per workload
- **Scalability**: Handles 100+ simultaneous workloads
- **Real-time**: All simulations are deterministic

## Compilation Time

```
Total lines of code: ~2,250 lines
Compile time: <1 second
Binary size: ~500 KB
```

## Example Usage

```c
#include "virtual_computer.h"

int main() {
    /* Create system */
    virtual_computer_t *vc = vc_create();
    
    /* Create workload */
    vc_create_workload(vc, WORKLOAD_GPU_COMPUTE, 5, 512, 5_000_000_000);
    
    /* Execute */
    vc_run_scheduled_workloads(vc);
    
    /* Get results */
    vc_print_system_status(vc);
    
    /* Cleanup */
    vc_destroy(vc);
    
    return 0;
}
```

## License

Same as Qallow project
