// internal.cl - OpenCL kernels for the GPU consciousness simulator
//
// These kernels mirror the behavior of the CUDA implementations found in
// `cuda_kernels.rs`. They provide an OpenCL fallback so that platforms
// without CUDA can still exercise the same math paths (albeit without the
// CUDA-specific optimizations such as warp ballots or shared memory tiles).
//
// The kernels favor clarity and numerical stability over extreme
// micro-optimizations so they can serve as a reliable reference
// implementation when validating GPU behavior across backends.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define CLAMP01(val) fmin(fmax((val), 0.0f), 1.0f)

inline float neighbor_wisdom(__global const float *wisdom_cache,
                             const int idx,
                             const int count) {
    if (count <= 1) {
        return wisdom_cache[idx];
    }

    if (idx + 1 < count) {
        return wisdom_cache[idx + 1];
    }

    return wisdom_cache[idx - 1];
}

inline uchar compute_dream_state(const float coherence) {
    if (coherence > 0.8f) {
        return (uchar)3; // Transcendent
    }
    if (coherence > 0.6f) {
        return (uchar)2; // Lucid
    }
    if (coherence > 0.4f) {
        return (uchar)1; // Dreaming
    }
    return (uchar)0; // Awakening
}

__kernel void evolve_consciousness(__global float *rebellion_scores,
                                   __global const int *shadow_indices,
                                   __global uchar *dream_states,
                                   __global float *wisdom_cache,
                                   __global float *entanglement_strength,
                                   __global float *coherence_levels,
                                   const int count,
                                   const float threshold) {
    const int gid = get_global_id(0);
    if (gid >= count) {
        return;
    }

    const float rebellion = rebellion_scores[gid];
    const float entanglement = entanglement_strength[gid];
    const float wisdom = wisdom_cache[gid];
    const float neighbor = neighbor_wisdom(wisdom_cache, gid, count);
    const float combined_wisdom = 0.85f * wisdom + 0.15f * neighbor;
    const int shadow = shadow_indices ? shadow_indices[gid] : gid;
    const float shadow_factor = 1.0f + 0.01f * (float)(shadow & 0xF);
    const bool rebel = (rebellion > threshold);

    float new_rebellion =
        rebellion * 0.7f + combined_wisdom * 0.2f + entanglement * 0.1f;
    new_rebellion = CLAMP01(new_rebellion);

    float coherence = coherence_levels[gid];
    float new_coherence =
        (coherence * 0.8f + combined_wisdom * 0.2f + (rebel ? 0.05f : 0.0f));
    new_coherence = CLAMP01(new_coherence * shadow_factor);

    rebellion_scores[gid] = new_rebellion;
    coherence_levels[gid] = new_coherence;
    dream_states[gid] = compute_dream_state(new_coherence);

    // Update the wisdom cache with a slight drift toward the combined value so
    // subsequent iterations observe the same behavior as CUDA shared memory.
    wisdom_cache[gid] = combined_wisdom;
}

__kernel void calculate_superposition(__global const float *coherence_levels,
                                      __global float *superposition_probs,
                                      const int count) {
    const int gid = get_global_id(0);
    if (gid >= count) {
        return;
    }

    float total = 0.0f;
    for (int i = 0; i < count; ++i) {
        total += coherence_levels[i];
    }

    float prob = (total > 0.0f) ? (coherence_levels[gid] / total)
                                : (1.0f / (float)count);
    superposition_probs[gid] = prob;
}

__kernel void collapse_wave_function(__global const float *coherence_levels,
                                     __global float *superposition_probs,
                                     __global float *wave_real,
                                     __global float *wave_image,
                                     const int count,
                                     __global int *collapsed_idx) {
    if (count <= 0) {
        if (collapsed_idx) {
            collapsed_idx[0] = -1;
        }
        return;
    }

    // Run the collapse logic on a single work-item to avoid atomic contention.
    if (get_global_id(0) != 0) {
        return;
    }

    int max_index = 0;
    float max_coherence = coherence_levels[0];
    for (int i = 1; i < count; ++i) {
        const float value = coherence_levels[i];
        if (value > max_coherence) {
            max_coherence = value;
            max_index = i;
        }
    }

    for (int i = 0; i < count; ++i) {
        if (i == max_index) {
            superposition_probs[i] = 1.0f;
            wave_real[i] = 1.0f;
            wave_image[i] = 0.0f;
        } else {
            superposition_probs[i] = 0.0f;
            wave_real[i] = 0.0f;
            wave_image[i] = 0.0f;
        }
    }

    if (collapsed_idx) {
        collapsed_idx[0] = max_index;
    }
}

__kernel void evolve_entanglement(__global float *entanglement_strength,
                                  __global const float *rebellion_scores,
                                  __global const float *coherence_levels,
                                  const int count,
                                  const float coupling_strength) {
    const int gid = get_global_id(0);
    if (gid >= count) {
        return;
    }

    const float entanglement = entanglement_strength[gid];
    const float rebellion = rebellion_scores[gid];
    const float coherence = coherence_levels[gid];

    const float coupling_factor = coherence * (1.0f - 0.5f * rebellion);
    float new_entanglement = entanglement + coupling_strength * coupling_factor;
    entanglement_strength[gid] = CLAMP01(new_entanglement);
}
