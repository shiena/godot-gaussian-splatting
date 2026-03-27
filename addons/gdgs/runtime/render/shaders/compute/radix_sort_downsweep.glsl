#[compute]
#version 460 core

/**
 * Subgroup-free STABLE radix sort downsweep.
 * Computes per-element rank by counting preceding same-radix elements
 * in shared memory — no subgroup operations required.
 *
 * Based on vulkan_radix_sort (MIT License).
 */

#define RADIX              (256)
#define WORKGROUP_SIZE     (512)
#define PARTITION_DIVISION (8)
#define PARTITION_SIZE     (PARTITION_DIVISION * WORKGROUP_SIZE)

layout (local_size_x = WORKGROUP_SIZE) in;

layout (std430, set = 0, binding = 0) restrict readonly buffer Histogram {
    uint element_count;
    uint global_histogram[4*RADIX];
    uint partition_histogram[PARTITION_SIZE*RADIX];
};

layout (std430, set = 0, binding = 1) restrict buffer KeysBuffer {
    uint keys[];
};

layout (std430, set = 0, binding = 2) restrict buffer ValuesBuffer {
    uint values[];
};

layout (push_constant) uniform PushConstant {
    int pass;
    uint in_offset;
    uint out_offset;
};

shared uint shared_radix[WORKGROUP_SIZE];   // Radix values for current pass (2KB)
shared uint running_count[RADIX];           // Per-radix running offset (1KB)
shared uint global_base[RADIX];             // Global base offset per radix (1KB)

void main() {
    uint index = gl_LocalInvocationIndex;
    uint partition_index = gl_WorkGroupID.x;
    uint partition_start = partition_index * PARTITION_SIZE;
    uint ec = element_count;

    if (partition_start >= ec) return;

    // Phase 1: Load global base offset and clear running count
    if (index < RADIX) {
        global_base[index] = global_histogram[RADIX * pass + index]
                           + partition_histogram[RADIX * partition_index + index];
        running_count[index] = 0u;
    }
    barrier();

    // Phase 2: Process elements in PARTITION_DIVISION sequential passes.
    // Within each pass, compute stable rank by counting preceding same-radix elements.
    for (int p = 0; p < PARTITION_DIVISION; ++p) {
        uint key_index = partition_start + p * WORKGROUP_SIZE + index;
        uint key = key_index < ec ? keys[key_index + in_offset] : 0xFFFFFFFFu;
        uint val = key_index < ec ? values[key_index + in_offset] : 0u;
        uint radix = bitfieldExtract(key, pass * 8, 8);

        // Write radix to shared memory so all threads can see it
        shared_radix[index] = radix;
        barrier();

        // Count how many threads with LOWER index in this pass have the same radix.
        // This gives a deterministic, stable rank within this pass.
        uint rank = 0u;
        for (uint j = 0u; j < index; j++) {
            if (shared_radix[j] == radix) rank++;
        }

        // Final offset = global_base + running_count (from previous passes) + rank
        uint offset = running_count[radix] + rank;
        barrier();

        // Update running_count for next pass (only one thread per radix needs to add)
        // Thread with highest index for each radix writes the new count
        // (rank + 1 = count of this radix in this pass up to and including this thread)
        // Use atomicMax to find the highest rank+1 for each radix
        atomicMax(running_count[radix], offset + 1u);
        barrier();

        // Scatter
        if (key_index < ec) {
            uint dst = global_base[radix] + offset;
            keys[dst + out_offset] = key;
            values[dst + out_offset] = val;
        }
        barrier();
    }
}
