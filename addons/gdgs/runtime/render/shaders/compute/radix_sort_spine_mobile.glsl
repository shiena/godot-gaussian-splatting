#[compute]
#version 460 core

/**
 * Subgroup-free radix sort spine (prefix sum).
 * Blelloch scan inlined in main() to avoid Adreno barrier-in-function bugs.
 *
 * Based on vulkan_radix_sort (MIT License).
 */

#define RADIX              (256)
#define WORKGROUP_SIZE     (512)
#define PARTITION_DIVISION (8)
#define PARTITION_SIZE     (PARTITION_DIVISION * WORKGROUP_SIZE)

layout (local_size_x = WORKGROUP_SIZE) in;

layout (std430, set = 0, binding = 0) restrict buffer Histogram {
    uint element_count;
    uint sort_overflow_count;
    uint global_histogram[4*RADIX];
    uint parition_histogram[PARTITION_SIZE*RADIX];
};

layout (push_constant) uniform PushConstant {
    int pass;
};

shared uint scan_data[WORKGROUP_SIZE];
shared uint block_total;
shared uint carry;

void main() {
    uint index = gl_LocalInvocationIndex;
    uint radix = gl_WorkGroupID.x;

    uint ec = element_count;
    uint partition_count = (ec + PARTITION_SIZE - 1u) / PARTITION_SIZE;

    if (index == 0u) carry = 0u;
    barrier();

    for (uint iter = 0u; WORKGROUP_SIZE * iter < partition_count; ++iter) {
        uint partition_index = WORKGROUP_SIZE * iter + index;
        scan_data[index] = partition_index < partition_count
            ? parition_histogram[RADIX * partition_index + radix]
            : 0u;
        barrier();

        // === Inlined Blelloch exclusive scan (up-sweep) ===
        for (uint stride = 1u; stride < WORKGROUP_SIZE; stride <<= 1u) {
            if (index < (WORKGROUP_SIZE >> 1u) / stride) {
                uint ai = (index + 1u) * (stride << 1u) - 1u;
                scan_data[ai] += scan_data[ai - stride];
            }
            barrier();
        }
        if (index == 0u) {
            block_total = scan_data[WORKGROUP_SIZE - 1u];
            scan_data[WORKGROUP_SIZE - 1u] = 0u;
        }
        barrier();
        // === Inlined Blelloch exclusive scan (down-sweep) ===
        for (uint stride = WORKGROUP_SIZE >> 1u; stride >= 1u; stride >>= 1u) {
            if (index < (WORKGROUP_SIZE >> 1u) / stride) {
                uint ai = (index + 1u) * (stride << 1u) - 1u;
                uint temp = scan_data[ai - stride];
                scan_data[ai - stride] = scan_data[ai];
                scan_data[ai] += temp;
            }
            barrier();
        }
        // === End Blelloch scan ===

        uint carry_val = carry;
        barrier();

        if (partition_index < partition_count) {
            parition_histogram[RADIX * partition_index + radix] = scan_data[index] + carry_val;
        }

        if (index == 0u) carry = carry_val + block_total;
        barrier();
    }

    // Workgroup 0: exclusive prefix sum of global histogram
    if (radix == 0u) {
        scan_data[index] = index < RADIX
            ? global_histogram[RADIX * pass + index]
            : 0u;
        barrier();

        // === Inlined Blelloch exclusive scan (up-sweep) ===
        for (uint stride = 1u; stride < WORKGROUP_SIZE; stride <<= 1u) {
            if (index < (WORKGROUP_SIZE >> 1u) / stride) {
                uint ai = (index + 1u) * (stride << 1u) - 1u;
                scan_data[ai] += scan_data[ai - stride];
            }
            barrier();
        }
        if (index == 0u) scan_data[WORKGROUP_SIZE - 1u] = 0u;
        barrier();
        // === Inlined Blelloch exclusive scan (down-sweep) ===
        for (uint stride = WORKGROUP_SIZE >> 1u; stride >= 1u; stride >>= 1u) {
            if (index < (WORKGROUP_SIZE >> 1u) / stride) {
                uint ai = (index + 1u) * (stride << 1u) - 1u;
                uint temp = scan_data[ai - stride];
                scan_data[ai - stride] = scan_data[ai];
                scan_data[ai] += temp;
            }
            barrier();
        }
        // === End Blelloch scan ===

        if (index < RADIX) {
            global_histogram[RADIX * pass + index] = scan_data[index];
        }
    }
}
