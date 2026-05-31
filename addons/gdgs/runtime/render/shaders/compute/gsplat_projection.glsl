// Multiview gaussian splat projection.
//
// Mono (view_count=1): identical to single-view — no stereo overhead.
// Stereo (view_count=2): projects both eyes in one dispatch. Frustum
// culling uses an expanded clip-space margin so that splats near the
// primary eye's frustum edge are not discarded when they fall inside the
// secondary eye's FOV due to IPD offset.  Sorting uses only the primary
// (left) eye depth — averaged with the right eye's depth in stereo for
// a center-eye sort key — so it runs once for both views.
// The right eye shares the 2D covariance from the left eye and only
// recomputes clip position + depth (the IPD-induced difference in
// screen-space covariance is negligible for typical stereo baselines).
//
// culled_buffer layout: interleaved [left_0, right_0, left_1, right_1, ...]
//   index = splat_id * view_count + eye_index
//
// Reference: arghyasur1991/UnityGaussianSplatting (MIT License)
// https://github.com/arghyasur1991/UnityGaussianSplatting

#[compute]
#version 460

#define SH_C0 0.28209479177387814
#define SH_C1 0.4886025119029199

#define SH_C2_0 1.0925484305920792
#define SH_C2_1 1.0925484305920792
#define SH_C2_2 0.31539156525252005
#define SH_C2_3 1.0925484305920792
#define SH_C2_4 0.5462742152960396

#define SH_C3_0 0.5900435899266435
#define SH_C3_1 2.890611442640554
#define SH_C3_2 0.4570457994644658
#define SH_C3_3 0.3731763325901154
#define SH_C3_4 0.4570457994644658
#define SH_C3_5 1.445305721320277
#define SH_C3_6 0.5900435899266435

#define TILE_SIZE                (16)
#define NUM_BLOCKS_PER_WORKGROUP (32)
#define SORT_WORKGROUP_SIZE      (512)
#define SORT_PARTITION_DIVISION  (8)
#define SORT_PARTITION_SIZE      (SORT_PARTITION_DIVISION * SORT_WORKGROUP_SIZE)

#define DECODE_COVARIANCE(c) (mat3(c[0], c[1], c[2], c[1], c[3], c[4], c[2], c[4], c[5]))

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

struct Splat {
	vec3 position;
	float time;
	float covariance[6]; // Contains top triangle of symmetric matrix
	float opacity;
	float _pad;
	float sh_coefficients[16*3]; // Spherical harmonic coefficients in increasing order
};

struct RasterizeData {
	vec2 image_pos;
    vec2 pos_xy;
	vec3 conic;
	float pos_z;
	vec4 color;
	vec4 depth_data;
};

layout(std430, set = 0, binding = 0) restrict readonly buffer SplatsBuffer {
	Splat splat_buffer[];
};

layout(std430, set = 0, binding = 1) restrict writeonly buffer CulledBuffer {
	RasterizeData culled_buffer[];
};

layout (std430, set = 0, binding = 2) restrict buffer Histograms {
	uint sort_buffer_size;
	uint sort_overflow_count;
	uint histogram[];
};

layout (std430, set = 0, binding = 3) restrict writeonly buffer SortKeysBuffer {
    uint sort_keys[];
};

layout (std430, set = 0, binding = 4) restrict writeonly buffer SortValuesBuffer {
    uint sort_values[];
};

layout (std430, set = 0, binding = 5) restrict buffer GridDimensionsBuffer {
	uint grid_dims[];
};

layout (std430, set = 0, binding = 6) restrict readonly buffer SplatInstanceIdsBuffer {
	uvec2 splat_instance_data[]; // x = unique instance id, y = which splat data to use
};

layout (std430, set = 0, binding = 7) restrict readonly buffer InstanceTransformsBuffer {
	mat4 instance_model_matrices[];
};

layout(push_constant) uniform PushConstant {
	uint mode;       // 0=project, 1=clear
	uint sh_degree;  // 0-3: spherical harmonics evaluation degree
	float min_radius; // minimum projected screen-space radius in pixels
	uint _pad0;
};

layout (std140, set = 0, binding = 8) restrict uniform Uniforms {
	vec3 camera_pos;
	float time;
	ivec2 dims; // Texture size
	int point_count;
	int view_count; // 1 = mono, 2 = stereo
	int sort_capacity;
	int _pad1;
	int _pad2;
	int _pad3;
	mat4 view_matrix;
	mat4 projection_matrix;
	mat4 view_matrix_right;
	mat4 projection_matrix_right;
};

float ease_out_cubic(in float x) {
	float a = 1.0 - x;
	return 1.0 - a*a*a;
}

/** Calculates the color from given spherical harmonic coefficients and view direction. */
#define SH_COEFFICIENTS(x) (vec3(sh_coefficients[x*3], sh_coefficients[x*3+1], sh_coefficients[x*3+2]))
vec3 get_color(in vec3 view_dir, in float sh_coefficients[16*3], in uint degree) {
	vec3 result = SH_COEFFICIENTS(0) * SH_C0;

	if (degree >= 1u) {
		const float x = view_dir.x, y = view_dir.y, z = view_dir.z;
		result += - SH_COEFFICIENTS(1) * SH_C1 * y
		          + SH_COEFFICIENTS(2) * SH_C1 * z
		          - SH_COEFFICIENTS(3) * SH_C1 * x;

		if (degree >= 2u) {
			const float xx = x*x, yy = y*y, zz = z*z,
			            xy = x*y, yz = y*z, xz = x*z;
			result += + SH_COEFFICIENTS(4) * SH_C2_0 * xy
			          - SH_COEFFICIENTS(5) * SH_C2_1 * yz
			          + SH_COEFFICIENTS(6) * SH_C2_2 * (2.0*zz - xx - yy)
			          - SH_COEFFICIENTS(7) * SH_C2_3 * xz
			          + SH_COEFFICIENTS(8) * SH_C2_4 * (xx - yy);

			if (degree >= 3u) {
				result += - SH_COEFFICIENTS(9)  * SH_C3_0 * y * (3.0*xx - yy)
				          + SH_COEFFICIENTS(10) * SH_C3_1 * x * yz
				          - SH_COEFFICIENTS(11) * SH_C3_2 * y * (4.0*zz - xx - yy)
				          + SH_COEFFICIENTS(12) * SH_C3_3 * z * (2.0*zz - 3.0*xx - 3.0*yy)
				          - SH_COEFFICIENTS(13) * SH_C3_4 * x * (4.0*zz - xx - yy)
				          + SH_COEFFICIENTS(14) * SH_C3_5 * z * (xx - yy)
				          - SH_COEFFICIENTS(15) * SH_C3_6 * x * (xx - 3.0*yy);
			}
		}
	}
	return max(vec3(0), 0.5 + result);
}

/** Computes a 2D projected covariance matrix from the given Gaussian parameters. */
vec3 project_covariance(in mat3 covariance_3d, in float scale_modifier, in vec3 mean, in ivec2 p_dims, in mat4 v_mat, in mat4 p_mat) {
	const mat3 cov_3d = covariance_3d * scale_modifier*scale_modifier;
	// Godot camera space looks down -Z, so use positive forward depth here.
	vec2 tan_fov_inv = vec2(p_mat[0][0], p_mat[1][1]);
	vec2 focal = vec2(p_dims - 1) * 0.5 * tan_fov_inv;
	// RenderData projections can encode a Y flip in projection_matrix[1][1].
	// Keep that sign in the focal scale, but use absolute FOV extents for clamping.
	vec2 tan_fov = 1.0 / abs(tan_fov_inv);
	float depth_inv = -1.0 / mean.z;
	focal *= depth_inv;

	mean.xy = clamp(mean.xy * depth_inv, -tan_fov * 1.3, tan_fov * 1.3);
	mat3 view_linear = mat3(v_mat);
	mat3 jacobian = mat3(
		focal.x, 0, 0,
		0, focal.y, 0,
		focal.x * mean.x, focal.y * mean.y, 0);
	mat3 screen_transform = jacobian * view_linear;
	mat3 cov_2d = screen_transform * cov_3d * transpose(screen_transform);
	return vec3(cov_2d[0][0] + 0.3, cov_2d[0][1], cov_2d[1][1] + 0.3);
}

uvec4 get_rect(in vec2 image_pos, in float radius, in uvec2 grid_size) {
	return ivec4(
		clamp(     (image_pos - radius) / TILE_SIZE,  vec2(0), grid_size),
		clamp(ceil((image_pos + radius) / TILE_SIZE), vec2(0), grid_size));
}

bool reserve_sort_slots(uint count, out uint offset) {
	uint capacity = uint(max(sort_capacity, 0));
	if (count == 0u) return false;
	if (count > capacity) {
		atomicAdd(sort_overflow_count, count);
		return false;
	}

	offset = atomicAdd(sort_buffer_size, count);
	uint next = offset + count;
	if (next > capacity || next < offset) {
		atomicAdd(sort_overflow_count, count);
		atomicMin(sort_buffer_size, capacity);
		return false;
	}
	return true;
}

void main() {
	const int id = int(gl_GlobalInvocationID.x);
	const uvec2 grid_size = (dims + TILE_SIZE - 1) / TILE_SIZE;

	// mode == 1: clear-only mode (dispatched with 4 workgroups before projection)
	// Use atomicExchange instead of plain writes — plain stores may not be
	// visible to atomicAdd in the next dispatch on Adreno GPUs even with a
	// compute barrier, because the atomic unit and the L1 store path can
	// use separate caches.
	if (mode == 1u) {
		if (id == 0) atomicExchange(sort_buffer_size, 0u);
		if (id == 1) atomicExchange(sort_overflow_count, 0u);
		if (id < 4 * 256) atomicExchange(histogram[id], 0u);
		return;
	}

	if (id >= uint(point_count)) return;

	barrier();
	uvec2 instance_data = splat_instance_data[id];
	uint instance_id = instance_data.x;
	uint unique_splat_index = instance_data.y;

	const Splat splat = splat_buffer[unique_splat_index];
	mat4 model_matrix = instance_model_matrices[instance_id];

	// --- VISIBILITY ---
	float is_visible = model_matrix[0][3];
	if (is_visible < 0.5) return;
	model_matrix[0][3] = 0.0;

	// --- FRUSTUM CULLING (combined for stereo) ---
	// In stereo mode the IPD offset can place a splat inside one eye's
	// frustum while it sits just outside the other's.  Expanding the
	// clip-space margin from 1.2 to 1.5 covers typical VR baselines
	// (IPD ~63 mm) down to near-plane distances without a second
	// frustum test, keeping the single-dispatch design intact.
	mat3 object_linear = mat3(model_matrix);
	mat3 world_covariance = object_linear * DECODE_COVARIANCE(splat.covariance) * transpose(object_linear);
	vec4 world_pos = model_matrix * vec4(splat.position, 1.0);
	vec4 view_pos = view_matrix * world_pos;
	vec4 clip_pos = projection_matrix * view_pos;
	float frustum_margin = view_count >= 2 ? 1.5 : 1.2;
	vec2 view_bounds = clip_pos.ww * frustum_margin;
	if (any(lessThan(clip_pos.xyz, vec3(-view_bounds, 0.0))) || any(greaterThan(clip_pos.xyz, vec3(view_bounds, clip_pos.w)))) {
		return;
	}

	// --- GAUSSIAN PROJECTION (primary/left eye) ---
	float splat_time = time - splat.time;
	float time_factor = ease_out_cubic(clamp(splat_time, 0, 1));
	float time_factor_late = ease_out_cubic(clamp(splat_time - 0.35, 0, 1));

	float splat_opacity = splat.opacity * time_factor_late*time_factor_late;
	float splat_scale = mix(2.0, 1.0, time_factor_late);

	const vec3 covariance = project_covariance(world_covariance, splat_scale, view_pos.xyz, dims, view_matrix, projection_matrix);
	float det = covariance.x*covariance.z - covariance.y*covariance.y;
	if (det == 0.0) return;

	float mid = 0.5 * (covariance.x + covariance.z);
	vec2 eigenvalues = mid + vec2(1, -1)*sqrt(max(0.1, mid*mid - det));
	if (any(lessThan(eigenvalues, vec2(0)))) return;

	vec3 ndc_pos = clip_pos.xyz / clip_pos.w;
	vec2 image_pos = ((ndc_pos.xy + 1.0)*0.5 - vec2(1,0.75)*(1.0 - time_factor)) * (dims - 1);

	// We bias the radius (w/ base=2.5x standard deviation) such that low opacity splats cover
	// fewer screen tiles. This has the effect of making the image *slightly* brighter while
	// minimizing perceptible tile artifacts.
	float radius = pow(splat_opacity, 0.2) * 2.5*sqrt(max(eigenvalues.x, eigenvalues.y));
	if (radius < min_radius) return;

	// --- RIGHT EYE PROJECTION (stereo only) ---
	// Compute right eye clip position + image_pos early so that the tile rect
	// can be expanded to cover both eyes.  The 2D covariance difference between
	// eyes is negligible for typical IPD, so it is shared from the left eye.
	float clip_w_right = clip_pos.w; // default to left eye for mono
	vec2 image_pos_right = image_pos;
	vec4 view_pos_right = view_pos;
	if (view_count >= 2) {
		view_pos_right = view_matrix_right * world_pos;
		vec4 clip_pos_right = projection_matrix_right * view_pos_right;
		clip_w_right = clip_pos_right.w;
		vec3 ndc_pos_right = clip_pos_right.xyz / clip_pos_right.w;
		image_pos_right = ((ndc_pos_right.xy + 1.0)*0.5 - vec2(1,0.75)*(1.0 - time_factor)) * (dims - 1);
	}

	// Tile rect: union of both eyes' rects in stereo so that every splat
	// appears in the correct tiles for both views.
	uvec4 rect_bounds = get_rect(image_pos, radius, grid_size);
	if (view_count >= 2) {
		uvec4 rect_right = get_rect(image_pos_right, radius, grid_size);
		rect_bounds = uvec4(min(rect_bounds.xy, rect_right.xy), max(rect_bounds.zw, rect_right.zw));
	}
	uint num_tiles_touched = (rect_bounds.z - rect_bounds.x)*(rect_bounds.w - rect_bounds.y);

	if (num_tiles_touched == 0 /*|| num_tiles_touched > grid_size.x*grid_size.y/3*/) return;

	uint sort_buffer_offset = 0u;
	if (!reserve_sort_slots(num_tiles_touched, sort_buffer_offset)) {
		return;
	}
	vec3 view_dir = normalize(world_pos.xyz - camera_pos);

	RasterizeData data;
	data.image_pos = image_pos;
	data.conic = vec3(covariance.z, -covariance.y, covariance.x) / det; // Inverse 2D covariance
	data.color = vec4(get_color(view_dir, splat.sh_coefficients, sh_degree), splat_opacity);
	data.pos_xy = world_pos.xy;
	data.pos_z = world_pos.z;
	data.depth_data = vec4(-view_pos.z, 0.0, 0.0, 0.0);
	culled_buffer[id * view_count] = data;

	if (view_count >= 2) {
		RasterizeData data_right = data;
		data_right.image_pos = image_pos_right;
		data_right.depth_data = vec4(-view_pos_right.z, 0.0, 0.0, 0.0);
		culled_buffer[id * view_count + 1] = data_right;
	}

	// --- GAUSSIAN DUPLICATION ---
	// Use clip-space w as a monotonic distance proxy so ordering stays front-to-back
	// even when the renderer uses reverse-z projection.
	// In stereo mode, average left and right eye depths for a center-eye
	// sort key that is equally fair to both views.  Inspired by Nebula
	// (arxiv:2512.20495) which uses a virtual camera between both eyes
	// for shared sorting.
	float view_depth = max(0.0, (clip_pos.w + clip_w_right) * 0.5);
	float depth01 = view_depth / (1.0 + view_depth);
	uint depth = uint(depth01 * 65535.0) & 0xFFFF;
	for (uint y = rect_bounds.y; y < rect_bounds.w; ++y)
	for (uint x = rect_bounds.x; x < rect_bounds.z; ++x) {
		uint tile_id = y*grid_size.x + x;
		uint key = (tile_id << 16) | depth;
		sort_keys[sort_buffer_offset] = key;
		sort_values[sort_buffer_offset] = id;
		sort_buffer_offset++;
	}
}
