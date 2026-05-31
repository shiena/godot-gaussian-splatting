// Raster-based gaussian composite — functionally identical to the compute
// version (gaussian_composite.glsl) but uses a fullscreen-triangle fragment
// shader with hardware blending instead of imageLoad/imageStore.
//
// This path works on renderers (e.g. Godot Mobile) whose scene-colour
// textures lack TEXTURE_USAGE_STORAGE_BIT.  The blend state is configured
// as premultiplied-alpha over (src*ONE + dst*ONE_MINUS_SRC_ALPHA) so the
// fragment shader never needs to read the scene colour buffer.
//
// Reference: BastiaanOlij/RERadialSunRays (MIT License)
// https://github.com/BastiaanOlij/RERadialSunRays

#[vertex]
#version 450

layout(location = 0) out vec2 v_uv;

void main() {
	// Fullscreen triangle: three vertices cover the entire screen.
	v_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	gl_Position = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);
}

#[fragment]
#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D gsplat_tex;
layout(set = 0, binding = 1) uniform sampler2D gsplat_depth_tex;
layout(set = 0, binding = 2) uniform sampler2D scene_depth_tex;

layout(push_constant, std430) uniform Params {
	vec2 screen_size;
	float alpha_cutoff;
	float depth_bias;
	float depth_test_min_alpha;
	float debug_view;
	float use_scene_depth;
	float view_index; // 0 = left/mono, 1 = right
	mat4 inv_projection;
} p;

const float INVALID_DEPTH = 1e19;

vec3 visualize_depth(float depth) {
	float normalized = clamp(depth / 20.0, 0.0, 1.0);
	return vec3(normalized);
}

vec3 unpremultiply_color(vec3 color, float alpha) {
	if (alpha <= 1e-5) {
		return vec3(0.0);
	}
	return color / alpha;
}

vec3 srgb_to_linear(vec3 color) {
	bvec3 cutoff = lessThanEqual(color, vec3(0.04045));
	vec3 lower = color / 12.92;
	vec3 higher = pow((color + 0.055) / 1.055, vec3(2.4));
	return mix(higher, lower, cutoff);
}

float get_scene_view_depth(ivec2 pixel, out bool has_scene_depth) {
	float raw_depth = texelFetch(scene_depth_tex, pixel, 0).r;
	if (raw_depth <= 0.0) {
		has_scene_depth = false;
		return 0.0;
	}

	vec2 uv = (vec2(pixel) + vec2(0.5)) / p.screen_size;
	vec3 ndc = vec3(uv * 2.0 - 1.0, raw_depth);
	vec4 view = p.inv_projection * vec4(ndc, 1.0);
	view.xyz /= view.w;
	has_scene_depth = true;
	return -view.z;
}

void main() {
	ivec2 pixel = ivec2(gl_FragCoord.xy);
	if (pixel.x >= int(p.screen_size.x) || pixel.y >= int(p.screen_size.y)) {
		discard;
	}

	// Use UV-based sampling to handle GS textures at different resolution than screen
	vec2 uv = (vec2(pixel) + 0.5) / p.screen_size;
	vec4 gsplat_color = texture(gsplat_tex, uv);
	float gsplat_alpha = gsplat_color.a;
	float gsplat_view_depth = texture(gsplat_depth_tex, uv).r;
	bool has_gsplat_depth = gsplat_view_depth < INVALID_DEPTH;

	bool has_scene_depth = false;
	float scene_view_depth = 0.0;
	if (p.use_scene_depth > 0.5) {
		scene_view_depth = get_scene_view_depth(pixel, has_scene_depth);
	}
	bool depth_rejected = has_scene_depth && has_gsplat_depth
		&& gsplat_alpha >= p.depth_test_min_alpha
		&& gsplat_view_depth > scene_view_depth + p.depth_bias;

	// Debug views — alpha=1.0 makes the blend replace the scene colour entirely.
	int dv = int(p.debug_view + 0.5);
	if (dv == 1) {
		frag_color = vec4(vec3(gsplat_alpha), 1.0);
		return;
	}
	if (dv == 2) {
		vec3 gsplat_straight = unpremultiply_color(gsplat_color.rgb, gsplat_alpha);
		frag_color = vec4(srgb_to_linear(gsplat_straight), 1.0);
		return;
	}
	if (dv == 3) {
		frag_color = vec4(gsplat_view_depth >= INVALID_DEPTH ? vec3(0.0) : visualize_depth(gsplat_view_depth), 1.0);
		return;
	}
	if (dv == 4) {
		frag_color = vec4(has_scene_depth ? visualize_depth(scene_view_depth) : vec3(0.0), 1.0);
		return;
	}
	if (dv == 5) {
		frag_color = depth_rejected ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 1.0, 0.0, 1.0);
		return;
	}
	if (dv == 6) {
		// EYE_TAG: left=red, right=green (mono shows red)
		frag_color = (p.view_index < 0.5) ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 1.0, 0.0, 1.0);
		return;
	}
	if (dv == 7) {
		// GS_DEPTH_PER_EYE: depth ramp tinted by eye
		float d = gsplat_view_depth >= INVALID_DEPTH ? 0.0 : clamp(gsplat_view_depth / 20.0, 0.0, 1.0);
		frag_color = (p.view_index < 0.5) ? vec4(d, 0.0, 0.0, 1.0) : vec4(0.0, d, 0.0, 1.0);
		return;
	}

	// Normal composite — discard preserves scene colour untouched
	// because the blend equation yields (0 + dst*1) = dst.
	if (gsplat_alpha <= p.alpha_cutoff) {
		discard;
	}
	if (has_scene_depth && !has_gsplat_depth) {
		discard;
	}
	if (depth_rejected) {
		discard;
	}

	vec3 gsplat_straight = unpremultiply_color(gsplat_color.rgb, gsplat_alpha);
	vec3 gsplat_linear = srgb_to_linear(gsplat_straight) * gsplat_alpha;
	// Premultiplied colour. The hardware blend state
	// (src*ONE + dst*ONE_MINUS_SRC_ALPHA) composites this onto the scene.
	frag_color = vec4(gsplat_linear, gsplat_alpha);
}
