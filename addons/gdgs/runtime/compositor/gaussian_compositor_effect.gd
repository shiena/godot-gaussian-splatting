@tool
class_name GaussianCompositorEffect
extends CompositorEffect

const WORKGROUP_SIZE := 16
const MANAGER_SCRIPT := preload("res://addons/gdgs/runtime/render/gaussian_render_manager.gd")
const DIRECT_TEXTURE_SHADER := preload("res://addons/gdgs/runtime/debug/shaders/direct_texture_overlay.gdshader")
const DIRECT_TEXTURE_OVERLAY_NAME := "_GdgsDirectTextureOverlay"
const DEFAULT_TEXTURE_USAGE_BITS := 0x18B

enum DisplayMode {
	COMPOSITOR,
	DIRECT_TEXTURE
}

enum DebugView {
	COMPOSITE,
	GS_ALPHA,
	GS_COLOR,
	GS_DEPTH,
	SCENE_DEPTH,
	DEPTH_REJECT_MASK,
	EYE_TAG,            # Solid colour per eye (left=red, right=green) for multiview FB verification
	GS_DEPTH_PER_EYE,   # GS depth tinted by eye (left=red ramp, right=green ramp)
}

enum CompositeMethod {
	AUTO,
	COMPUTE,
	RASTER
}

@export_range(0.0, 1.0, 0.001) var alpha_cutoff := 0.01
@export_range(0.0, 1.0, 0.001) var depth_bias := 0.05
@export_range(0.0, 1.0, 0.001) var depth_test_min_alpha := 0.05
@export_range(0.0, 1.0, 0.001) var depth_capture_alpha = 0.5
## Spherical harmonics degree (0-3). Lower values reduce ALU cost at the expense of view-dependent color detail.
@export_range(0, 3) var sh_degree: int = 3
## Minimum projected screen-space radius in pixels. Splats smaller than this are culled.
@export_range(0.0, 16.0, 0.5) var min_radius: float = 0.0
## Resolution scale for the gaussian splatting render pass. Lower values improve performance at the cost of sharpness.
@export_range(0.25, 1.0, 0.05) var render_scale: float = 1.0
## Enables verbose render-thread diagnostics for Quest/multiview debugging.
@export var debug_logging := false
@export_enum("Compositor", "Direct Texture") var display_mode: int:
	set(value):
		_display_mode = clampi(value, DisplayMode.COMPOSITOR, DisplayMode.DIRECT_TEXTURE)
		if _display_mode != DisplayMode.DIRECT_TEXTURE:
			_queue_direct_texture_overlay_state(false, RID())
	get:
		return _display_mode
@export_enum("Composite", "GS Alpha", "GS Color", "GS Depth", "Scene Depth", "Depth Reject Mask", "Eye Tag", "GS Depth Per Eye") var debug_view: int = DebugView.COMPOSITE
## Composite method. Auto selects Compute on Forward+ and Raster on Mobile.
@export_enum("Auto", "Compute", "Raster") var composite_method: int = CompositeMethod.AUTO

var rd: RenderingDevice
# Compute composite (Forward+)
var shader: RID
var pipeline: RID
# Raster composite (Mobile-compatible)
var raster_shader: RID
var raster_pipeline: RID
# Shared
var depth_sampler: RID  # nearest — for depth textures
var linear_sampler: RID # bilinear — for GS colour upscaling
var fallback_depth_texture: RID

var _display_mode := DisplayMode.COMPOSITOR
var _direct_texture_resource: Texture2DRD
var _overlay_mutex := Mutex.new()
var _overlay_sync_queued := false
var _overlay_pending_visible := false
var _overlay_pending_texture_rid := RID()
const _DEBUG_TAG := "GDGS"
var _resource_log_done: Dictionary = {}  # key: size Vector2i, value: true once logged

func _init() -> void:
	effect_callback_type = EFFECT_CALLBACK_TYPE_PRE_TRANSPARENT
	access_resolved_depth = true
	RenderingServer.call_on_render_thread(_initialize_shaders)

func _notification(what: int) -> void:
	if what != NOTIFICATION_PREDELETE:
		return

	_overlay_mutex.lock()
	_overlay_sync_queued = false
	_overlay_pending_visible = false
	_overlay_pending_texture_rid = RID()
	_overlay_mutex.unlock()

	if _direct_texture_resource != null:
		_direct_texture_resource.texture_rd_rid = RID()
		_direct_texture_resource = null

	var main_loop := Engine.get_main_loop()
	if main_loop is SceneTree:
		var tree: SceneTree = main_loop
		if tree.root != null:
			var overlay := tree.root.get_node_or_null(DIRECT_TEXTURE_OVERLAY_NAME) as MeshInstance3D
			if overlay != null:
				overlay.queue_free()

	if rd != null:
		if fallback_depth_texture.is_valid():
			rd.free_rid(fallback_depth_texture)
		if pipeline.is_valid():
			rd.free_rid(pipeline)
		if shader.is_valid():
			rd.free_rid(shader)
		if raster_pipeline.is_valid():
			rd.free_rid(raster_pipeline)
		if raster_shader.is_valid():
			rd.free_rid(raster_shader)
		if depth_sampler.is_valid():
			rd.free_rid(depth_sampler)
		if linear_sampler.is_valid():
			rd.free_rid(linear_sampler)
	fallback_depth_texture = RID()
	linear_sampler = RID()
	pipeline = RID()
	shader = RID()
	raster_pipeline = RID()
	raster_shader = RID()
	depth_sampler = RID()

func _render_callback(_effect_callback_type: int, render_data: RenderData) -> void:
	var is_direct_texture_mode := display_mode == DisplayMode.DIRECT_TEXTURE
	var resolved := _resolve_composite_method()
	# At least one composite path must be available.
	var has_valid_pipeline := rd != null and (
		raster_shader.is_valid() or (shader.is_valid() and pipeline.is_valid()))
	if not is_direct_texture_mode and not has_valid_pipeline:
		_queue_direct_texture_overlay_state(false, RID())
		return

	var scene_buffers: RenderSceneBuffersRD = render_data.get_render_scene_buffers()
	var scene_data: RenderSceneDataRD = render_data.get_render_scene_data()
	if scene_buffers == null or scene_data == null:
		_queue_direct_texture_overlay_state(false, RID())
		return

	var manager = MANAGER_SCRIPT.get_instance()
	if manager == null:
		_queue_direct_texture_overlay_state(false, RID())
		return

	var size: Vector2i = scene_buffers.get_internal_size()
	if size.x <= 0 or size.y <= 0:
		_queue_direct_texture_overlay_state(false, RID())
		return

	var view_count: int = scene_buffers.get_view_count()
	var should_log := debug_logging and Engine.get_frames_drawn() % 60 == 0

	if should_log:
		var raw_cam_transform: Transform3D = scene_data.get_cam_transform()
		var raw_cam_projection: Projection = scene_data.get_cam_projection()
		print("[%s] === frame=%d view_count=%d size=%s ===" % [_DEBUG_TAG, Engine.get_frames_drawn(), view_count, str(size)])
		print("[%s] cam_transform.origin=%s" % [_DEBUG_TAG, str(raw_cam_transform.origin)])
		print("[%s] cam_transform.basis.x=%s" % [_DEBUG_TAG, str(raw_cam_transform.basis.x)])
		print("[%s] cam_transform.basis.y=%s" % [_DEBUG_TAG, str(raw_cam_transform.basis.y)])
		print("[%s] cam_transform.basis.z=%s" % [_DEBUG_TAG, str(raw_cam_transform.basis.z)])
		print("[%s] cam_projection.x=%s" % [_DEBUG_TAG, str(raw_cam_projection.x)])
		print("[%s] cam_projection.y=%s" % [_DEBUG_TAG, str(raw_cam_projection.y)])
		print("[%s] cam_projection.z=%s" % [_DEBUG_TAG, str(raw_cam_projection.z)])
		print("[%s] cam_projection.w=%s" % [_DEBUG_TAG, str(raw_cam_projection.w)])
		for v in view_count:
			var vp: Projection = scene_data.get_view_projection(v)
			var eo: Vector3 = scene_data.get_view_eye_offset(v)
			print("[%s] view[%d] eye_offset=%s" % [_DEBUG_TAG, v, str(eo)])
			print("[%s] view[%d] view_projection.x=%s" % [_DEBUG_TAG, v, str(vp.x)])
			print("[%s] view[%d] view_projection.y=%s" % [_DEBUG_TAG, v, str(vp.y)])
			print("[%s] view[%d] view_projection.z=%s" % [_DEBUG_TAG, v, str(vp.z)])
			print("[%s] view[%d] view_projection.w=%s" % [_DEBUG_TAG, v, str(vp.w)])

	# Collect camera data for all views
	var camera_data_array: Array = []
	for view in view_count:
		var camera_data := _get_camera_data(scene_data, view)
		if camera_data.is_empty():
			_queue_direct_texture_overlay_state(false, RID())
			return
		camera_data_array.append(camera_data)

	if should_log:
		for v in camera_data_array.size():
			var cd: Dictionary = camera_data_array[v]
			var t: Transform3D = cd["transform"]
			var p: Projection = cd["projection"]
			print("[%s] camera_data[%d] transform.origin=%s" % [_DEBUG_TAG, v, str(t.origin)])
			print("[%s] camera_data[%d] transform.basis.z=%s" % [_DEBUG_TAG, v, str(t.basis.z)])
			print("[%s] camera_data[%d] projection.x=%s" % [_DEBUG_TAG, v, str(p.x)])
			print("[%s] camera_data[%d] projection.y=%s" % [_DEBUG_TAG, v, str(p.y)])
			print("[%s] camera_data[%d] projection.z=%s" % [_DEBUG_TAG, v, str(p.z)])
			print("[%s] camera_data[%d] projection.w=%s" % [_DEBUG_TAG, v, str(p.w)])
			print("[%s] camera_data[%d] world_position=%s" % [_DEBUG_TAG, v, str(cd["world_position"])])

	# Render all views at once (projection + sort once, render per eye)
	var gs_scale := clampf(render_scale, 0.25, 1.0)
	var gs_size := Vector2i(maxi(1, int(size.x * gs_scale)), maxi(1, int(size.y * gs_scale)))
	var gsplat_result: Dictionary = manager.render_for_compositor_multiview(
		gs_size, camera_data_array, _get_depth_capture_alpha(), sh_degree, min_radius, debug_logging
	)
	var gsplat_views: Array = gsplat_result.get("views", [])
	if gsplat_views.size() != view_count:
		_queue_direct_texture_overlay_state(false, RID())
		return

	# Direct texture mode: show first view only
	if is_direct_texture_mode:
		var gsplat_texture: RID = gsplat_views[0].get("color_alpha_texture", RID())
		if gsplat_texture.is_valid():
			_queue_direct_texture_overlay_state(true, gsplat_texture)
		else:
			_queue_direct_texture_overlay_state(false, RID())
		return

	# Composite each view.
	# In multiview (VR), always use the raster path — compute imageStore
	# to texture-array layer views is unreliable on some drivers/backends.
	var use_raster := resolved == CompositeMethod.RASTER or view_count >= 2
	if use_raster and raster_shader.is_valid():
		_composite_raster(view_count, gsplat_views, scene_buffers, camera_data_array, size)
	elif not use_raster and shader.is_valid() and pipeline.is_valid():
		_composite_compute(view_count, gsplat_views, scene_buffers, camera_data_array, size)
	else:
		# Fallback: try whichever path has valid resources
		if raster_shader.is_valid():
			_composite_raster(view_count, gsplat_views, scene_buffers, camera_data_array, size)
		elif shader.is_valid() and pipeline.is_valid():
			_composite_compute(view_count, gsplat_views, scene_buffers, camera_data_array, size)

	if not is_direct_texture_mode:
		_queue_direct_texture_overlay_state(false, RID())

# ---------------------------------------------------------------------------
# Composite: Compute path (Forward+)
# ---------------------------------------------------------------------------
func _composite_compute(view_count: int, gsplat_views: Array, scene_buffers: RenderSceneBuffersRD, camera_data_array: Array, size: Vector2i) -> void:
	if debug_logging:
		_log_resources_once("compute", view_count, scene_buffers, gsplat_views, size)
	var x_groups: int = int(ceili(size.x / float(WORKGROUP_SIZE)))
	var y_groups: int = int(ceili(size.y / float(WORKGROUP_SIZE)))

	for view in view_count:
		var gsplat_texture: RID = gsplat_views[view].get("color_alpha_texture", RID())
		var gsplat_depth_texture: RID = gsplat_views[view].get("depth_texture", RID())
		if not gsplat_texture.is_valid() or not gsplat_depth_texture.is_valid():
			continue

		var scene_tex: RID = scene_buffers.get_color_layer(view)
		if not scene_tex.is_valid() or not depth_sampler.is_valid():
			continue

		var use_scene_depth := _debug_view_needs_scene_depth(debug_view)
		var scene_depth_tex: RID = _get_scene_depth_texture(scene_buffers, view)
		# If scene depth is unavailable for this view (common for the right eye
		# in mobile multiview), disable depth testing instead of skipping the
		# entire view — otherwise the composite is never drawn for that eye.
		if not scene_depth_tex.is_valid():
			use_scene_depth = false
			scene_depth_tex = fallback_depth_texture
		if not scene_depth_tex.is_valid():
			continue

		var push_constants := PackedFloat32Array([
			size.x,
			size.y,
			alpha_cutoff,
			depth_bias,
			depth_test_min_alpha,
			float(debug_view),
			1.0 if use_scene_depth else 0.0,
			float(view)
		] + _projection_to_column_major_floats(camera_data_array[view]["projection"].inverse()))

		var scene_uniform := RDUniform.new()
		scene_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
		scene_uniform.binding = 0
		scene_uniform.add_id(scene_tex)

		var gsplat_uniform := RDUniform.new()
		gsplat_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
		gsplat_uniform.binding = 1
		gsplat_uniform.add_id(gsplat_texture)

		var gsplat_depth_uniform := RDUniform.new()
		gsplat_depth_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
		gsplat_depth_uniform.binding = 2
		gsplat_depth_uniform.add_id(gsplat_depth_texture)

		var scene_depth_uniform := RDUniform.new()
		scene_depth_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		scene_depth_uniform.binding = 3
		scene_depth_uniform.add_id(depth_sampler)
		scene_depth_uniform.add_id(scene_depth_tex)

		var uniform_set: RID = UniformSetCacheRD.get_cache(shader, 0, [
			scene_uniform,
			gsplat_uniform,
			gsplat_depth_uniform,
			scene_depth_uniform
		])
		var compute_list: int = rd.compute_list_begin()
		rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
		rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
		rd.compute_list_set_push_constant(
			compute_list,
			push_constants.to_byte_array(),
			push_constants.size() * 4
		)
		rd.compute_list_dispatch(compute_list, x_groups, y_groups, 1)
		rd.compute_list_end()

# ---------------------------------------------------------------------------
# Composite: Raster path (Mobile-compatible)
# ---------------------------------------------------------------------------
func _composite_raster(view_count: int, gsplat_views: Array, scene_buffers: RenderSceneBuffersRD, camera_data_array: Array, size: Vector2i) -> void:
	var log_fb := debug_logging and not _resource_log_done.has(size)
	if debug_logging:
		_log_resources_once("raster", view_count, scene_buffers, gsplat_views, size)
	for view in view_count:
		var gsplat_texture: RID = gsplat_views[view].get("color_alpha_texture", RID())
		var gsplat_depth_texture: RID = gsplat_views[view].get("depth_texture", RID())
		if not gsplat_texture.is_valid() or not gsplat_depth_texture.is_valid():
			continue

		var scene_tex: RID = scene_buffers.get_color_layer(view)
		if not scene_tex.is_valid() or not depth_sampler.is_valid():
			continue

		var use_scene_depth := _debug_view_needs_scene_depth(debug_view)
		var scene_depth_tex: RID = _get_scene_depth_texture(scene_buffers, view)
		# If scene depth is unavailable for this view (common for the right eye
		# in mobile multiview), disable depth testing instead of skipping the
		# entire view — otherwise the composite is never drawn for that eye.
		if not scene_depth_tex.is_valid():
			use_scene_depth = false
			scene_depth_tex = fallback_depth_texture
		if not scene_depth_tex.is_valid():
			continue

		_ensure_raster_pipeline(scene_tex)
		if not raster_pipeline.is_valid():
			continue

		var push_constants := PackedFloat32Array([
			size.x,
			size.y,
			alpha_cutoff,
			depth_bias,
			depth_test_min_alpha,
			float(debug_view),
			1.0 if use_scene_depth else 0.0,
			float(view)
		] + _projection_to_column_major_floats(camera_data_array[view]["projection"].inverse()))

		var gsplat_uniform := RDUniform.new()
		gsplat_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		gsplat_uniform.binding = 0
		gsplat_uniform.add_id(linear_sampler)
		gsplat_uniform.add_id(gsplat_texture)

		var gsplat_depth_uniform := RDUniform.new()
		gsplat_depth_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		gsplat_depth_uniform.binding = 1
		gsplat_depth_uniform.add_id(depth_sampler)
		gsplat_depth_uniform.add_id(gsplat_depth_texture)

		var scene_depth_uniform := RDUniform.new()
		scene_depth_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		scene_depth_uniform.binding = 2
		scene_depth_uniform.add_id(depth_sampler)
		scene_depth_uniform.add_id(scene_depth_tex)

		var uniform_set: RID = UniformSetCacheRD.get_cache(raster_shader, 0, [
			gsplat_uniform,
			gsplat_depth_uniform,
			scene_depth_uniform
		])

		var fb: RID = rd.framebuffer_create([scene_tex])
		if log_fb:
			print("[%s] raster fb[view=%d] rid=%s scene_tex=%s fb_format=%d" % [
				_DEBUG_TAG, view, str(fb), str(scene_tex), rd.framebuffer_get_format(fb)
			])
		var draw_list: int = rd.draw_list_begin(fb)
		rd.draw_list_bind_render_pipeline(draw_list, raster_pipeline)
		rd.draw_list_bind_uniform_set(draw_list, uniform_set, 0)
		rd.draw_list_set_push_constant(
			draw_list,
			push_constants.to_byte_array(),
			push_constants.size() * 4
		)
		rd.draw_list_draw(draw_list, false, 1, 3)
		rd.draw_list_end()
		rd.free_rid(fb)

# ---------------------------------------------------------------------------
# Raster pipeline (lazy init — needs framebuffer format from a scene texture)
# ---------------------------------------------------------------------------
func _ensure_raster_pipeline(scene_tex: RID) -> void:
	if raster_pipeline.is_valid():
		return
	if not raster_shader.is_valid():
		return

	var fb: RID = rd.framebuffer_create([scene_tex])
	var fb_format: int = rd.framebuffer_get_format(fb)
	rd.free_rid(fb)
	if debug_logging:
		print("[%s] _ensure_raster_pipeline: pipeline created with fb_format=%d (scene_tex=%s)" % [_DEBUG_TAG, fb_format, str(scene_tex)])
		var scene_fmt: RDTextureFormat = rd.texture_get_format(scene_tex)
		if scene_fmt != null:
			print("[%s]   scene_tex format=%d w=%d h=%d layers=%d usage=0x%X" % [
				_DEBUG_TAG, scene_fmt.format, scene_fmt.width, scene_fmt.height,
				scene_fmt.array_layers, scene_fmt.usage_bits
			])

	var blend := RDPipelineColorBlendStateAttachment.new()
	blend.enable_blend = true
	blend.src_color_blend_factor = RenderingDevice.BLEND_FACTOR_ONE
	blend.dst_color_blend_factor = RenderingDevice.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA
	blend.color_blend_op = RenderingDevice.BLEND_OP_ADD
	blend.src_alpha_blend_factor = RenderingDevice.BLEND_FACTOR_ONE
	blend.dst_alpha_blend_factor = RenderingDevice.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA
	blend.alpha_blend_op = RenderingDevice.BLEND_OP_ADD

	var color_blend := RDPipelineColorBlendState.new()
	color_blend.attachments.push_back(blend)

	raster_pipeline = rd.render_pipeline_create(
		raster_shader,
		fb_format,
		-1, # no vertex format — vertices generated from gl_VertexIndex
		RenderingDevice.RENDER_PRIMITIVE_TRIANGLES,
		RDPipelineRasterizationState.new(),
		RDPipelineMultisampleState.new(),
		RDPipelineDepthStencilState.new(),
		color_blend
	)

# ---------------------------------------------------------------------------
# Composite method resolution
# ---------------------------------------------------------------------------
func _resolve_composite_method() -> int:
	if composite_method != CompositeMethod.AUTO:
		return composite_method
	# Use the runtime feature tag instead of ProjectSettings — the tag
	# reflects the renderer actually active on this device/platform,
	# which can differ from the project setting on Android.
	if OS.has_feature("forward_plus"):
		return CompositeMethod.COMPUTE
	return CompositeMethod.RASTER

# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------
func _get_camera_data(scene_data: RenderSceneDataRD, view: int) -> Dictionary:
	if scene_data == null:
		return {}
	if not scene_data.has_method("get_cam_transform") or not scene_data.has_method("get_view_projection"):
		return {}
	var camera_transform: Transform3D = scene_data.get_cam_transform()
	var camera_projection: Projection = scene_data.get_view_projection(view)
	var world_position: Vector3 = camera_transform.origin

	# Build per-eye camera data using the raw per-eye projection.
	#
	# get_view_projection(v) = raw_proj[v] * Projection(view_offset[v].inverse())
	# where view_offset[v] ≈ translate(eye_offset).
	#
	# The baked-in eye_offset causes projection[3][3] != 0, which introduces
	# precision issues on Adreno GPUs (Quest native).
	#
	# Fix: undo the eye_offset from the projection, and apply it to the
	# view matrix instead (per-eye view + clean projection).
	if scene_data.has_method("get_view_eye_offset"):
		var eye_offset: Vector3 = scene_data.get_view_eye_offset(view)
		if eye_offset != Vector3.ZERO and scene_data.has_method("get_view_projection"):
			# Extract raw per-eye projection by undoing the eye_offset:
			#   raw_proj[v] = get_view_projection(v) * Projection(view_offset)
			var view_offset := Transform3D(Basis.IDENTITY, eye_offset)
			camera_projection = scene_data.get_view_projection(view) * Projection(view_offset)
		elif scene_data.has_method("get_view_projection"):
			camera_projection = scene_data.get_view_projection(view)
		# Apply eye offset to transform → per-eye view matrix
		camera_transform.origin += camera_transform.basis * eye_offset
		world_position = camera_transform.origin


	return {
		"transform": camera_transform,
		"projection": camera_projection,
		"world_position": world_position
	}

func _log_texture_info(tag: String, rid: RID) -> void:
	if rd == null:
		return
	if not rid.is_valid():
		print("[%s] %s rid=INVALID" % [_DEBUG_TAG, tag])
		return
	var fmt: RDTextureFormat = rd.texture_get_format(rid)
	if fmt == null:
		print("[%s] %s rid=%s (texture_get_format returned null)" % [_DEBUG_TAG, tag, str(rid)])
		return
	print("[%s] %s rid=%s fmt=%d w=%d h=%d depth=%d layers=%d mips=%d type=%d samples=%d usage=0x%X" % [
		_DEBUG_TAG, tag, str(rid),
		fmt.format, fmt.width, fmt.height, fmt.depth, fmt.array_layers,
		fmt.mipmaps, fmt.texture_type, fmt.samples, fmt.usage_bits
	])

func _log_resources_once(path_tag: String, view_count: int, scene_buffers: RenderSceneBuffersRD, gsplat_views: Array, size: Vector2i) -> void:
	if _resource_log_done.has(size) or scene_buffers == null:
		return
	_resource_log_done[size] = true
	print("[%s] === Resource snapshot (%s) size=%s view_count=%d ===" % [_DEBUG_TAG, path_tag, str(size), view_count])
	for view in view_count:
		var color_layer: RID = scene_buffers.get_color_layer(view) if scene_buffers.has_method("get_color_layer") else RID()
		_log_texture_info("color_layer[%d]" % view, color_layer)
		if scene_buffers.has_method("has_texture") and scene_buffers.has_method("get_texture_slice") and scene_buffers.has_texture("render_buffers", "color"):
			var color_slice: RID = scene_buffers.get_texture_slice("render_buffers", "color", view, 0, 1, 1)
			_log_texture_info("color_slice[%d]" % view, color_slice)
		else:
			print("[%s] color_slice[%d] not available (no render_buffers/color texture)" % [_DEBUG_TAG, view])
		var depth_tex: RID = _get_scene_depth_texture(scene_buffers, view)
		_log_texture_info("scene_depth[%d]" % view, depth_tex)
		if view < gsplat_views.size():
			var gv: Dictionary = gsplat_views[view]
			_log_texture_info("gs_color[%d]" % view, gv.get("color_alpha_texture", RID()))
			_log_texture_info("gs_depth[%d]" % view, gv.get("depth_texture", RID()))
	if scene_buffers.has_method("get_view_count"):
		print("[%s] scene_buffers.get_view_count()=%d" % [_DEBUG_TAG, scene_buffers.get_view_count()])
	if scene_buffers.has_method("get_internal_size"):
		print("[%s] scene_buffers.get_internal_size()=%s" % [_DEBUG_TAG, str(scene_buffers.get_internal_size())])
	print("[%s] === Resource snapshot end ===" % _DEBUG_TAG)

func _get_scene_depth_texture(scene_buffers: RenderSceneBuffersRD, view: int) -> RID:
	if scene_buffers == null:
		return RID()

	if scene_buffers.has_method("has_texture") and scene_buffers.has_method("get_texture_slice") and scene_buffers.has_texture("render_buffers", "depth"):
		var depth_slice: RID = scene_buffers.get_texture_slice("render_buffers", "depth", view, 0, 1, 1)
		if depth_slice.is_valid():
			return depth_slice

	if scene_buffers.has_method("get_depth_layer"):
		return scene_buffers.get_depth_layer(view)

	return RID()

func _projection_to_column_major_floats(matrix: Projection) -> Array:
	return [
		matrix.x[0], matrix.x[1], matrix.x[2], matrix.x[3],
		matrix.y[0], matrix.y[1], matrix.y[2], matrix.y[3],
		matrix.z[0], matrix.z[1], matrix.z[2], matrix.z[3],
		matrix.w[0], matrix.w[1], matrix.w[2], matrix.w[3]
	]

func _get_depth_capture_alpha() -> float:
	if depth_capture_alpha == null:
		return 0.5
	return clampf(float(depth_capture_alpha), 0.0, 1.0)

func _debug_view_needs_scene_depth(view: int) -> bool:
	return view == DebugView.COMPOSITE or view == DebugView.SCENE_DEPTH or view == DebugView.DEPTH_REJECT_MASK

# ---------------------------------------------------------------------------
# Shader / pipeline initialisation (called on render thread)
# ---------------------------------------------------------------------------
func _initialize_shaders() -> void:
	rd = RenderingServer.get_rendering_device()
	if not rd:
		return

	# Load both composite shaders so that the method can be switched at
	# runtime (e.g. AUTO falls back to raster in VR multiview where
	# compute imageStore to texture-array layer views is unreliable).
	var compute_glsl: RDShaderFile = load("res://addons/gdgs/runtime/compositor/shaders/gaussian_composite.glsl")
	if compute_glsl != null:
		shader = rd.shader_create_from_spirv(compute_glsl.get_spirv())
		if shader.is_valid():
			pipeline = rd.compute_pipeline_create(shader)

	var raster_glsl: RDShaderFile = load("res://addons/gdgs/runtime/compositor/shaders/gaussian_composite_raster.glsl")
	if raster_glsl != null:
		raster_shader = rd.shader_create_from_spirv(raster_glsl.get_spirv())

	# Shared resources
	var nearest_state := RDSamplerState.new()
	depth_sampler = rd.sampler_create(nearest_state)
	var linear_state := RDSamplerState.new()
	linear_state.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	linear_state.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	linear_sampler = rd.sampler_create(linear_state)
	fallback_depth_texture = _create_fallback_depth_texture()

func _create_fallback_depth_texture() -> RID:
	if rd == null:
		return RID()

	var texture_format := RDTextureFormat.new()
	texture_format.format = RenderingDevice.DATA_FORMAT_R32_SFLOAT
	texture_format.width = 1
	texture_format.height = 1
	texture_format.usage_bits = DEFAULT_TEXTURE_USAGE_BITS

	return rd.texture_create(
		texture_format,
		RDTextureView.new(),
		[PackedFloat32Array([1.0]).to_byte_array()]
	)

# ---------------------------------------------------------------------------
# Direct texture overlay (debug)
# ---------------------------------------------------------------------------
func _queue_direct_texture_overlay_state(visible: bool, texture_rid: RID) -> void:
	var next_visible := visible and texture_rid.is_valid()
	var next_texture_rid := texture_rid if next_visible else RID()

	_overlay_mutex.lock()
	var state_changed := _overlay_pending_visible != next_visible or _overlay_pending_texture_rid != next_texture_rid
	_overlay_pending_visible = next_visible
	_overlay_pending_texture_rid = next_texture_rid
	var should_queue := state_changed and not _overlay_sync_queued
	if should_queue:
		_overlay_sync_queued = true
	_overlay_mutex.unlock()

	if should_queue:
		call_deferred("_sync_direct_texture_overlay")

func _sync_direct_texture_overlay() -> void:
	var pending_visible := false
	var pending_texture_rid := RID()

	_overlay_mutex.lock()
	pending_visible = _overlay_pending_visible
	pending_texture_rid = _overlay_pending_texture_rid
	_overlay_sync_queued = false
	_overlay_mutex.unlock()

	var overlay := _ensure_direct_texture_overlay() if pending_visible else _get_direct_texture_overlay()
	if overlay == null:
		return

	var texture := _ensure_direct_texture_resource()
	if texture == null:
		return

	texture.texture_rd_rid = pending_texture_rid if pending_visible else RID()
	overlay.visible = pending_visible and pending_texture_rid.is_valid()

func _ensure_direct_texture_overlay() -> MeshInstance3D:
	var overlay := _get_direct_texture_overlay()
	if overlay != null:
		_configure_direct_texture_overlay(overlay)
		return overlay

	var tree := _get_scene_tree()
	if tree == null or tree.root == null:
		return null

	overlay = MeshInstance3D.new()
	overlay.name = DIRECT_TEXTURE_OVERLAY_NAME
	overlay.visible = false
	tree.root.add_child(overlay)
	_configure_direct_texture_overlay(overlay)
	return overlay

func _configure_direct_texture_overlay(overlay: MeshInstance3D) -> void:
	if overlay == null:
		return

	var mesh := overlay.mesh as QuadMesh
	if mesh == null:
		mesh = QuadMesh.new()
	overlay.mesh = mesh
	mesh.flip_faces = true
	mesh.size = Vector2(2.0, 2.0)

	overlay.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	overlay.extra_cull_margin = 16384.0
	overlay.gi_mode = GeometryInstance3D.GI_MODE_DISABLED

	var material := overlay.get_active_material(0) as ShaderMaterial
	if material == null:
		material = ShaderMaterial.new()
	material.shader = DIRECT_TEXTURE_SHADER
	material.render_priority = 127
	material.set_shader_parameter("render_texture", _ensure_direct_texture_resource())
	overlay.set_surface_override_material(0, material)

func _ensure_direct_texture_resource() -> Texture2DRD:
	if _direct_texture_resource == null:
		_direct_texture_resource = Texture2DRD.new()
	return _direct_texture_resource

func _get_direct_texture_overlay() -> MeshInstance3D:
	var tree := _get_scene_tree()
	if tree == null or tree.root == null:
		return null
	return tree.root.get_node_or_null(DIRECT_TEXTURE_OVERLAY_NAME) as MeshInstance3D

func _free_direct_texture_overlay() -> void:
	if _direct_texture_resource != null:
		_direct_texture_resource.texture_rd_rid = RID()
		_direct_texture_resource = null

	var overlay := _get_direct_texture_overlay()
	if overlay != null:
		overlay.queue_free()

func _get_scene_tree() -> SceneTree:
	var main_loop := Engine.get_main_loop()
	if main_loop is SceneTree:
		return main_loop
	return null
