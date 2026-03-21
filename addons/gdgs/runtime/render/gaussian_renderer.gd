@tool
extends RefCounted
class_name GaussianRenderer

const RenderingDeviceContext := preload("res://addons/gdgs/runtime/render/gaussian_rendering_device_context.gd")
const RADIX := 256
const MAX_SORT_ELEMENTS_PER_SPLAT := 10

## Single-view render (backward-compatible). Wraps multiview with view_count=1.
func render_for_compositor(
	state_cache: GaussianGpuStateCache,
	scene_registry: GaussianSceneRegistry,
	texture_size: Vector2i,
	camera_transform: Transform3D,
	camera_projection: Projection,
	camera_world_position: Vector3,
	depth_capture_alpha: float = 0.5
) -> Dictionary:
	var result := render_for_compositor_multiview(
		state_cache, scene_registry, texture_size,
		[{"transform": camera_transform, "projection": camera_projection, "world_position": camera_world_position}],
		depth_capture_alpha
	)
	var views: Array = result.get("views", [])
	return views[0] if views.size() > 0 else {}

## Multiview render. Runs projection + sort once, then renders per eye.
## camera_data_array: Array of {"transform": Transform3D, "projection": Projection, "world_position": Vector3}
## Returns {"views": [{"color_alpha_texture": RID, "depth_texture": RID}, ...]}
func render_for_compositor_multiview(
	state_cache: GaussianGpuStateCache,
	scene_registry: GaussianSceneRegistry,
	texture_size: Vector2i,
	camera_data_array: Array,
	depth_capture_alpha: float = 0.5
) -> Dictionary:
	state_cache.flush_pending_cleanup()

	if not scene_registry.has_gpu_data():
		if state_cache.has_render_states():
			state_cache.cleanup_all()
		return {}

	var view_count := camera_data_array.size()
	if view_count <= 0 or view_count > 2:
		return {}

	var point_count := scene_registry.get_point_count()
	var safe_size := Vector2i(maxi(texture_size.x, 1), maxi(texture_size.y, 1))
	var state = state_cache.get_or_create_render_state(safe_size)

	# Update view_count — triggers GPU rebuild if changed
	if state.view_count != view_count:
		state.view_count = view_count
		state.needs_gpu_rebuild = true

	# Update primary (left) eye camera
	var primary: Dictionary = camera_data_array[0]
	_update_camera(state, primary["transform"], primary["projection"], primary["world_position"])
	state.depth_capture_alpha = clampf(depth_capture_alpha, 0.0, 1.0)

	# Update right eye camera (stereo)
	if view_count >= 2:
		var right: Dictionary = camera_data_array[1]
		state.camera_view_right = Projection(right["transform"].affine_inverse())
		state.camera_projection_right = right["projection"]
		state.camera_world_position_right = right["world_position"]

	if state.context == null or state.needs_gpu_rebuild:
		state_cache.rebuild_gpu_state(state, point_count, scene_registry.get_instance_count())
	if state.context == null:
		return {}

	if state.needs_splat_upload:
		state_cache.upload_splats(state, scene_registry.get_point_data_byte(), scene_registry.get_splat_instance_ids_byte())
	if state.needs_instance_upload:
		state_cache.upload_instance_transforms(state, scene_registry.get_instance_transforms_byte())

	_rasterize_state(state, point_count)

	var views := []
	for v in range(state.view_count):
		var rt_key := "render_texture_%d" % v
		var dt_key := "depth_texture_%d" % v
		if state.descriptors.has(rt_key) and state.descriptors.has(dt_key):
			views.append({
				"color_alpha_texture": state.descriptors[rt_key].rid,
				"depth_texture": state.descriptors[dt_key].rid
			})
	return {"views": views} if views.size() == state.view_count else {}

func _rasterize_state(state, point_count: int) -> void:
	if state.context == null:
		return

	var ubo_data := RenderingDeviceContext.create_buffer_data(
		[
			state.camera_world_position.x,
			state.camera_world_position.y,
			state.camera_world_position.z,
			Time.get_ticks_msec() * 1e-3,
			state.texture_size.x,
			state.texture_size.y,
			point_count,
			state.view_count
		]
		+ _projection_to_column_major_floats(state.camera_view)
		+ _projection_to_column_major_floats(state.camera_projection)
		+ _projection_to_column_major_floats(state.camera_view_right)
		+ _projection_to_column_major_floats(state.camera_projection_right)
	)
	state.context.device.buffer_update(state.descriptors["uniforms"].rid, 0, ubo_data.size(), ubo_data)
	state.context.device.buffer_clear(state.descriptors["histogram"].rid, 0, 4 + 4 * RADIX * 4)
	state.context.device.buffer_clear(state.descriptors["tile_bounds"].rid, 0, state.tile_dims.x * state.tile_dims.y * 2 * 4)

	# Projection pass — runs once for all views
	var compute_list: int = state.context.compute_list_begin()
	state.pipelines["gsplat_projection"].call(state.context, compute_list, PackedByteArray())
	state.context.compute_list_end()

	# Radix sort — runs once using primary eye depth
	compute_list = state.context.compute_list_begin()
	for radix_shift_pass in range(4):
		var sort_push_constant := RenderingDeviceContext.create_push_constant([
			radix_shift_pass,
			point_count * MAX_SORT_ELEMENTS_PER_SPLAT * (radix_shift_pass % 2),
			point_count * MAX_SORT_ELEMENTS_PER_SPLAT * (1 - (radix_shift_pass % 2))
		])
		state.pipelines["radix_sort_upsweep"].call(state.context, compute_list, sort_push_constant, [], state.descriptors["grid_dimensions"].rid, 0)
		state.pipelines["radix_sort_spine"].call(state.context, compute_list, sort_push_constant)
		state.pipelines["radix_sort_downsweep"].call(state.context, compute_list, sort_push_constant, [], state.descriptors["grid_dimensions"].rid, 0)
	state.context.compute_list_end()

	# Boundaries pass — runs once
	compute_list = state.context.compute_list_begin()
	state.pipelines["gsplat_boundaries"].call(state.context, compute_list, PackedByteArray(), [], state.descriptors["grid_dimensions"].rid, 3 * 4)
	state.context.compute_list_end()

	# Render pass — runs once per eye with per-view output textures
	for eye_index in range(state.view_count):
		compute_list = state.context.compute_list_begin()
		var render_push_constant := RenderingDeviceContext.create_push_constant([
			0.0, -1, state.depth_capture_alpha, eye_index, state.view_count, 0, 0, 0
		])
		state.pipelines["gsplat_render"].call(
			state.context, compute_list, render_push_constant,
			[state.render_sets[eye_index]]
		)
		state.context.compute_list_end()

func _update_camera(state, camera_transform: Transform3D, camera_projection: Projection, camera_world_position: Vector3) -> void:
	state.camera_view = Projection(camera_transform.affine_inverse())
	state.camera_projection = camera_projection
	state.camera_world_position = camera_world_position

func _projection_to_column_major_floats(matrix: Projection) -> Array:
	return [
		matrix.x[0], matrix.x[1], matrix.x[2], matrix.x[3],
		matrix.y[0], matrix.y[1], matrix.y[2], matrix.y[3],
		matrix.z[0], matrix.z[1], matrix.z[2], matrix.z[3],
		matrix.w[0], matrix.w[1], matrix.w[2], matrix.w[3]
	]
