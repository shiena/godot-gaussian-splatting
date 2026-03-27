@tool
extends RefCounted
class_name GaussianRenderer

const RenderingDeviceContext := preload("res://addons/gdgs/runtime/render/gaussian_rendering_device_context.gd")
const RADIX := 256
const MAX_SORT_ELEMENTS_PER_SPLAT := 10

## Multiview render. Runs projection + sort once, then renders per eye.
## camera_data_array: Array of {"transform": Transform3D, "projection": Projection, "world_position": Vector3}
## Returns {"views": [{"color_alpha_texture": RID, "depth_texture": RID}, ...]}
func render_for_compositor_multiview(
	state_cache: GaussianGpuStateCache,
	scene_registry: GaussianSceneRegistry,
	texture_size: Vector2i,
	camera_data_array: Array,
	depth_capture_alpha: float = 0.5,
	sh_degree: int = 3,
	min_radius: float = 0.0
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

	# Update view_count — triggers GPU rebuild if changed (resizes culled_buffer
	# and creates per-view render textures / descriptor sets).
	if state.view_count != view_count:
		state.view_count = view_count
		state.needs_gpu_rebuild = true

	# Update primary (left) eye camera
	var primary: Dictionary = camera_data_array[0]
	_update_camera(state, primary["transform"], primary["projection"], primary["world_position"])
	state.depth_capture_alpha = clampf(depth_capture_alpha, 0.0, 1.0)
	state.sh_degree = clampi(sh_degree, 0, 3)
	state.min_radius = maxf(min_radius, 0.0)

	var unique_data_size := scene_registry.get_point_data_byte().size()

	# Update right eye camera (stereo).
	# For mono, reset to identity so the UBO contains clean data.
	if view_count >= 2:
		var right: Dictionary = camera_data_array[1]
		state.camera_view_right = Projection(right["transform"].affine_inverse())
		state.camera_projection_right = right["projection"]
		state.camera_world_position_right = right["world_position"]
	else:
		state.camera_view_right = Projection.IDENTITY
		state.camera_projection_right = Projection.IDENTITY
		state.camera_world_position_right = Vector3.ZERO

	if state.context == null or state.needs_gpu_rebuild:
		state_cache.rebuild_gpu_state(state, point_count, unique_data_size, scene_registry.get_instance_count())
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
	# Use buffer_update with zero data instead of buffer_clear.
	# buffer_clear uses vkCmdFillBuffer (transfer op) which may lack a
	# proper transfer→compute barrier on some mobile Vulkan drivers.
	# buffer_update goes through a staging-buffer path that is more
	# reliably synchronised with subsequent compute dispatches.
	var histogram_clear_size: int = 4 + 4 * RADIX * 4
	var zero_histogram := PackedByteArray()
	zero_histogram.resize(histogram_clear_size)
	zero_histogram.fill(0)
	state.context.device.buffer_update(state.descriptors["histogram"].rid, 0, histogram_clear_size, zero_histogram)
	var tile_bounds_clear_size: int = state.tile_dims.x * state.tile_dims.y * 2 * 4
	var zero_tile_bounds := PackedByteArray()
	zero_tile_bounds.resize(tile_bounds_clear_size)
	zero_tile_bounds.fill(0)
	state.context.device.buffer_update(state.descriptors["tile_bounds"].rid, 0, tile_bounds_clear_size, zero_tile_bounds)

	# All compute work runs in a single compute list so that
	# compute_list_add_barrier() (called after every dispatch inside
	# create_pipeline) guarantees correct memory ordering on mobile GPUs
	# where inter-list synchronisation is not implicit.
	var compute_list: int = state.context.compute_list_begin()

	# Clear histogram inside compute list (avoids transfer→compute barrier issues)
	var clear_push_constant := RenderingDeviceContext.create_push_constant([1, 0, 0.0, 0])
	state.pipelines["gsplat_projection_clear"].call(state.context, compute_list, clear_push_constant)

	# Projection pass — runs once for all views
	var projection_push_constant := RenderingDeviceContext.create_push_constant([0, state.sh_degree, state.min_radius, 0])
	state.pipelines["gsplat_projection"].call(state.context, compute_list, projection_push_constant)

	# Radix sort — runs once using primary eye depth
	for radix_shift_pass in range(4):
		var sort_push_constant := RenderingDeviceContext.create_push_constant([
			radix_shift_pass,
			point_count * MAX_SORT_ELEMENTS_PER_SPLAT * (radix_shift_pass % 2),
			point_count * MAX_SORT_ELEMENTS_PER_SPLAT * (1 - (radix_shift_pass % 2))
		])
		state.pipelines["radix_sort_upsweep"].call(state.context, compute_list, sort_push_constant)
		state.pipelines["radix_sort_spine"].call(state.context, compute_list, sort_push_constant)
		state.pipelines["radix_sort_downsweep"].call(state.context, compute_list, sort_push_constant)

	# Clear tile_bounds inside compute list (avoids transfer→compute barrier issues)
	var tile_clear_push_constant := RenderingDeviceContext.create_push_constant([1, state.tile_count])
	state.pipelines["gsplat_tile_bounds_clear"].call(state.context, compute_list, tile_clear_push_constant)

	# Boundaries pass — runs once
	var boundaries_push_constant := RenderingDeviceContext.create_push_constant([0, 0])
	state.pipelines["gsplat_boundaries"].call(state.context, compute_list, boundaries_push_constant)

	# Render pass — runs once per eye with per-view output textures
	for eye_index in range(state.view_count):
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
