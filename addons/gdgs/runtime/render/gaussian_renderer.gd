@tool
extends RefCounted
class_name GaussianRenderer

const RenderingDeviceContext := preload("res://addons/gdgs/runtime/render/gaussian_rendering_device_context.gd")
const RADIX := 256
const MAX_SORT_ELEMENTS_PER_SPLAT := 10
const _DEBUG_TAG := "GDGS"

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
	min_radius: float = 0.0,
	debug_logging: bool = false
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

	_rasterize_state(state, point_count, debug_logging)

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

func _rasterize_state(state, point_count: int, debug_logging: bool) -> void:
	if state.context == null:
		return

	var should_log := debug_logging and Engine.get_frames_drawn() % 60 == 0

	if should_log:
		print("[%s] --- UBO data --- point_count=%d view_count=%d tex=%s" % [
			_DEBUG_TAG, point_count, state.view_count, str(state.texture_size)])
		print("[%s] UBO camera_world_pos=%s" % [_DEBUG_TAG, str(state.camera_world_position)])
		# Log view matrix (should be cam_transform.inverse — NOT identity if head tracking works)
		var vm: Projection = state.camera_view
		print("[%s] UBO view_matrix row0=[%.4f, %.4f, %.4f, %.4f]" % [_DEBUG_TAG, vm.x[0], vm.y[0], vm.z[0], vm.w[0]])
		print("[%s] UBO view_matrix row1=[%.4f, %.4f, %.4f, %.4f]" % [_DEBUG_TAG, vm.x[1], vm.y[1], vm.z[1], vm.w[1]])
		print("[%s] UBO view_matrix row2=[%.4f, %.4f, %.4f, %.4f]" % [_DEBUG_TAG, vm.x[2], vm.y[2], vm.z[2], vm.w[2]])
		print("[%s] UBO view_matrix row3=[%.4f, %.4f, %.4f, %.4f]" % [_DEBUG_TAG, vm.x[3], vm.y[3], vm.z[3], vm.w[3]])
		# Log projection matrix
		var pm: Projection = state.camera_projection
		print("[%s] UBO projection row0=[%.4f, %.4f, %.4f, %.4f]" % [_DEBUG_TAG, pm.x[0], pm.y[0], pm.z[0], pm.w[0]])
		print("[%s] UBO projection row1=[%.4f, %.4f, %.4f, %.4f]" % [_DEBUG_TAG, pm.x[1], pm.y[1], pm.z[1], pm.w[1]])
		print("[%s] UBO projection row2=[%.4f, %.4f, %.4f, %.4f]" % [_DEBUG_TAG, pm.x[2], pm.y[2], pm.z[2], pm.w[2]])
		print("[%s] UBO projection row3=[%.4f, %.4f, %.4f, %.4f]" % [_DEBUG_TAG, pm.x[3], pm.y[3], pm.z[3], pm.w[3]])
		if state.view_count >= 2:
			var vmr: Projection = state.camera_view_right
			print("[%s] UBO view_matrix_right row0=[%.4f, %.4f, %.4f, %.4f]" % [_DEBUG_TAG, vmr.x[0], vmr.y[0], vmr.z[0], vmr.w[0]])
			print("[%s] UBO view_matrix_right row3=[%.4f, %.4f, %.4f, %.4f]" % [_DEBUG_TAG, vmr.x[3], vmr.y[3], vmr.z[3], vmr.w[3]])
			var pmr: Projection = state.camera_projection_right
			print("[%s] UBO projection_right row0=[%.4f, %.4f, %.4f, %.4f]" % [_DEBUG_TAG, pmr.x[0], pmr.y[0], pmr.z[0], pmr.w[0]])
			print("[%s] UBO projection_right row3=[%.4f, %.4f, %.4f, %.4f]" % [_DEBUG_TAG, pmr.x[3], pmr.y[3], pmr.z[3], pmr.w[3]])
		print("[%s] sort_capacity=%d" % [_DEBUG_TAG, state.sort_capacity])
		# Check if view_matrix is identity (would cause "stuck to viewport")
		var is_identity := is_equal_approx(vm.x[0], 1.0) and is_equal_approx(vm.y[1], 1.0) and is_equal_approx(vm.z[2], 1.0) and is_equal_approx(vm.w[3], 1.0) and is_equal_approx(vm.w[0], 0.0) and is_equal_approx(vm.w[1], 0.0) and is_equal_approx(vm.w[2], 0.0)
		if is_identity:
			push_warning("[%s] WARNING: view_matrix is IDENTITY — head tracking not applied!" % _DEBUG_TAG)

	var ubo_data := RenderingDeviceContext.create_buffer_data(
		[
			state.camera_world_position.x,
			state.camera_world_position.y,
			state.camera_world_position.z,
			Time.get_ticks_msec() * 1e-3,
			state.texture_size.x,
			state.texture_size.y,
			point_count,
			state.view_count,
			state.sort_capacity,
			0,
			0,
			0
		]
		+ _projection_to_column_major_floats(state.camera_view)
		+ _projection_to_column_major_floats(state.camera_projection)
		+ _projection_to_column_major_floats(state.camera_view_right)
		+ _projection_to_column_major_floats(state.camera_projection_right)
	)
	state.context.device.buffer_update(state.descriptors["uniforms"].rid, 0, ubo_data.size(), ubo_data)
	# --- Clear pass (separate compute list) ---
	# On Adreno GPUs, compute_list_add_barrier() between dispatches within a
	# single compute list does NOT reliably flush atomic/SSBO writes.
	# Splitting clear into its own compute list forces a full queue submission
	# boundary, which provides stronger synchronisation guarantees.
	var clear_list: int = state.context.compute_list_begin()
	var clear_push_constant := RenderingDeviceContext.create_push_constant([1, 0, 0.0, 0])
	state.pipelines["gsplat_projection_clear"].call(state.context, clear_list, clear_push_constant)
	var tile_clear_push_constant := RenderingDeviceContext.create_push_constant([1, state.tile_count])
	state.pipelines["gsplat_tile_bounds_clear"].call(state.context, clear_list, tile_clear_push_constant)
	state.context.compute_list_end()

	# --- Main compute list (projection → sort → boundaries → render) ---
	var compute_list: int = state.context.compute_list_begin()

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

	# Boundaries pass — runs once
	var boundaries_push_constant := RenderingDeviceContext.create_push_constant([0, 0])
	state.pipelines["gsplat_boundaries"].call(state.context, compute_list, boundaries_push_constant)

	# Render pass — runs once per eye
	for eye_index in range(state.view_count):
		var render_push_constant := RenderingDeviceContext.create_push_constant([
			0.0, -1, state.depth_capture_alpha, eye_index, state.view_count, 0, 0, 0
		])
		state.pipelines["gsplat_render"].call(
			state.context, compute_list, render_push_constant
		)

	state.context.compute_list_end()
	if should_log:
		_log_sort_stats(state)

func _update_camera(state, camera_transform: Transform3D, camera_projection: Projection, camera_world_position: Vector3) -> void:
	state.camera_view = Projection(camera_transform.affine_inverse())
	state.camera_projection = camera_projection
	state.camera_world_position = camera_world_position

func _log_sort_stats(state) -> void:
	if state.context == null or not state.descriptors.has("histogram"):
		return
	if not state.context.device.has_method("buffer_get_data"):
		return
	var histogram_rid: RID = state.descriptors["histogram"].rid
	var data: PackedByteArray = state.context.device.buffer_get_data(histogram_rid, 0, 8)
	if data.size() < 8:
		return
	var sort_size := data.decode_u32(0)
	var overflow_count := data.decode_u32(4)
	print("[%s] sort_buffer_size=%d sort_overflow_count=%d capacity=%d" % [
		_DEBUG_TAG, sort_size, overflow_count, state.sort_capacity
	])

func _projection_to_column_major_floats(matrix: Projection) -> Array:
	return [
		matrix.x[0], matrix.x[1], matrix.x[2], matrix.x[3],
		matrix.y[0], matrix.y[1], matrix.y[2], matrix.y[3],
		matrix.z[0], matrix.z[1], matrix.z[2], matrix.z[3],
		matrix.w[0], matrix.w[1], matrix.w[2], matrix.w[3]
	]
