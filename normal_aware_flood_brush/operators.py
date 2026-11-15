# SPDX-License-Identifier: GPL-3.0-or-later

import math
import time
import bpy
from bpy.props import FloatProperty, IntProperty
from bpy_extras import view3d_utils
import numpy as np

from .flood_engine import (
    compute_flood_faces,
    find_unpainted_faces,
    smooth_jagged_borders,
    smooth_jagged_texture_borders,
    smooth_local_texture_paint,
    smooth_local_vertex_paint,
)
from .mesh_cache import MeshFloodCache, rasterize_uv_triangles, srgb_to_linear
from .overlay import FloodOverlayState, draw_callback_pixel, draw_callback_view
from .properties import (
    flip_brush_colors,
    get_effective_fill_rgba,
    get_effective_secondary_rgba,
    get_effective_target_mode,
    get_effective_weight_value,
    get_mesh_active_color_attribute,
    get_paint_settings,
    resolve_texture_paint_target,
    set_effective_fill_rgb,
)


def _find_view3d_window_region(context):
    """Return the main WINDOW region and RegionView3D for the active VIEW_3D area."""
    area = context.area
    if area is None or area.type != 'VIEW_3D':
        return None, None
    window_reg = None
    for reg in area.regions:
        if reg.type == 'WINDOW':
            window_reg = reg
            break
    space = context.space_data
    rv3d = space.region_3d if (space is not None and space.type == 'VIEW_3D') else None
    if rv3d is None and window_reg is not None:
        rv3d = getattr(window_reg, "data", None)
    return window_reg, rv3d


def _target_label(target_output: str) -> str:
    if target_output == 'WEIGHT':
        return "Vertex Group Weight"
    if target_output == 'TEXTURE':
        return "UV Image Texture"
    if target_output == 'COLOR':
        return "Color Attribute"
    if target_output == 'FACE_SET':
        return "Face Sets"
    return "Color & Face Sets"


def _compute_mesh_uv_mask(mesh, uv_layer, width: int, height: int, restrict_mat_index: int | None):
    """Build a (H, W) bool mask of UV-covered texels directly from a mesh."""
    if len(mesh.loop_triangles) == 0:
        mesh.calc_loop_triangles()
    nt = len(mesh.loop_triangles)
    nl = len(mesh.loops)
    if nt == 0 or nl == 0 or uv_layer is None:
        return None

    tri_loops = np.empty(nt * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("loops", tri_loops)
    tri_loops = tri_loops.reshape((nt, 3))

    loop_uvs = np.empty(nl * 2, dtype=np.float32)
    uv_layer.uv.foreach_get("vector", loop_uvs)
    loop_uvs = loop_uvs.reshape((nl, 2))
    tri_uvs = loop_uvs[tri_loops]

    if restrict_mat_index is not None:
        tri_poly = np.empty(nt, dtype=np.int32)
        mesh.loop_triangles.foreach_get("polygon_index", tri_poly)
        mat_idx = np.empty(len(mesh.polygons), dtype=np.int32)
        mesh.polygons.foreach_get("material_index", mat_idx)
        tri_uvs = tri_uvs[mat_idx[tri_poly] == restrict_mat_index]

    return rasterize_uv_triangles(tri_uvs, width, height)


class VIEW3D_OT_normal_flood_brush(bpy.types.Operator):
    """Interactive normal-aware flood fill brush for coloring-book style painting on dense meshes"""

    bl_idname = "view3d.normal_flood_brush"
    bl_label = "Normal-Aware Flood Brush"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            context.area is not None
            and context.area.type == 'VIEW_3D'
            and obj is not None
            and obj.type == 'MESH'
            and len(obj.data.polygons) > 0
            and context.mode in {'SCULPT', 'PAINT_VERTEX', 'PAINT_TEXTURE', 'PAINT_WEIGHT', 'OBJECT'}
        )

    def invoke(self, context, event):
        obj = context.active_object
        if context.mode == 'SCULPT' and getattr(obj, "use_dynamic_topology_sculpting", False):
            self.report({'WARNING'}, "Disable Dyntopo before running Normal-Aware Flood Brush")
            return {'CANCELLED'}

        window_reg, rv3d = _find_view3d_window_region(context)
        if window_reg is None or rv3d is None:
            self.report({'WARNING'}, "Active 3D Viewport window region not found")
            return {'CANCELLED'}

        props = context.scene.normal_flood_brush
        mesh = obj.data
        had_color_attr = len(mesh.color_attributes) > 0
        target_mode = get_effective_target_mode(context)

        tex_img = None
        uv_layer = None
        restrict_mat = None
        seam_bleed = 2
        if target_mode == 'TEXTURE':
            tex_img, uv_layer, restrict_mat = resolve_texture_paint_target(context, obj)
            if uv_layer is None:
                self.report({'WARNING'}, "Mesh has no active UV Map for Texture Paint flood fill")
                return {'CANCELLED'}
            if tex_img is None:
                self.report(
                    {'WARNING'},
                    "No active Paint Texture found — click 'Create Paint Texture' in Flood Paint panel",
                )
                return {'CANCELLED'}
            ip = getattr(context.tool_settings, "image_paint", None)
            if ip is not None:
                seam_bleed = int(getattr(ip, "seam_bleed", 2))

        t0 = time.perf_counter()
        try:
            depsgraph = context.evaluated_depsgraph_get()
            self._cache = MeshFloodCache(
                obj=obj,
                depsgraph=depsgraph,
                target_output=target_mode,
                default_domain=props.default_color_domain,
                default_type=props.default_color_type,
                tex_image=tex_img,
                uv_layer=uv_layer,
                tex_seam_bleed=seam_bleed,
                restrict_mat_index=restrict_mat,
                store_initial_tex=False,
            )
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to initialize flood cache: {exc}")
            return {'CANCELLED'}
        cache_ms = (time.perf_counter() - t0) * 1000.0

        self._had_color_attr = had_color_attr
        self._target_mode = target_mode
        self._window_reg = window_reg
        self._rv3d = rv3d
        self._area = context.area

        # Weight Paint target session vertex group setup
        self._session_created_vertex_group = None
        self._active_vertex_group = None
        self._initial_weights = None
        if target_mode == 'WEIGHT':
            if props.weight_auto_vertex_group or len(obj.vertex_groups) == 0:
                base_name = f"Flood_{obj.name}"
                vg = obj.vertex_groups.new(name=base_name)
                obj.vertex_groups.active = vg
                self._session_created_vertex_group = vg
                self._active_vertex_group = vg
            else:
                vg = obj.vertex_groups.active
                if vg is None:
                    vg = obj.vertex_groups[0]
                    obj.vertex_groups.active = vg
                self._active_vertex_group = vg
                self._initial_weights = {}
                for v in range(len(obj.data.vertices)):
                    try:
                        self._initial_weights[v] = vg.weight(v)
                    except RuntimeError:
                        pass

        # Per-stroke tracking and compact delta undo stack.
        self._face_stroke_ids = np.zeros(self._cache.num_faces, dtype=np.int32)
        self._stroke_counter = 0
        self._active_stroke = None
        self._undo_stack = []
        n_color_elems = len(self._cache.color_buf) if self._cache.color_buf is not None else 0
        self._stroke_elem_mask = np.zeros(n_color_elems, dtype=bool)
        n_tex_px = self._cache.tex_width * self._cache.tex_height
        self._stroke_tex_mask = np.zeros(n_tex_px, dtype=bool)

        # Interaction state.
        self._default_radius = 0.25
        self._is_painting = False
        self._is_navigating = False
        self._is_resizing = False
        self._suppress_paint_until_release = False
        self._resize_locked_toggle = False
        self._resize_anchor_x = event.mouse_x
        self._resize_start_radius = props.fill_radius
        self._last_hit_pos = None
        self._last_hit_normal = None
        self._last_hit_face = None
        self._last_tex_push_time = 0.0
        self._tex_push_pending = False

        self._overlay = FloodOverlayState(context.area)
        self._sync_overlay_props(context)
        self._overlay.set_toast(
            f"Cached {self._cache.num_faces:,} faces in {cache_ms:.0f} ms",
            duration=1.8,
        )

        # Install 3D preview and 2D HUD draw handlers.
        self._handle_view = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback_view, (self._overlay, context), 'WINDOW', 'POST_VIEW'
        )
        self._handle_pixel = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback_pixel, (self._overlay, context), 'WINDOW', 'POST_PIXEL'
        )

        context.window.cursor_modal_set('PAINT_BRUSH')
        context.workspace.status_text_set(
            "LMB: Paint  |  RMB / Enter / Space: Commit  |  S: Smooth Tool  |  I: Inverse  |  E: Sample Color  |  "
            "X: Flip Swatch  |  C: Cycle Palette  |  F / R: Radius  |  Wheel/[ ]: Angle  |  P: Preserve  |  U: Fill Unpainted  |  Ctrl+Z: Undo"
        )
        context.window_manager.modal_handler_add(self)

        # Evaluate initial cursor hover preview immediately.
        self._update_from_mouse(context, event.mouse_x, event.mouse_y, paint=False)
        self._redraw_viewport(context)
        return {'RUNNING_MODAL'}

    def _redraw_viewport(self, context):
        if getattr(self, "_window_reg", None) is not None:
            self._window_reg.tag_redraw()
        if getattr(context, "area", None) is not None:
            context.area.tag_redraw()
        elif getattr(self, "_area", None) is not None:
            self._area.tag_redraw()
        screen = getattr(context, "screen", None)
        if screen is not None and self._target_mode == 'TEXTURE':
            for ar in screen.areas:
                if ar.type in {'VIEW_3D', 'IMAGE_EDITOR'}:
                    ar.tag_redraw()

    def _sync_overlay_props(self, context):
        props = context.scene.normal_flood_brush
        self._target_mode = get_effective_target_mode(context)
        ov = self._overlay
        ov.fill_rgba = get_effective_fill_rgba(context)
        ov.secondary_rgba = get_effective_secondary_rgba(context)
        ov.angle_deg = props.normal_angle_threshold
        ov.radius = props.fill_radius
        ov.crease_resistance = props.crease_resistance
        ov.preserve_existing = props.preserve_existing
        ov.invert_flood = props.invert_flood
        ov.smooth_tool = props.smooth_tool
        ov.target_label = _target_label(self._target_mode)
        ov.is_painting = self._is_painting
        ov.is_resizing = self._is_resizing
        ov.undo_count = len(self._undo_stack)

        ps = get_paint_settings(context)
        pal = getattr(ps, "palette", None) if ps else None
        if pal and len(pal.colors) > 0:
            colors_list = list(pal.colors)
            ov.palette_colors = [tuple(float(v) for v in c.color[:3]) for c in colors_list]
            act = getattr(pal.colors, "active", None)
            ov.palette_active_idx = colors_list.index(act) if act in colors_list else -1
        else:
            ov.palette_colors = []
            ov.palette_active_idx = -1

    def _ensure_target_buffers(self, context):
        """Ensure target buffers exist if the user switched Target Output mid-modal."""
        props = context.scene.normal_flood_brush
        target_mode = get_effective_target_mode(context)
        self._target_mode = target_mode

        if target_mode == 'TEXTURE' and self._cache.tex_buf is None:
            tex_img, uv_layer, restrict_mat = resolve_texture_paint_target(context, context.active_object)
            if tex_img is not None and uv_layer is not None:
                self._cache._init_texture_paint_cache(
                    tex_img, uv_layer, restrict_mat, store_initial_tex=False
                )
                self._stroke_tex_mask = np.zeros(self._cache.tex_width * self._cache.tex_height, dtype=bool)
        if target_mode in {'COLOR', 'BOTH'} and self._cache.color_buf is None:
            self._cache.ensure_color_buffer(
                props.default_color_domain, props.default_color_type, create_if_missing=True
            )
            self._stroke_elem_mask = np.zeros(len(self._cache.color_buf), dtype=bool)
        if target_mode in {'FACE_SET', 'BOTH'} and self._cache.face_set_buf is None:
            self._cache.ensure_face_set_buffer(create_if_missing=True)

    def _is_over_ui(self, mouse_x: int, mouse_y: int) -> bool:
        """Check if mouse cursor is over any non-3D-viewport UI region (N-panel sidebar, toolbar, header)."""
        area = getattr(self, "_area", None)
        if area is None:
            return False
        if not (area.x <= mouse_x < area.x + area.width and area.y <= mouse_y < area.y + area.height):
            return True
        for r in area.regions:
            if r.type != 'WINDOW' and r.width > 1 and r.height > 1:
                if r.x <= mouse_x < r.x + r.width and r.y <= mouse_y < r.y + r.height:
                    return True
        return False

    def _sample_screen_pixel_linear(self, mouse_x: int, mouse_y: int) -> tuple[float, float, float, float] | None:
        """Sample display color at screen coordinates (mouse_x, mouse_y) converted to linear RGBA."""
        try:
            import gpu
            fb = gpu.state.active_framebuffer_get()
            buf = fb.read_color(int(mouse_x), int(mouse_y), 1, 1, 4, 0, 'FLOAT')
            buf.dimensions = 4
            srgb = np.asarray(buf[:3], dtype=np.float32)
            lin = srgb_to_linear(srgb)
            return (float(lin[0]), float(lin[1]), float(lin[2]), float(buf[3]))
        except Exception:
            return None

    def _cycle_palette_color(self, context):
        """Advance to the next color in the active palette."""
        ps = get_paint_settings(context)
        pal = getattr(ps, "palette", None) if ps else None
        if not pal or len(pal.colors) == 0:
            self._overlay.set_toast("No active palette assigned to cycle", duration=1.2)
            self._redraw_viewport(context)
            return
        colors = list(pal.colors)
        active_c = getattr(pal.colors, "active", None)
        curr_idx = colors.index(active_c) if active_c in colors else -1
        next_idx = (curr_idx + 1) % len(colors)
        next_c = colors[next_idx]
        pal.colors.active = next_c
        rgb = (float(next_c.color[0]), float(next_c.color[1]), float(next_c.color[2]))
        set_effective_fill_rgb(context, rgb)
        self._sync_overlay_props(context)
        self._overlay.set_toast(
            f"Palette [{next_idx + 1}/{len(colors)}]: ({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})",
            duration=1.4,
        )
        self._redraw_viewport(context)

    def _handle_color_pick(self, context, event) -> bool:
        """Pick/sample color on Right-Click or E from 3D surface, or cycle active palette in empty space."""
        if self._is_over_ui(event.mouse_x, event.mouse_y):
            return False

        ps = get_paint_settings(context)
        pal = getattr(ps, "palette", None) if ps else None

        # 1. Exact 3D surface sampling at raycast hit point (barycentric UV pixel or vertex).
        hit_pos, hit_normal, face_idx = self._raycast_mouse(event.mouse_x, event.mouse_y)
        if face_idx is not None and hit_pos is not None:
            col = self._cache.get_color_at_hit(face_idx, hit_pos, self._target_mode)
            if col is not None:
                rgb = (float(col[0]), float(col[1]), float(col[2]))
                set_effective_fill_rgb(context, rgb)
                matched_pal = None
                if pal and len(pal.colors) > 0:
                    best_c = min(pal.colors, key=lambda pc: sum((a - b) ** 2 for a, b in zip(pc.color, rgb)))
                    if sum((a - b) ** 2 for a, b in zip(best_c.color, rgb)) < 0.05:
                        pal.colors.active = best_c
                        matched_pal = best_c
                self._sync_overlay_props(context)
                suffix = " (palette)" if matched_pal else ""
                self._overlay.set_toast(
                    f"Sampled surface: ({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f}){suffix}",
                    duration=1.5,
                )
                self._redraw_viewport(context)
                return True

        # 2. Empty 3D space: cycle to next color in active palette.
        if pal and len(pal.colors) > 0:
            self._cycle_palette_color(context)
            return True

        self._overlay.set_toast("E: Sample surface color", duration=1.4)
        self._redraw_viewport(context)
        return False

    def _handle_smooth_key(self, context, mesh_wide: bool = False):
        props = context.scene.normal_flood_brush
        hit_pos = None if mesh_wide else self._last_hit_pos
        radius = None if mesh_wide else props.fill_radius

        if not mesh_wide and hit_pos is None:
            self._overlay.set_toast("Hover over mesh boundary to smooth vertex paint", duration=1.4)
            self._redraw_viewport(context)
            return

        t0 = time.perf_counter()

        # 1. Texture Paint mode smoothing
        if self._target_mode == 'TEXTURE':
            if self._cache.tex_buf is None or self._cache.tex_image is None:
                self._overlay.set_toast("No active texture image to smooth", duration=1.2)
                self._redraw_viewport(context)
                return

            mod_pix, old_pix = smooth_local_texture_paint(
                cache=self._cache,
                hit_pos=hit_pos,
                radius=radius,
                iterations=props.smooth_iterations,
                strength=props.smooth_strength,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if mod_pix.size == 0:
                self._overlay.set_toast(
                    "No jagged borders in cursor radius" if not mesh_wide else "No jagged borders detected",
                    duration=1.2,
                )
            else:
                self._undo_stack.append({
                    "faces": np.empty(0, dtype=np.int32),
                    "old_sids": np.empty(0, dtype=np.int32),
                    "elems": np.empty(0, dtype=np.int32),
                    "old_colors": np.empty((0, 4), dtype=np.float32),
                    "old_fs": np.empty(0, dtype=np.int32),
                    "pix": mod_pix,
                    "old_pix": old_pix,
                    "prev_next_fs_id": self._cache.next_face_set_id,
                })
                if len(self._undo_stack) > 64:
                    self._undo_stack.pop(0)

                scope = "mesh-wide" if mesh_wide else "at cursor"
                self._overlay.set_toast(
                    f"Smoothed {mod_pix.size:,} border pixels {scope} in {elapsed_ms:.1f} ms",
                    duration=1.4,
                )

            self._refresh_at_last_hit(context)
            self._redraw_viewport(context)
            return

        # 2. Vertex Paint / Color Attribute mode smoothing
        if self._cache.color_buf is None:
            self._overlay.set_toast("No active Color Attribute to smooth", duration=1.2)
            self._redraw_viewport(context)
            return

        mod_indices, old_colors = smooth_local_vertex_paint(
            cache=self._cache,
            hit_pos=hit_pos,
            radius=radius,
            iterations=props.smooth_iterations,
            strength=props.smooth_strength,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if mod_indices.size == 0:
            self._overlay.set_toast(
                "No jagged borders in cursor radius" if not mesh_wide else "No jagged borders detected",
                duration=1.2,
            )
        else:
            self._undo_stack.append({
                "faces": np.empty(0, dtype=np.int32),
                "old_sids": np.empty(0, dtype=np.int32),
                "elems": mod_indices,
                "old_colors": old_colors,
                "old_fs": np.empty(0, dtype=np.int32),
                "pix": np.empty(0, dtype=np.int32),
                "old_pix": np.empty((0, 4), dtype=np.float32),
                "prev_next_fs_id": self._cache.next_face_set_id,
            })
            if len(self._undo_stack) > 64:
                self._undo_stack.pop(0)

            scope = "mesh-wide" if mesh_wide else "at cursor"
            unit = "corners" if self._cache.color_domain == 'CORNER' else "vertices"
            self._overlay.set_toast(
                f"Smoothed {mod_indices.size:,} boundary {unit} {scope} in {elapsed_ms:.1f} ms",
                duration=1.4,
            )

        self._refresh_at_last_hit(context)
        self._redraw_viewport(context)

    def _get_rv3d(self):
        area = getattr(self, "_area", None)
        if area is not None and getattr(area, "spaces", None) and area.spaces.active:
            sp = area.spaces.active
            if hasattr(sp, "region_3d") and sp.region_3d is not None:
                return sp.region_3d
        return getattr(self, "_rv3d", None)

    def _raycast_mouse(self, mouse_x: int, mouse_y: int):
        if self._is_over_ui(mouse_x, mouse_y):
            return None, None, None
        reg = self._window_reg
        rv3d = self._get_rv3d()
        if reg is None or rv3d is None:
            return None, None, None
        mx = mouse_x - reg.x
        my = mouse_y - reg.y
        if mx < 0 or my < 0 or mx > reg.width or my > reg.height:
            return None, None, None
        coord = (mx, my)
        ray_origin = view3d_utils.region_2d_to_origin_3d(reg, rv3d, coord)
        ray_dir = view3d_utils.region_2d_to_vector_3d(reg, rv3d, coord)
        return self._cache.raycast_world(ray_origin, ray_dir)

    def _run_flood_at_hit(self, context, hit_pos, hit_normal, face_idx, paint: bool, commit_tex: bool = True):
        props = context.scene.normal_flood_brush
        self._ensure_target_buffers(context)
        self._sync_overlay_props(context)

        stroke = self._active_stroke
        sid = stroke["stroke_id"] if stroke is not None else 0
        anchor_col = stroke["anchor_color"] if stroke is not None else None
        anchor_fs = stroke["anchor_fs"] if stroke is not None else None

        t0 = time.perf_counter()
        filled_faces = compute_flood_faces(
            cache=self._cache,
            seed_face=face_idx,
            hit_pos=hit_pos,
            radius=props.fill_radius,
            angle_deg=props.normal_angle_threshold,
            crease_resistance=props.crease_resistance,
            preserve_existing=props.preserve_existing,
            target_output=self._target_mode,
            active_stroke_id=sid,
            face_stroke_ids=self._face_stroke_ids,
            anchor_color=anchor_col,
            anchor_fs=anchor_fs,
            invert_flood=props.invert_flood,
            smooth_tool=props.smooth_tool,
        )
        flood_ms = (time.perf_counter() - t0) * 1000.0
        self._overlay.last_flood_ms = flood_ms

        is_protected = (
            props.preserve_existing
            and not props.invert_flood
            and filled_faces.size == 0
            and self._face_stroke_ids[face_idx] != sid
        )
        self._overlay.protected_hit = is_protected

        if paint and filled_faces.size > 0 and stroke is not None:
            self._apply_flood_to_stroke(context, filled_faces, commit_tex=commit_tex)

        if props.show_preview_overlay:
            # If hovering over an already-painted segment in Preserve mode (which produces 0 fill faces),
            # compute the segment geometry without preserve so the red boundary outline and warning tint stay visible!
            if is_protected and not paint:
                preview_faces = compute_flood_faces(
                    cache=self._cache,
                    seed_face=face_idx,
                    hit_pos=hit_pos,
                    radius=props.fill_radius,
                    angle_deg=props.normal_angle_threshold,
                    crease_resistance=props.crease_resistance,
                    preserve_existing=False,
                    target_output=self._target_mode,
                    active_stroke_id=sid,
                    face_stroke_ids=self._face_stroke_ids,
                    invert_flood=False,
                    smooth_tool=props.smooth_tool,
                )
            else:
                preview_faces = filled_faces

            self._overlay.update_preview_geometry(
                self._cache,
                preview_faces,
                hit_pos,
                hit_normal,
                props.fill_radius,
                smooth_boundary=props.smooth_tool,
            )
        else:
            self._overlay.has_hit = False
            self._overlay.clear_preview_geometry()

    def _update_from_mouse(self, context, mouse_x: int, mouse_y: int, paint: bool, commit_tex: bool = True):
        hit_pos, hit_normal, face_idx = self._raycast_mouse(mouse_x, mouse_y)
        if face_idx is None:
            self._last_hit_pos = None
            self._last_hit_normal = None
            self._last_hit_face = None
            self._overlay.protected_hit = False
            self._overlay.has_hit = False
            self._overlay.clear_preview_geometry()
            self._sync_overlay_props(context)
            return

        if paint and face_idx == self._last_hit_face and self._last_hit_pos is not None:
            props = context.scene.normal_flood_brush
            min_move = props.fill_radius * 0.02
            if float(np.sum((hit_pos - self._last_hit_pos) ** 2)) < (min_move * min_move):
                return

        self._last_hit_pos = hit_pos
        self._last_hit_normal = hit_normal
        self._last_hit_face = face_idx
        self._run_flood_at_hit(context, hit_pos, hit_normal, face_idx, paint=paint, commit_tex=commit_tex)

    def _refresh_at_last_hit(self, context):
        self._sync_overlay_props(context)
        if self._last_hit_face is not None and self._last_hit_pos is not None:
            self._run_flood_at_hit(
                context,
                self._last_hit_pos,
                self._last_hit_normal,
                self._last_hit_face,
                paint=False,
            )

    def _begin_stroke(self, context, seed_face: int | None):
        self._ensure_target_buffers(context)
        props = context.scene.normal_flood_brush
        self._stroke_counter += 1
        sid = self._stroke_counter

        anchor_color = None
        anchor_fs = None
        if seed_face is not None and not props.invert_flood:
            anchor_color = self._cache.get_face_color_at(seed_face, self._target_mode)
            if self._cache.face_set_buf is not None:
                anchor_fs = int(self._cache.face_set_buf[seed_face])

        prev_next_fs_id = self._cache.next_face_set_id
        stroke_fs_id = None
        if self._target_mode in {'FACE_SET', 'BOTH'}:
            stroke_fs_id = prev_next_fs_id
            self._cache.next_face_set_id += 1

        self._active_stroke = {
            "stroke_id": sid,
            "fs_id": stroke_fs_id,
            "prev_next_fs_id": prev_next_fs_id,
            "anchor_color": anchor_color,
            "anchor_fs": anchor_fs,
            "face_chunks": [],
            "old_sid_chunks": [],
            "old_fs_chunks": [],
            "elem_chunks": [],
            "old_color_chunks": [],
            "pix_chunks": [],
            "old_pix_chunks": [],
            "weight_chunks": [],
        }

    def _apply_flood_to_stroke(self, context, filled_faces: np.ndarray, commit_tex: bool = True):
        stroke = self._active_stroke
        if stroke is None or filled_faces.size == 0:
            return

        props = context.scene.normal_flood_brush
        target_mode = self._target_mode
        sid = stroke["stroke_id"]
        new_faces = filled_faces[self._face_stroke_ids[filled_faces] != sid]
        if new_faces.size == 0:
            return

        stroke["face_chunks"].append(new_faces)
        stroke["old_sid_chunks"].append(self._face_stroke_ids[new_faces].copy())
        self._face_stroke_ids[new_faces] = sid

        mesh_dirty = False
        if target_mode == 'TEXTURE' and self._cache.tex_buf is not None:
            fill_rgba = get_effective_fill_rgba(context)
            pix_idx, old_pix = self._cache.apply_texture_faces(
                new_faces,
                fill_rgba,
                preserve_existing=(props.preserve_existing or props.invert_flood),
                stroke_tex_mask=self._stroke_tex_mask,
                commit=commit_tex,
                smooth_boundary=props.smooth_tool,
            )
            if pix_idx.size > 0:
                stroke["pix_chunks"].append(pix_idx)
                stroke["old_pix_chunks"].append(old_pix)
                if commit_tex:
                    self._last_tex_push_time = time.monotonic()
                    self._tex_push_pending = False
                    mesh_dirty = True
                else:
                    self._tex_push_pending = True

        if target_mode in {'FACE_SET', 'BOTH'} and self._cache.face_set_buf is not None:
            apply_faces = new_faces
            if props.smooth_tool and self._last_hit_pos is not None:
                apply_faces = smooth_flood_boundary_faces(self._cache, new_faces, self._last_hit_pos, props.fill_radius)
            stroke["old_fs_chunks"].append(self._cache.face_set_buf[apply_faces].copy())
            self._cache.apply_face_set_indices(apply_faces, stroke["fs_id"])
            mesh_dirty = True

        if target_mode in {'COLOR', 'BOTH'} and self._cache.color_buf is not None:
            elems = self._cache.faces_to_color_indices(new_faces)
            new_elems = elems[~self._stroke_elem_mask[elems]]
            if new_elems.size > 0:
                self._stroke_elem_mask[new_elems] = True
                stroke["elem_chunks"].append(new_elems)
                stroke["old_color_chunks"].append(self._cache.color_buf[new_elems].copy())
                fill_rgba = get_effective_fill_rgba(context)

                # Hard brush: 100% solid color, trim isolated single-quad stair bumps (ratios >= 0.35)
                if props.smooth_tool and self._cache.color_domain == 'POINT':
                    verts, ratios = self._cache.faces_to_vert_incident_coverage(new_faces)
                    in_new = np.isin(verts, new_elems)
                    v_apply = verts[in_new & (ratios >= 0.35)]
                    if v_apply.size > 0:
                        self._cache.apply_color_indices(v_apply, fill_rgba)
                else:
                    self._cache.apply_color_indices(new_elems, fill_rgba)
                mesh_dirty = True

        if target_mode == 'WEIGHT' and self._active_vertex_group is not None:
            vg = self._active_vertex_group
            weight_val = get_effective_weight_value(context)
            if props.smooth_tool:
                verts, ratios = self._cache.faces_to_vert_incident_coverage(new_faces)
                apply_verts = verts[ratios >= 0.35] if ratios.size > 0 else verts
            else:
                apply_verts = self._cache.faces_to_vert_indices(new_faces)

            if apply_verts.size > 0:
                old_weights = {}
                for vi in apply_verts.tolist():
                    try:
                        old_weights[vi] = vg.weight(vi)
                    except RuntimeError:
                        old_weights[vi] = None

                # Hard brush: 100% solid weight, no checkered fractional spots
                vg.add(apply_verts.tolist(), weight_val, 'REPLACE')
                stroke["weight_chunks"].append((vg, old_weights, apply_verts.copy()))
                mesh_dirty = True

        if mesh_dirty:
            self._cache.mesh.update_gpu_tag()
            self._cache.mesh.update()

    def _end_stroke(self):
        if getattr(self, "_tex_push_pending", False) and self._cache.tex_image is not None:
            self._cache.push_texture_buffer()
            self._last_tex_push_time = time.monotonic()
            self._tex_push_pending = False

        stroke = self._active_stroke
        self._active_stroke = None
        if stroke is None:
            return

        if not stroke["face_chunks"]:
            self._cache.next_face_set_id = stroke["prev_next_fs_id"]
            return

        faces = np.concatenate(stroke["face_chunks"])
        old_sids = np.concatenate(stroke["old_sid_chunks"])

        if stroke["elem_chunks"]:
            elems = np.concatenate(stroke["elem_chunks"])
            old_colors = np.concatenate(stroke["old_color_chunks"], axis=0)
            self._stroke_elem_mask[elems] = False
        else:
            elems = np.empty(0, dtype=np.int32)
            old_colors = np.empty((0, 4), dtype=np.float32)

        if stroke["old_fs_chunks"]:
            old_fs = np.concatenate(stroke["old_fs_chunks"])
        else:
            old_fs = np.empty(0, dtype=np.int32)

        if stroke["pix_chunks"]:
            pix = np.concatenate(stroke["pix_chunks"])
            old_pix = np.concatenate(stroke["old_pix_chunks"], axis=0)
            self._stroke_tex_mask[pix] = False
            if self._cache.tex_protected_mask is not None:
                self._cache.tex_protected_mask.ravel()[pix] = True
        else:
            pix = np.empty(0, dtype=np.int32)
            old_pix = np.empty((0, 4), dtype=np.float32)

        weight_data = stroke.get("weight_chunks", [])

        self._undo_stack.append({
            "faces": faces,
            "old_sids": old_sids,
            "elems": elems,
            "old_colors": old_colors,
            "old_fs": old_fs,
            "pix": pix,
            "old_pix": old_pix,
            "weight_data": weight_data,
            "prev_next_fs_id": stroke["prev_next_fs_id"],
        })
        if len(self._undo_stack) > 64:
            self._undo_stack.pop(0)

    def _undo_last_stroke(self, context):
        if self._active_stroke is not None:
            self._is_painting = False
            self._end_stroke()

        if not self._undo_stack:
            self._overlay.set_toast("Nothing to undo", duration=1.2)
            self._sync_overlay_props(context)
            return

        entry = self._undo_stack.pop()
        faces = entry["faces"]
        self._face_stroke_ids[faces] = entry["old_sids"]

        mesh_dirty = False
        if entry["elems"].size > 0:
            self._cache.restore_color_indices(entry["elems"], entry["old_colors"])
            mesh_dirty = True
        if entry["old_fs"].size > 0:
            self._cache.restore_face_set_indices(faces, entry["old_fs"])
            mesh_dirty = True
        if entry.get("weight_data"):
            for vg, old_weights, verts in reversed(entry["weight_data"]):
                for v, w in old_weights.items():
                    if w is None:
                        try:
                            vg.remove([v])
                        except RuntimeError:
                            pass
                    else:
                        vg.add([v], w, 'REPLACE')
            mesh_dirty = True
        if entry["pix"].size > 0:
            self._cache.restore_texture_pixels(entry["pix"], entry["old_pix"], faces=faces)
            if self._cache.tex_protected_mask is not None:
                prot_flat = self._cache.tex_protected_mask.ravel()
                prot_flat.fill(False)
                for st in self._undo_stack:
                    if st["pix"].size > 0:
                        prot_flat[st["pix"]] = True

        self._cache.next_face_set_id = entry["prev_next_fs_id"]
        if mesh_dirty:
            self._cache.mesh.update_gpu_tag()
            self._cache.mesh.update()

        self._overlay.set_toast(f"Undid stroke ({len(self._undo_stack)} left)", duration=1.4)
        self._refresh_at_last_hit(context)

    def _enter_resize(self, context, event, locked_toggle: bool = False):
        props = context.scene.normal_flood_brush
        self._is_resizing = True
        self._suppress_paint_until_release = False
        self._resize_locked_toggle = locked_toggle
        self._resize_anchor_x = event.mouse_x
        self._resize_start_radius = props.fill_radius
        self._sync_overlay_props(context)

    def _exit_resize(self, context):
        self._is_resizing = False
        self._resize_locked_toggle = False
        self._sync_overlay_props(context)

    def _cleanup(self, context):
        if getattr(self, "_handle_view", None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_view, 'WINDOW')
            self._handle_view = None
        if getattr(self, "_handle_pixel", None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_pixel, 'WINDOW')
            self._handle_pixel = None
        context.window.cursor_modal_restore()
        context.workspace.status_text_set(None)
        self._is_navigating = False
        if getattr(self, "_area", None) is not None:
            self._area.tag_redraw()

    def modal(self, context, event):
        props = context.scene.normal_flood_brush

        # Confirm and exit modal (or confirm locked radius resize if currently resizing).
        # Right-Click commits the paint and finishes modal (or Enter / Space).
        if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE', 'RIGHTMOUSE'} and event.value == 'PRESS':
            if self._is_resizing:
                self._exit_resize(context)
                self._refresh_at_last_hit(context)
                self._redraw_viewport(context)
                return {'RUNNING_MODAL'}
            if event.type == 'RIGHTMOUSE' and self._is_over_ui(event.mouse_x, event.mouse_y):
                # Right-clicking on the N-panel / UI palette picks color without committing/exiting
                self._handle_color_pick(context, event)
                return {'RUNNING_MODAL'}
            if self._is_painting:
                self._is_painting = False
                self._end_stroke()
            self._cleanup(context)
            return {'FINISHED'}

        # Switch/flip foreground and background color swatches of Blender (X key).
        if event.type == 'X' and not event.ctrl and not event.alt and not event.shift and event.value == 'PRESS':
            if self._is_resizing:
                self._exit_resize(context)
            new_rgba = flip_brush_colors(context)
            self._sync_overlay_props(context)
            self._overlay.set_toast(
                f"Switched Swatch Color: ({new_rgba[0]:.2f}, {new_rgba[1]:.2f}, {new_rgba[2]:.2f})",
                duration=1.2,
            )
            self._refresh_at_last_hit(context)
            self._redraw_viewport(context)
            return {'RUNNING_MODAL'}

        # Eyedropper / Sample color from 3D surface, UI palette, or cycle palette (E key or Alt+S).
        is_eyedropper = (
            (event.type == 'E' and not event.ctrl and not event.alt and not event.shift and event.value == 'PRESS')
            or (event.type == 'S' and event.alt and not event.ctrl and not event.shift and event.value == 'PRESS')
        )
        if is_eyedropper:
            if self._is_resizing:
                self._exit_resize(context)
                self._refresh_at_last_hit(context)
                self._redraw_viewport(context)
                return {'RUNNING_MODAL'}
            if self._is_painting:
                self._is_painting = False
                self._end_stroke()
            self._handle_color_pick(context, event)
            return {'RUNNING_MODAL'}

        # Smooth Tool toggle (S key), or one-shot mesh-wide smooth pass (Shift+S).
        if event.type == 'S' and not event.ctrl and not event.alt and event.value == 'PRESS':
            if self._is_resizing:
                self._exit_resize(context)
            if self._is_painting:
                self._is_painting = False
                self._end_stroke()
            if event.shift:
                self._handle_smooth_key(context, mesh_wide=True)
            else:
                props.smooth_tool = not props.smooth_tool
                state_str = "ON" if props.smooth_tool else "OFF"
                self._overlay.set_toast(f"Smooth Tool: {state_str}", duration=1.4)
                self._refresh_at_last_hit(context)
                self._redraw_viewport(context)
            return {'RUNNING_MODAL'}

        # Cycle active palette color (C).
        if event.type == 'C' and not event.ctrl and not event.alt and not event.shift and event.value == 'PRESS':
            self._cycle_palette_color(context)
            return {'RUNNING_MODAL'}

        # Cancel modal and revert strokes made during this session (or cancel radius resize and reset to default).
        if event.type == 'ESC' and event.value == 'PRESS':
            if self._is_resizing:
                default_r = getattr(self, "_default_radius", 0.25)
                props.fill_radius = default_r
                self._exit_resize(context)
                self._overlay.set_toast(f"Radius reset to default: {default_r:.3f} m", duration=1.5)
                self._refresh_at_last_hit(context)
                self._redraw_viewport(context)
                return {'RUNNING_MODAL'}
            if self._is_painting:
                self._is_painting = False
                self._end_stroke()
            self._cache.revert_all()
            if not self._had_color_attr and self._cache.color_attr is not None:
                try:
                    self._cache.mesh.color_attributes.remove(self._cache.color_attr)
                except Exception:
                    pass
            if getattr(self, "_session_created_vertex_group", None) is not None:
                try:
                    obj = context.active_object
                    if obj is not None and self._session_created_vertex_group in obj.vertex_groups.values():
                        obj.vertex_groups.remove(self._session_created_vertex_group)
                except Exception:
                    pass
            elif getattr(self, "_initial_weights", None) is not None and self._active_vertex_group is not None:
                try:
                    vg = self._active_vertex_group
                    obj = context.active_object
                    for v in range(len(obj.data.vertices)):
                        if v in self._initial_weights:
                            vg.add([v], self._initial_weights[v], 'REPLACE')
                        else:
                            try:
                                vg.remove([v])
                            except RuntimeError:
                                pass
                except Exception:
                    pass
            self._cleanup(context)
            return {'CANCELLED'}

        # Allow 3D viewport navigation (Middle Mouse orbit/pan/zoom, Alt/Ctrl/Shift+Wheel zoom/pan, Numpad).
        emulate_3_btn = getattr(getattr(context.preferences, "inputs", None), "use_mouse_emulate_3_button", False)
        is_middle_nav = (event.type == 'MIDDLEMOUSE') or (emulate_3_btn and event.alt and event.type == 'LEFTMOUSE')
        if is_middle_nav:
            if self._is_resizing:
                self._exit_resize(context)
            if self._is_painting:
                self._is_painting = False
                self._end_stroke()
            if event.value == 'RELEASE':
                # Viewport orbit/pan completed: refresh preview at new view angle immediately!
                self._update_from_mouse(context, event.mouse_x, event.mouse_y, paint=False)
                self._redraw_viewport(context)
            return {'PASS_THROUGH'}

        if (
            (event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and (event.alt or event.ctrl or event.shift))
            or event.type in {
                'NUMPAD_0', 'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4',
                'NUMPAD_5', 'NUMPAD_6', 'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9',
                'NUMPAD_PERIOD', 'NUMPAD_PLUS', 'NUMPAD_MINUS',
            }
        ):
            return {'PASS_THROUGH'}

        # Undo last flood stroke cleanly (Ctrl+Z or Backspace).
        if (
            (event.type == 'Z' and event.ctrl and not event.shift and event.value == 'PRESS')
            or (event.type == 'BACK_SPACE' and event.value == 'PRESS')
        ):
            self._undo_last_stroke(context)
            self._redraw_viewport(context)
            return {'RUNNING_MODAL'}

        # Toggle Preserve Existing Colors/Face Sets/Texture Pixels (P).
        if event.type == 'P' and not event.alt and not event.ctrl and event.value == 'PRESS':
            props.preserve_existing = not props.preserve_existing
            state_str = "ON" if props.preserve_existing else "OFF"
            self._overlay.set_toast(f"Preserve Existing: {state_str}", duration=1.4)
            self._refresh_at_last_hit(context)
            self._redraw_viewport(context)
            return {'RUNNING_MODAL'}

        # Toggle Inverse Flood mode (I): paint complementary faces inside brush radius.
        if event.type == 'I' and not event.alt and not event.ctrl and event.value == 'PRESS':
            props.invert_flood = not props.invert_flood
            state_str = "ON" if props.invert_flood else "OFF"
            self._overlay.set_toast(f"Inverse Flood (Gap-Fill): {state_str}", duration=1.4)
            self._refresh_at_last_hit(context)
            self._redraw_viewport(context)
            return {'RUNNING_MODAL'}

        # Fill all remaining unpainted / unreachable gaps across the mesh (U), undoable.
        if event.type == 'U' and not event.alt and not event.ctrl and event.value == 'PRESS':
            if not self._is_painting:
                self._ensure_target_buffers(context)
                unpainted = find_unpainted_faces(
                    cache=self._cache,
                    target_output=self._target_mode,
                    face_stroke_ids=self._face_stroke_ids,
                )
                if unpainted.size == 0:
                    self._overlay.set_toast("No unpainted gaps remaining", duration=1.4)
                else:
                    self._begin_stroke(context, None)
                    self._apply_flood_to_stroke(context, unpainted, commit_tex=True)
                    self._end_stroke()
                    self._overlay.set_toast(
                        f"Filled {unpainted.size:,} unpainted faces", duration=1.8
                    )
                self._refresh_at_last_hit(context)
                self._redraw_viewport(context)
            return {'RUNNING_MODAL'}

        # Radius resize toggle (F or R): tap to enter dynamic resize, click/Enter/Space/F/R confirms, Esc cancels.
        if event.type in {'F', 'R'} and not event.ctrl and not event.alt and not event.shift:
            if event.value == 'PRESS':
                if self._is_resizing:
                    self._exit_resize(context)
                    self._refresh_at_last_hit(context)
                else:
                    self._enter_resize(context, event, locked_toggle=True)
                self._redraw_viewport(context)
                return {'RUNNING_MODAL'}

        # Adjust Normal Angle Threshold via Scroll Wheel or [ / ].
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'LEFT_BRACKET', 'RIGHT_BRACKET'}:
            is_press = event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} or event.value == 'PRESS'
            if is_press:
                direction = 1.0 if event.type in {'WHEELUPMOUSE', 'RIGHT_BRACKET'} else -1.0
                step = 0.5 if event.alt else 2.0
                props.normal_angle_threshold = max(
                    0.0, min(180.0, props.normal_angle_threshold + direction * step)
                )
                self._overlay.set_toast(
                    f"Normal Angle Threshold: {props.normal_angle_threshold:.1f}°",
                    duration=1.1,
                )
                self._refresh_at_last_hit(context)
                self._redraw_viewport(context)
                return {'RUNNING_MODAL'}

        # Left-Click press / release for continuous painting (or confirming radius resize).
        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                if self._is_resizing:
                    self._exit_resize(context)
                    self._suppress_paint_until_release = True
                    self._refresh_at_last_hit(context)
                    self._redraw_viewport(context)
                    return {'RUNNING_MODAL'}
                if self._suppress_paint_until_release:
                    return {'RUNNING_MODAL'}
                if self._is_over_ui(event.mouse_x, event.mouse_y):
                    if self._is_painting:
                        self._is_painting = False
                        self._end_stroke()
                    self._sync_overlay_props(context)
                    self._redraw_viewport(context)
                    return {'PASS_THROUGH'}
                hit_pos, hit_normal, face_idx = self._raycast_mouse(event.mouse_x, event.mouse_y)
                self._is_painting = True
                self._begin_stroke(context, face_idx)
                if face_idx is not None:
                    self._last_hit_pos = hit_pos
                    self._last_hit_normal = hit_normal
                    self._last_hit_face = face_idx
                    self._run_flood_at_hit(context, hit_pos, hit_normal, face_idx, paint=True, commit_tex=True)
                else:
                    self._sync_overlay_props(context)
                self._redraw_viewport(context)
                return {'RUNNING_MODAL'}

            if event.value == 'RELEASE':
                if self._suppress_paint_until_release:
                    self._suppress_paint_until_release = False
                    self._refresh_at_last_hit(context)
                    self._redraw_viewport(context)
                    return {'RUNNING_MODAL'}
                if self._is_painting:
                    self._is_painting = False
                    self._end_stroke()
                    self._update_from_mouse(context, event.mouse_x, event.mouse_y, paint=False)
                    self._redraw_viewport(context)
                    return {'RUNNING_MODAL'}
                if self._is_over_ui(event.mouse_x, event.mouse_y):
                    self._sync_overlay_props(context)
                    self._redraw_viewport(context)
                    return {'PASS_THROUGH'}

        # Mouse movement: resize radius dynamically OR raycast & preview/paint continuously.
        if event.type == 'MOUSEMOVE':
            if self._is_resizing:
                dx = event.mouse_x - self._resize_anchor_x
                new_rad = self._resize_start_radius * math.exp(dx * 0.005)
                props.fill_radius = max(0.001, min(50.0, new_rad))
                self._overlay.update_ring_radius(props.fill_radius)
                self._sync_overlay_props(context)
                self._redraw_viewport(context)
                return {'RUNNING_MODAL'}
            elif self._is_over_ui(event.mouse_x, event.mouse_y):
                if self._is_painting:
                    self._is_painting = False
                    self._end_stroke()
                self._last_hit_pos = None
                self._last_hit_normal = None
                self._last_hit_face = None
                self._overlay.protected_hit = False
                self._overlay.has_hit = False
                self._overlay.clear_preview_geometry()
                self._sync_overlay_props(context)
                self._redraw_viewport(context)
                return {'PASS_THROUGH'}
            else:
                commit_tex = (
                    not self._is_painting
                    or self._target_mode != 'TEXTURE'
                    or ((time.monotonic() - self._last_tex_push_time) >= 0.033)
                )
                paint_flag = self._is_painting and not self._suppress_paint_until_release
                self._update_from_mouse(
                    context,
                    event.mouse_x,
                    event.mouse_y,
                    paint=paint_flag,
                    commit_tex=commit_tex,
                )
                self._redraw_viewport(context)
                return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}


class VIEW3D_OT_smooth_jagged_borders(bpy.types.Operator):
    """Smooth jagged triangle stair-stepping along color attribute boundaries for clean 3D printing"""

    bl_idname = "view3d.flood_smooth_jagged_borders"
    bl_label = "Smooth Jagged Borders"
    bl_options = {'REGISTER', 'UNDO'}

    iterations: IntProperty(
        name="Iterations",
        description="Number of boundary smoothing passes",
        default=2,
        min=1,
        max=20,
    )
    strength: FloatProperty(
        name="Strength",
        description="Color averaging blend factor along detected boundaries",
        default=0.75,
        min=0.05,
        max=1.0,
        subtype='FACTOR',
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            return False
        if get_effective_target_mode(context) == 'TEXTURE':
            img, uv_layer, _ = resolve_texture_paint_target(context, obj)
            return img is not None and uv_layer is not None
        ca = get_mesh_active_color_attribute(obj.data)
        return ca is not None

    def invoke(self, context, event):
        props = context.scene.normal_flood_brush
        self.iterations = props.smooth_iterations
        self.strength = props.smooth_strength
        return self.execute(context)

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            return {'CANCELLED'}
        mesh = obj.data

        t0 = time.perf_counter()
        if get_effective_target_mode(context) == 'TEXTURE':
            img, uv_layer, restrict_mat = resolve_texture_paint_target(context, obj)
            if img is None or uv_layer is None:
                self.report({'WARNING'}, "Texture Paint mode requires an active UV Map and Paint Image")
                return {'CANCELLED'}
            w, h = int(img.size[0]), int(img.size[1])
            cov = _compute_mesh_uv_mask(mesh, uv_layer, w, h, restrict_mat)
            ip = getattr(context.tool_settings, "image_paint", None)
            seam_bleed = int(getattr(ip, "seam_bleed", 2)) if ip is not None else 2
            modified = smooth_jagged_texture_borders(
                image=img,
                uv_covered_mask=cov,
                iterations=self.iterations,
                strength=self.strength,
                seam_bleed=seam_bleed,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if modified == 0:
                self.report({'INFO'}, "No jagged texture color boundaries detected")
            else:
                self.report({'INFO'}, f"Smoothed {modified:,} texture border pixels in {elapsed_ms:.1f} ms")
            return {'FINISHED'}

        if get_mesh_active_color_attribute(mesh) is None:
            self.report({'WARNING'}, "Mesh has no active Color Attribute to smooth")
            return {'CANCELLED'}

        modified = smooth_jagged_borders(
            mesh=mesh,
            iterations=self.iterations,
            strength=self.strength,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if modified == 0:
            self.report({'INFO'}, "No jagged color boundaries detected")
        else:
            self.report({'INFO'}, f"Smoothed {modified:,} boundary elements in {elapsed_ms:.1f} ms")
        return {'FINISHED'}


class VIEW3D_OT_flood_setup_shading(bpy.types.Operator):
    """Configure Solid viewport shading and overlays to display Color Attributes, Textures, and Face Sets"""

    bl_idname = "view3d.flood_setup_shading"
    bl_label = "Show Color / Texture in Viewport"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return context.space_data is not None and context.space_data.type == 'VIEW_3D'

    def execute(self, context):
        space = context.space_data
        props = context.scene.normal_flood_brush
        target_mode = get_effective_target_mode(context)
        if space.shading.type == 'SOLID':
            space.shading.color_type = 'TEXTURE' if target_mode == 'TEXTURE' else 'VERTEX'
        if target_mode != 'TEXTURE' and props.target_output in {'FACE_SET', 'BOTH'}:
            space.overlay.show_sculpt_face_sets = True
        self.report({'INFO'}, "Viewport configured for Flood Paint display")
        return {'FINISHED'}


class VIEW3D_OT_flood_create_paint_image(bpy.types.Operator):
    """Create a blank white paint texture and attach it for Texture Paint flood filling"""

    bl_idname = "view3d.flood_create_paint_image"
    bl_label = "Create Paint Texture"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        mesh = obj.data
        props = context.scene.normal_flood_brush
        res = int(props.texpaint_resolution)
        ip = getattr(context.tool_settings, "image_paint", None)

        img = bpy.data.images.new(name=f"{obj.name}_FloodPaint", width=res, height=res, alpha=True)
        img.pixels.foreach_set(np.ones(res * res * 4, dtype=np.float32))
        img.update()
        if hasattr(img, "gl_free"):
            img.gl_free()
        if hasattr(img, "update_tag"):
            img.update_tag()

        if ip is not None and ip.mode == 'IMAGE':
            ip.canvas = img
        else:
            mat = obj.active_material
            if mat is None:
                mat = bpy.data.materials.new(name=f"{obj.name}_FloodPaintMat")
                obj.data.materials.append(mat)
            if getattr(mat, "node_tree", None) is None:
                mat.use_nodes = True
            nt = mat.node_tree
            if nt is not None:
                bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if bsdf is None:
                    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
                tex_node = nt.nodes.new("ShaderNodeTexImage")
                tex_node.image = img
                tex_node.location = (bsdf.location.x - 320.0, bsdf.location.y)
                nt.nodes.active = tex_node
                base_in = bsdf.inputs.get("Base Color")
                if base_in is not None:
                    for link in list(base_in.links):
                        nt.links.remove(link)
                    nt.links.new(tex_node.outputs["Color"], base_in)
            if ip is not None:
                ip.canvas = img

        if len(mesh.uv_layers) == 0:
            self.report({'WARNING'}, f"Created {res}x{res} '{img.name}' (mesh needs a UV Map before painting)")
        else:
            self.report({'INFO'}, f"Created {res}x{res} Paint Texture '{img.name}'")
        return {'FINISHED'}


class VIEW3D_OT_flood_reset_canvas(bpy.types.Operator):
    """Fill the active Paint Texture, Color Attribute, and/or Face Sets cleanly back to blank"""

    bl_idname = "view3d.flood_reset_canvas"
    bl_label = "Initialize / Reset Canvas"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and len(obj.data.polygons) > 0

    def execute(self, context):
        obj = context.active_object
        mesh = obj.data
        props = context.scene.normal_flood_brush
        target_mode = get_effective_target_mode(context)

        if target_mode == 'TEXTURE':
            img, _uv, _mat = resolve_texture_paint_target(context, obj)
            if img is None:
                self.report({'WARNING'}, "No active Paint Texture to reset")
                return {'CANCELLED'}
            w, h = int(img.size[0]), int(img.size[1])
            if w > 0 and h > 0:
                img.pixels.foreach_set(np.ones(w * h * 4, dtype=np.float32))
                img.update()
                if hasattr(img, "gl_free"):
                    img.gl_free()
                if hasattr(img, "update_tag"):
                    img.update_tag()
            screen = getattr(context, "screen", None)
            if screen is not None:
                for ar in screen.areas:
                    if ar.type in {'VIEW_3D', 'IMAGE_EDITOR'}:
                        ar.tag_redraw()
            elif context.area is not None:
                context.area.tag_redraw()
            self.report({'INFO'}, f"Reset texture '{img.name}' to white")
            return {'FINISHED'}

        if target_mode in {'COLOR', 'BOTH'}:
            ca = get_mesh_active_color_attribute(mesh)
            if ca is None:
                ca = mesh.color_attributes.new(
                    name="Color",
                    type=props.default_color_type,
                    domain=props.default_color_domain,
                )
            mesh.color_attributes.active_color = ca
            mesh.color_attributes.active = ca
            if mesh.color_attributes.render_color_index < 0 and mesh.color_attributes.active_color_index >= 0:
                mesh.color_attributes.render_color_index = mesh.color_attributes.active_color_index

            white = np.ones((len(ca.data), 4), dtype=np.float32)
            if ca.data_type == 'BYTE_COLOR':
                ca.data.foreach_set("color_srgb", white.ravel())
            else:
                ca.data.foreach_set("color", white.ravel())

        if target_mode in {'FACE_SET', 'BOTH'}:
            fs = mesh.attributes.get(".sculpt_face_set")
            if fs is None:
                fs = mesh.attributes.new(name=".sculpt_face_set", type='INT', domain='FACE')
            fs.data.foreach_set("value", np.ones(len(mesh.polygons), dtype=np.int32))

        mesh.update_gpu_tag()
        mesh.update()
        self.report({'INFO'}, "Canvas reset cleanly")
        return {'FINISHED'}


class VIEW3D_OT_flood_fill_unpainted(bpy.types.Operator):
    """Fill all remaining unpainted faces/gaps (matching blank white or Face Set 1) with the active fill color"""

    bl_idname = "view3d.flood_fill_unpainted"
    bl_label = "Fill Unpainted Gaps"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and len(obj.data.polygons) > 0

    def execute(self, context):
        obj = context.active_object
        mesh = obj.data
        props = context.scene.normal_flood_brush
        target_mode = get_effective_target_mode(context)

        tex_img = None
        uv_layer = None
        restrict_mat = None
        seam_bleed = 2
        if target_mode == 'TEXTURE':
            tex_img, uv_layer, restrict_mat = resolve_texture_paint_target(context, obj)
            if tex_img is None or uv_layer is None:
                self.report({'WARNING'}, "Texture Paint mode requires an active UV Map and Paint Image")
                return {'CANCELLED'}
            ip = getattr(context.tool_settings, "image_paint", None)
            if ip is not None:
                seam_bleed = int(getattr(ip, "seam_bleed", 2))

        t0 = time.perf_counter()
        cache = MeshFloodCache(
            obj=obj,
            depsgraph=context.evaluated_depsgraph_get(),
            target_output=target_mode,
            default_domain=props.default_color_domain,
            default_type=props.default_color_type,
            tex_image=tex_img,
            uv_layer=uv_layer,
            tex_seam_bleed=seam_bleed,
            restrict_mat_index=restrict_mat,
            store_initial_tex=False,
            build_adjacency=False,
            build_bvh=False,
        )
        unpainted = find_unpainted_faces(cache=cache, target_output=target_mode)
        if unpainted.size == 0:
            self.report({'INFO'}, "No unpainted faces remaining")
            return {'FINISHED'}

        fill_rgba = get_effective_fill_rgba(context)
        if target_mode == 'TEXTURE':
            cache.apply_texture_faces(unpainted, fill_rgba)
        if target_mode in {'COLOR', 'BOTH'}:
            elems = cache.faces_to_color_indices(unpainted)
            cache.apply_color_indices(elems, fill_rgba)
        if target_mode in {'FACE_SET', 'BOTH'}:
            cache.apply_face_set_indices(unpainted, cache.next_face_set_id)
        if target_mode != 'TEXTURE':
            mesh.update_gpu_tag()
            mesh.update()
        screen = getattr(context, "screen", None)
        if screen is not None and target_mode == 'TEXTURE':
            for ar in screen.areas:
                if ar.type in {'VIEW_3D', 'IMAGE_EDITOR'}:
                    ar.tag_redraw()
        elif context.area is not None:
            context.area.tag_redraw()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.report({'INFO'}, f"Filled {unpainted.size:,} unpainted faces in {elapsed_ms:.1f} ms")
        return {'FINISHED'}


class VIEW3D_OT_flood_flip_colors(bpy.types.Operator):
    """Switch/flip the foreground (primary) and background (secondary) brush colors in Blender (Hotkey: X)"""

    bl_idname = "view3d.flood_flip_colors"
    bl_label = "Flip Swatch Colors"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        new_rgba = flip_brush_colors(context)
        self.report({'INFO'}, f"Switched swatch: ({new_rgba[0]:.2f}, {new_rgba[1]:.2f}, {new_rgba[2]:.2f})")
        return {'FINISHED'}


classes = (
    VIEW3D_OT_normal_flood_brush,
    VIEW3D_OT_smooth_jagged_borders,
    VIEW3D_OT_flood_setup_shading,
    VIEW3D_OT_flood_create_paint_image,
    VIEW3D_OT_flood_fill_unpainted,
    VIEW3D_OT_flood_reset_canvas,
    VIEW3D_OT_flood_flip_colors,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
