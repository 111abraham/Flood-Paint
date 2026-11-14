# SPDX-License-Identifier: GPL-3.0-or-later

import math
import time
import blf
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
import numpy as np


def _make_uniform_shader():
    return gpu.shader.from_builtin('UNIFORM_COLOR')


def _make_polyline_shader():
    return gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')


class FloodOverlayState:
    """Manages cached 3D GPU preview batches and 2D BLF HUD state for the modal brush."""

    def __init__(self, area):
        self.area = area
        self.tri_batch = None
        self.edge_batch = None
        self.ring_batch = None
        self.matrix_world = None

        self.has_hit = False
        self.hit_pos_world = Vector((0.0, 0.0, 0.0))
        self.hit_normal_world = Vector((0.0, 0.0, 1.0))
        self.preview_face_count = 0
        self.last_flood_ms = 0.0

        self.fill_rgba = (0.9, 0.25, 0.2, 1.0)
        self.secondary_rgba = (1.0, 1.0, 1.0, 1.0)
        self.angle_deg = 30.0
        self.radius = 0.25
        self.crease_resistance = 0.95
        self.preserve_existing = False
        self.invert_flood = False
        self.smooth_tool = False
        self.target_label = "Color Attribute"
        self.palette_colors: list[tuple[float, float, float]] = []
        self.palette_active_idx: int = -1
        self.is_painting = False
        self.is_resizing = False
        self.undo_count = 0
        self.protected_hit = False

        self._toast_text = ""
        self._toast_expires = 0.0

    def set_toast(self, text: str, duration: float = 1.6):
        self._toast_text = text
        self._toast_expires = time.monotonic() + duration

    def clear_preview_geometry(self):
        self.tri_batch = None
        self.edge_batch = None
        self.ring_batch = None
        self.preview_face_count = 0

    def update_preview_geometry(
        self,
        cache,
        preview_faces: np.ndarray,
        hit_pos_scaled: np.ndarray | None,
        hit_normal: np.ndarray | None,
        radius: float,
        smooth_boundary: bool = False,
    ):
        self.matrix_world = cache.matrix_world
        self.radius = float(radius)

        if hit_pos_scaled is None or hit_normal is None:
            self.has_hit = False
            self.clear_preview_geometry()
            return

        self.has_hit = True
        self.hit_pos_world = Vector(hit_pos_scaled) + cache.matrix_world.translation
        self.hit_normal_world = Vector(hit_normal).normalized()
        self.preview_face_count = int(preview_faces.size)

        # Build surface-aligned 3D radius circle around the hit point.
        n = self.hit_normal_world
        ref = Vector((0.0, 0.0, 1.0)) if abs(n.z) < 0.9 else Vector((1.0, 0.0, 0.0))
        t1 = n.cross(ref).normalized()
        t2 = n.cross(t1).normalized()
        segments = 48
        r = max(float(radius), 1e-5)
        offset = n * max(r * 0.0025, 1e-4)
        center = self.hit_pos_world + offset
        ring_pts = []
        step = (2.0 * math.pi) / segments
        for i in range(segments + 1):
            a = i * step
            ring_pts.append(center + (t1 * math.cos(a) + t2 * math.sin(a)) * r)

        poly_shader = _make_polyline_shader()
        self.ring_batch = batch_for_shader(poly_shader, 'LINE_STRIP', {"pos": ring_pts})

        if preview_faces.size == 0:
            self.tri_batch = None
            self.edge_batch = None
            return

        tri_coords, edge_coords = cache.extract_preview_geometry(preview_faces, smooth_boundary=smooth_boundary)
        if tri_coords is not None and tri_coords.size > 0:
            tri_shader = _make_uniform_shader()
            self.tri_batch = batch_for_shader(tri_shader, 'TRIS', {"pos": tri_coords})
        else:
            self.tri_batch = None

        if edge_coords is not None and edge_coords.size > 0:
            self.edge_batch = batch_for_shader(poly_shader, 'LINES', {"pos": edge_coords})
        else:
            self.edge_batch = None

    def update_ring_radius(self, radius: float):
        """Ultra-fast 3D radius ring rebuild for real-time interactive resize (zero mesh overhead)."""
        self.radius = float(radius)
        if not self.has_hit:
            return
        n = self.hit_normal_world
        ref = Vector((0.0, 0.0, 1.0)) if abs(n.z) < 0.9 else Vector((1.0, 0.0, 0.0))
        t1 = n.cross(ref).normalized()
        t2 = n.cross(t1).normalized()
        segments = 48
        r = max(float(radius), 1e-5)
        offset = n * max(r * 0.0025, 1e-4)
        center = self.hit_pos_world + offset
        ring_pts = []
        step = (2.0 * math.pi) / segments
        for i in range(segments + 1):
            a = i * step
            ring_pts.append(center + (t1 * math.cos(a) + t2 * math.sin(a)) * r)

        poly_shader = _make_polyline_shader()
        self.ring_batch = batch_for_shader(poly_shader, 'LINE_STRIP', {"pos": ring_pts})


def draw_callback_view(overlay: FloodOverlayState, context):
    """Draw 3D translucent flood preview, crisp boundary outline, and brush ring."""
    if context.area != overlay.area or not overlay.has_hit:
        return

    rv3d = context.region_data
    eps = max(overlay.radius * 0.002, 1e-4)
    view_nudge = (rv3d.view_rotation @ Vector((0.0, 0.0, 1.0))) * eps if rv3d is not None else Vector((0.0, 0.0, 0.0))
    vp_size = gpu.state.viewport_get()[2:]

    prev_blend = gpu.state.blend_get()
    prev_depth_test = gpu.state.depth_test_get()
    prev_depth_mask = gpu.state.depth_mask_get()

    try:
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(False)

        r, g, b, _a = overlay.fill_rgba

        if overlay.matrix_world is not None and (overlay.tri_batch is not None or overlay.edge_batch is not None):
            with gpu.matrix.push_pop():
                gpu.matrix.translate(view_nudge)
                gpu.matrix.multiply_matrix(overlay.matrix_world)

                if overlay.tri_batch is not None and not overlay.is_painting:
                    tri_shader = _make_uniform_shader()
                    tri_shader.bind()
                    fill_col = (1.0, 0.25, 0.25, 0.20) if overlay.protected_hit else (r, g, b, 0.32)
                    tri_shader.uniform_float("color", fill_col)
                    overlay.tri_batch.draw(tri_shader)

                if overlay.edge_batch is not None:
                    edge_shader = _make_polyline_shader()
                    edge_shader.bind()
                    edge_shader.uniform_float("viewportSize", vp_size)
                    edge_shader.uniform_float("lineWidth", 2.2)
                    outline_col = (1.0, 0.35, 0.35, 0.95) if overlay.protected_hit else (1.0, 0.95, 0.35, 0.95)
                    edge_shader.uniform_float("color", outline_col)
                    overlay.edge_batch.draw(edge_shader)

        if overlay.ring_batch is not None:
            with gpu.matrix.push_pop():
                gpu.matrix.translate(view_nudge)
                ring_shader = _make_polyline_shader()
                ring_shader.bind()
                ring_shader.uniform_float("viewportSize", vp_size)
                ring_shader.uniform_float("lineWidth", 2.2 if not overlay.is_resizing else 3.2)
                if overlay.protected_hit:
                    ring_col = (1.0, 0.3, 0.3, 0.95)
                elif overlay.is_resizing:
                    ring_col = (0.35, 0.85, 1.0, 0.95)
                else:
                    ring_col = (1.0, 1.0, 1.0, 0.90)
                ring_shader.uniform_float("color", ring_col)
                overlay.ring_batch.draw(ring_shader)
    finally:
        gpu.state.depth_mask_set(prev_depth_mask)
        gpu.state.depth_test_set(prev_depth_test)
        gpu.state.blend_set(prev_blend)


def _draw_rect_2d(x: float, y: float, w: float, h: float, color: tuple[float, float, float, float]):
    verts = (
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h),
    )
    indices = ((0, 1, 2), (2, 3, 0))
    shader = _make_uniform_shader()
    batch = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=indices)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def draw_callback_pixel(overlay: FloodOverlayState, context):
    """Draw compact, theme-independent BLF HUD in the bottom-left corner of the 3D Viewport."""
    if context.area != overlay.area:
        return

    ui_scale = max(context.preferences.system.ui_scale, 0.75)
    pad_x = 14.0 * ui_scale
    pad_y = 12.0 * ui_scale
    margin_x = 22.0 * ui_scale
    margin_y = 26.0 * ui_scale

    font_id = 0
    title_size = 14.0 * ui_scale
    body_size = 12.5 * ui_scale
    hint_size = 11.0 * ui_scale

    # Build HUD lines cleanly.
    preserve_str = "ON" if overlay.preserve_existing else "OFF"
    mode_suffix = "   |   INVERSE (GAP-FILL) [I]" if overlay.invert_flood else ""
    smooth_suffix = "   |   SMOOTH [S]" if overlay.smooth_tool else ""
    resist_pct = int(round(overlay.crease_resistance * 100.0))
    line_title = f"Normal-Aware Flood Brush   |   Target: {overlay.target_label}{mode_suffix}{smooth_suffix}"
    line_params = (
        f"Angle: {overlay.angle_deg:.1f}°   |   "
        f"Radius: {overlay.radius:.3f} m   |   "
        f"Resist: {resist_pct}%   |   "
        f"Preserve [P]: {preserve_str}"
    )

    now = time.monotonic()
    if overlay._toast_text and now < overlay._toast_expires:
        status_line = overlay._toast_text
    elif overlay.protected_hit:
        status_line = "Protected segment — press P to allow overwrite (or I for inverse)"
    elif overlay.is_resizing:
        status_line = f"Resizing Radius: {overlay.radius:.3f} m  (move mouse, click/Enter/F to lock, Esc to cancel)"
    elif overlay.is_painting:
        status_line = f"Painting: {overlay.preview_face_count:,} faces  ({overlay.last_flood_ms:.1f} ms)  |  Undo steps: {overlay.undo_count}"
    elif overlay.has_hit:
        status_line = f"Preview: {overlay.preview_face_count:,} faces  ({overlay.last_flood_ms:.1f} ms)  |  Undo steps: {overlay.undo_count}"
    else:
        status_line = f"Move cursor over mesh to preview / paint  |  Undo steps: {overlay.undo_count}"

    line_hints = (
        "LMB: Paint  |  RMB / Enter / Space: Commit  |  S: Smooth Tool  |  I: Inverse  |  E: Sample Color  |  "
        "X: Flip Swatch  |  C: Cycle Palette  |  F / R: Radius  |  Wheel/[ ]: Angle  |  P: Preserve  |  U: Fill Unpainted  |  Ctrl+Z: Undo"
    )

    # Measure text widths so the backdrop fits neatly.
    blf.size(font_id, title_size)
    w_title, h_title = blf.dimensions(font_id, line_title)
    blf.size(font_id, body_size)
    w_params, h_params = blf.dimensions(font_id, line_params)
    w_status, h_status = blf.dimensions(font_id, status_line)
    blf.size(font_id, hint_size)
    w_hints, h_hints = blf.dimensions(font_id, line_hints)

    swatch_size = h_title + 2.0 * ui_scale
    swatch_gap = 8.0 * ui_scale
    row_gap = 6.0 * ui_scale

    mini_size = max(10.0 * ui_scale, h_title * 0.75)
    mini_gap = 3.0 * ui_scale
    pal_count = len(overlay.palette_colors)
    pal_strip_w = (mini_size + mini_gap) * pal_count if pal_count > 0 else 0.0

    box_w = max(w_title + swatch_size + swatch_gap + (pal_strip_w + 12.0 * ui_scale if pal_count > 0 else 0.0), w_params, w_status, w_hints) + pad_x * 2.0
    box_h = h_title + h_params + h_status + h_hints + row_gap * 3.0 + pad_y * 2.0

    x0 = margin_x
    y0 = margin_y

    prev_blend = gpu.state.blend_get()
    try:
        gpu.state.blend_set('ALPHA')
        _draw_rect_2d(x0, y0, box_w, box_h, (0.06, 0.07, 0.09, 0.84))
        # Accent bar on left edge.
        bar_col = (0.95, 0.32, 0.32, 0.95) if overlay.protected_hit else (0.28, 0.72, 0.98, 0.95)
        _draw_rect_2d(x0, y0, 3.5 * ui_scale, box_h, bar_col)

        # Active fill color swatch next to title, with secondary swatch layered behind (Blender-style dual swatch).
        r, g, b, a = overlay.fill_rgba
        sr, sg, sb, sa = overlay.secondary_rgba
        swatch_x = x0 + pad_x
        swatch_y = y0 + box_h - pad_y - h_title - 1.0 * ui_scale
        sec_offset = 4.0 * ui_scale
        # 1. Background / secondary swatch with border
        _draw_rect_2d(swatch_x + sec_offset - 1.0 * ui_scale, swatch_y - sec_offset - 1.0 * ui_scale, swatch_size + 2.0 * ui_scale, swatch_size + 2.0 * ui_scale, (0.0, 0.0, 0.0, 0.75))
        _draw_rect_2d(swatch_x + sec_offset, swatch_y - sec_offset, swatch_size, swatch_size, (sr, sg, sb, max(sa, 0.35)))
        # 2. Foreground / primary swatch with crisp border
        _draw_rect_2d(swatch_x - 1.0 * ui_scale, swatch_y - 1.0 * ui_scale, swatch_size + 2.0 * ui_scale, swatch_size + 2.0 * ui_scale, (0.0, 0.0, 0.0, 0.85))
        _draw_rect_2d(swatch_x, swatch_y, swatch_size, swatch_size, (r, g, b, max(a, 0.35)))

        # Mini palette swatches strip
        if pal_count > 0:
            px_cur = x0 + pad_x + swatch_size + swatch_gap + w_title + 10.0 * ui_scale
            py_cur = swatch_y + (swatch_size - mini_size) * 0.5
            for i, pc in enumerate(overlay.palette_colors):
                pr, pg, pb = pc
                if i == overlay.palette_active_idx:
                    # Highlight active swatch with border
                    _draw_rect_2d(px_cur - 1.5 * ui_scale, py_cur - 1.5 * ui_scale, mini_size + 3.0 * ui_scale, mini_size + 3.0 * ui_scale, (1.0, 1.0, 1.0, 0.95))
                _draw_rect_2d(px_cur, py_cur, mini_size, mini_size, (pr, pg, pb, 1.0))
                px_cur += mini_size + mini_gap
    finally:
        gpu.state.blend_set(prev_blend)

    blf.enable(font_id, blf.SHADOW)
    blf.shadow(font_id, 5, 0.0, 0.0, 0.0, 0.85)
    blf.shadow_offset(font_id, 1, -1)

    try:
        cur_y = y0 + box_h - pad_y - h_title
        blf.size(font_id, title_size)
        blf.color(font_id, 0.96, 0.98, 1.0, 1.0)
        blf.position(font_id, x0 + pad_x + swatch_size + swatch_gap, cur_y, 0.0)
        blf.draw(font_id, line_title)

        cur_y -= (h_params + row_gap)
        blf.size(font_id, body_size)
        blf.color(font_id, 0.78, 0.92, 1.0, 1.0)
        blf.position(font_id, x0 + pad_x, cur_y, 0.0)
        blf.draw(font_id, line_params)

        cur_y -= (h_status + row_gap)
        if overlay.protected_hit:
            blf.color(font_id, 1.0, 0.45, 0.45, 1.0)
        elif overlay.is_painting:
            blf.color(font_id, 0.55, 1.0, 0.65, 1.0)
        else:
            blf.color(font_id, 0.90, 0.90, 0.92, 1.0)
        blf.position(font_id, x0 + pad_x, cur_y, 0.0)
        blf.draw(font_id, status_line)

        cur_y -= (h_hints + row_gap)
        blf.size(font_id, hint_size)
        blf.color(font_id, 0.65, 0.69, 0.75, 0.95)
        blf.position(font_id, x0 + pad_x, cur_y, 0.0)
        blf.draw(font_id, line_hints)
    finally:
        blf.disable(font_id, blf.SHADOW)
