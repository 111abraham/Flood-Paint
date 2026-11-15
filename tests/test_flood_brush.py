# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import bpy
import numpy as np
from mathutils import Euler, Vector

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import normal_aware_flood_brush
from normal_aware_flood_brush.flood_engine import (
    compute_flood_faces,
    find_unpainted_faces,
    offset_flood_boundary_faces,
    smooth_jagged_borders,
    smooth_jagged_texture_borders,
    smooth_local_texture_paint,
    smooth_local_vertex_paint,
)
from normal_aware_flood_brush.mesh_cache import MeshFloodCache, linear_to_srgb
from normal_aware_flood_brush.operators import VIEW3D_OT_normal_flood_brush
from normal_aware_flood_brush.overlay import FloodOverlayState
from normal_aware_flood_brush.preferences import KEYMAP_SPECS, get_active_shortcut_label
from normal_aware_flood_brush.properties import (
    get_effective_fill_rgba,
    resolve_texture_paint_target,
    set_effective_fill_rgb,
)


def _clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


class TestNormalAwareFloodBrush(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        normal_aware_flood_brush.register()

    @classmethod
    def tearDownClass(cls):
        normal_aware_flood_brush.unregister()

    def setUp(self):
        _clear_scene()

    def test_01_registration_and_properties(self):
        self.assertTrue(hasattr(bpy.types.Scene, "normal_flood_brush"))
        self.assertTrue(hasattr(bpy.ops.view3d, "normal_flood_brush"))
        self.assertTrue(hasattr(bpy.ops.view3d, "flood_smooth_jagged_borders"))
        self.assertTrue(hasattr(bpy.ops.view3d, "flood_setup_shading"))
        self.assertTrue(hasattr(bpy.ops.view3d, "flood_create_paint_image"))
        self.assertTrue(hasattr(bpy.ops.view3d, "flood_reset_canvas"))

        props = bpy.context.scene.normal_flood_brush
        self.assertAlmostEqual(props.normal_angle_threshold, 30.0, places=4)
        self.assertIn(props.target_output, {'COLOR', 'FACE_SET', 'BOTH'})
        self.assertEqual(props.texpaint_target, 'TEXTURE')

        # Verify bare W shortcut is registered in Sculpt, Vertex Paint, Image Paint, and Weight Paint keymaps.
        km_names = {spec[0] for spec in KEYMAP_SPECS}
        self.assertEqual(km_names, {"Sculpt", "Vertex Paint", "Image Paint", "Weight Paint"})
        addon_kc = bpy.context.window_manager.keyconfigs.addon
        for km_name in km_names:
            km = addon_kc.keymaps.get(km_name)
            self.assertIsNotNone(km, f"Missing keymap '{km_name}'")
            matches = [kmi for kmi in km.keymap_items if kmi.idname == "view3d.normal_flood_brush"]
            self.assertEqual(len(matches), 1, f"Expected 1 shortcut in '{km_name}'")
            kmi = matches[0]
            self.assertEqual(kmi.type, 'W')
            self.assertEqual(kmi.value, 'PRESS')
            self.assertFalse(kmi.alt)
            self.assertFalse(kmi.ctrl)
            self.assertFalse(kmi.shift)

        # Verify register/unregister idempotency.
        normal_aware_flood_brush.unregister()
        self.assertFalse(hasattr(bpy.types.Scene, "normal_flood_brush"))
        normal_aware_flood_brush.register()
        self.assertTrue(hasattr(bpy.types.Scene, "normal_flood_brush"))

    def test_02_cube_crease_isolation_and_angle_threshold(self):
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        obj = bpy.context.active_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.subdivide(number_cuts=8)
        bpy.ops.mesh.quads_convert_to_tris()
        bpy.ops.object.mode_set(mode='OBJECT')

        me = obj.data
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cache = MeshFloodCache(obj, depsgraph, 'COLOR', 'CORNER', 'FLOAT_COLOR')

        # Raycast onto the top face (+Z).
        hit_pos, hit_norm, seed_face = cache.raycast_world(Vector((0.0, 0.0, 5.0)), Vector((0.0, 0.0, -1.0)))
        self.assertIsNotNone(seed_face)
        self.assertGreater(float(hit_norm[2]), 0.99)

        # With 30° threshold and hard stop, only the top face (Z > 0.99 normal) should flood.
        filled_30 = compute_flood_faces(
            cache,
            seed_face=seed_face,
            hit_pos=hit_pos,
            radius=10.0,
            angle_deg=30.0,
            crease_resistance=1.0,
        )
        top_faces = np.flatnonzero(cache.face_normals[:, 2] > 0.99)
        self.assertEqual(set(filled_30.tolist()), set(top_faces.tolist()))
        self.assertGreater(filled_30.size, 100)

        # With 95° threshold, flood must cross the 90° edges onto side faces.
        filled_95 = compute_flood_faces(
            cache,
            seed_face=seed_face,
            hit_pos=hit_pos,
            radius=10.0,
            angle_deg=95.0,
            crease_resistance=1.0,
        )
        self.assertEqual(filled_95.size, len(me.polygons))

        # With soft crease resistance (0.5) and 30° threshold, propagation stalls across 90° crease
        # for a medium radius that reaches past the edge on flat regions.
        filled_soft_small = compute_flood_faces(
            cache,
            seed_face=seed_face,
            hit_pos=hit_pos,
            radius=1.3,
            angle_deg=30.0,
            crease_resistance=0.85,
        )
        # Edge is at distance 1.0 from center; crossing 90° with 0.85 resist incurs ~400x penalty.
        self.assertTrue(np.all(cache.face_normals[filled_soft_small, 2] > 0.99))

    def test_03_transformed_non_uniform_object_raycast_and_radius(self):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0)
        obj = bpy.context.active_object
        obj.location = (12.0, -7.5, 4.0)
        obj.rotation_euler = Euler((0.35, -0.5, 0.9), 'XYZ')
        obj.scale = (2.0, 0.5, 1.0)
        bpy.context.view_layer.update()

        depsgraph = bpy.context.evaluated_depsgraph_get()
        cache = MeshFloodCache(obj, depsgraph, 'BOTH', 'CORNER', 'FLOAT_COLOR')

        # Raycast toward object center from above.
        hit_pos, hit_norm, seed_face = cache.raycast_world(Vector((12.0, -7.5, 10.0)), Vector((0.0, 0.0, -1.0)))
        self.assertIsNotNone(seed_face)
        self.assertAlmostEqual(float(np.linalg.norm(hit_norm)), 1.0, places=4)

        filled = compute_flood_faces(
            cache,
            seed_face=seed_face,
            hit_pos=hit_pos,
            radius=0.6,
            angle_deg=45.0,
            crease_resistance=0.9,
        )
        self.assertGreater(filled.size, 10)
        # All flooded centers must lie within world-space radius 0.6 of hit_pos.
        dists = np.linalg.norm(cache.face_centers[filled] - hit_pos, axis=1)
        self.assertTrue(np.all(dists <= 0.6001))

    def test_04_preserve_existing_face_sets_byte_color_and_revert(self):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0)
        obj = bpy.context.active_object
        me = obj.data
        ca = me.color_attributes.new(name="PaintByte", type='BYTE_COLOR', domain='CORNER')
        me.color_attributes.active_color = ca
        me.color_attributes.active = ca
        ca.data.foreach_set("color_srgb", np.ones(len(ca.data) * 4, dtype=np.float32))

        depsgraph = bpy.context.evaluated_depsgraph_get()
        cache = MeshFloodCache(obj, depsgraph, 'BOTH', 'CORNER', 'BYTE_COLOR')
        stroke_ids = np.zeros(cache.num_faces, dtype=np.int32)

        # Stroke 1 on +Z hemisphere (Red, Face Set 2).
        hp1, _, sf1 = cache.raycast_world(Vector((0.0, 0.0, 5.0)), Vector((0.0, 0.0, -1.0)))
        f1 = compute_flood_faces(cache, sf1, hp1, radius=0.8, angle_deg=60.0, crease_resistance=0.5)
        elems1 = cache.faces_to_color_indices(f1)
        cache.apply_color_indices(elems1, (1.0, 0.0, 0.0, 1.0))
        cache.apply_face_set_indices(f1, 2)
        stroke_ids[f1] = 1

        # Stroke 2 nearby with Preserve Existing ON (Blue, Face Set 3) must NOT overwrite Stroke 1.
        hp2, _, sf2 = cache.raycast_world(Vector((0.0, 5.0, 0.0)), Vector((0.0, -1.0, 0.0)))
        f2 = compute_flood_faces(
            cache,
            sf2,
            hp2,
            radius=1.5,
            angle_deg=60.0,
            crease_resistance=0.5,
            preserve_existing=True,
            target_output='BOTH',
            active_stroke_id=2,
            face_stroke_ids=stroke_ids,
        )
        self.assertGreater(f2.size, 0)
        self.assertEqual(len(set(f1.tolist()) & set(f2.tolist())), 0)

        elems2 = cache.faces_to_color_indices(f2)
        cache.apply_color_indices(elems2, (0.0, 0.0, 1.0, 1.0))
        cache.apply_face_set_indices(f2, 3)

        # Check face set values persisted on mesh.
        fs_read = np.empty(cache.num_faces, dtype=np.int32)
        me.attributes[".sculpt_face_set"].data.foreach_get("value", fs_read)
        self.assertTrue(np.all(fs_read[f1] == 2))
        self.assertTrue(np.all(fs_read[f2] == 3))

        # Revert all restores white canvas and removes newly created face set attribute.
        cache.revert_all()
        self.assertIsNone(me.attributes.get(".sculpt_face_set"))
        col_read = np.empty(len(ca.data) * 4, dtype=np.float32)
        ca.data.foreach_get("color", col_read)
        self.assertTrue(np.allclose(col_read, 1.0, atol=1e-2))

    def test_05_smooth_jagged_borders_and_ply_export(self):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0)
        obj = bpy.context.active_object
        me = obj.data

        ca = me.color_attributes.new(name="Color", type='FLOAT_COLOR', domain='CORNER')
        me.color_attributes.active_color = ca
        me.color_attributes.active = ca

        centers = np.empty(len(me.polygons) * 3, dtype=np.float32)
        me.polygons.foreach_get("center", centers)
        centers = centers.reshape((-1, 3))
        top_mask = centers[:, 2] > 0.0

        loop_colors = np.zeros((len(me.loops), 4), dtype=np.float32)
        loop_colors[:, 3] = 1.0
        face_col = np.where(top_mask[:, None], np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32), np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32))
        loop_colors[:] = np.repeat(face_col, 3, axis=0)
        ca.data.foreach_set("color", loop_colors.ravel())
        me.update()

        bpy.context.scene.normal_flood_brush.smooth_iterations = 2
        bpy.context.scene.normal_flood_brush.smooth_strength = 0.75
        res = bpy.ops.view3d.flood_smooth_jagged_borders()
        self.assertEqual(res, {'FINISHED'})

        out = np.empty(len(me.loops) * 4, dtype=np.float32)
        ca.data.foreach_get("color", out)
        out = out.reshape((-1, 4))

        # Verify boundary corners were blended while polar interior triangles remained pure.
        blended = (out[:, 0] > 0.05) & (out[:, 0] < 0.95)
        self.assertGreater(int(np.count_nonzero(blended)), 10)
        north_pole = centers[:, 2] > 0.85
        north_loops = np.repeat(north_pole, 3)
        self.assertTrue(np.allclose(out[north_loops, :3], [1.0, 0.0, 0.0], atol=1e-5))

        # Export PLY and confirm RGB bytes are present and valid.
        with tempfile.TemporaryDirectory() as tmp:
            ply_path = os.path.join(tmp, "colored.ply")
            bpy.ops.wm.ply_export(filepath=ply_path, ascii_format=True, export_colors='SRGB')
            with open(ply_path, "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            end_idx = lines.index("end_header")
            first_vert_tokens = lines[end_idx + 1].split()
            # Format: x y z r g b a [nx ny nz] [s t]
            rgb_vals = [int(first_vert_tokens[3]), int(first_vert_tokens[4]), int(first_vert_tokens[5])]
            self.assertGreater(sum(rgb_vals), 0)

    def test_06_brush_color_sync(self):
        bpy.ops.mesh.primitive_cube_add()
        bpy.ops.object.mode_set(mode='VERTEX_PAINT')
        set_effective_fill_rgb(bpy.context, (0.2, 0.8, 0.4))
        rgba = get_effective_fill_rgba(bpy.context)
        self.assertAlmostEqual(rgba[0], 0.2, places=4)
        self.assertAlmostEqual(rgba[1], 0.8, places=4)
        self.assertAlmostEqual(rgba[2], 0.4, places=4)
        bpy.ops.object.mode_set(mode='OBJECT')

    def test_07_one_million_plus_triangle_benchmark(self):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=9, radius=1.0)
        obj = bpy.context.active_object
        me = obj.data
        nf = len(me.polygons)
        self.assertGreaterEqual(nf, 1_000_000)

        t_init = time.perf_counter()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cache = MeshFloodCache(obj, depsgraph, 'COLOR', 'CORNER', 'FLOAT_COLOR')
        cache_time = time.perf_counter() - t_init

        hp, hn, sf = cache.raycast_world(Vector((0.0, 0.0, 5.0)), Vector((0.0, 0.0, -1.0)))
        self.assertIsNotNone(sf)

        t_flood = time.perf_counter()
        filled = compute_flood_faces(
            cache,
            seed_face=sf,
            hit_pos=hp,
            radius=0.35,
            angle_deg=30.0,
            crease_resistance=0.95,
        )
        flood_ms = (time.perf_counter() - t_flood) * 1000.0

        t_write = time.perf_counter()
        elems = cache.faces_to_color_indices(filled)
        cache.apply_color_indices(elems, (0.1, 0.7, 0.9, 1.0))
        me.update_gpu_tag()
        me.update()
        write_ms = (time.perf_counter() - t_write) * 1000.0

        t_preview = time.perf_counter()
        tri_coords, edge_coords = cache.extract_preview_geometry(filled)
        preview_ms = (time.perf_counter() - t_preview) * 1000.0

        t_smooth = time.perf_counter()
        smoothed = smooth_jagged_borders(me, iterations=2, strength=0.75)
        smooth_ms = (time.perf_counter() - t_smooth) * 1000.0

        print(
            f"\n[1M+ BENCHMARK] faces={nf:,} | cache={cache_time:.3f}s | "
            f"flood({filled.size:,} faces)={flood_ms:.2f}ms | write={write_ms:.2f}ms | "
            f"preview={preview_ms:.2f}ms | smooth({smoothed:,} elems)={smooth_ms:.2f}ms"
        )
        self.assertGreater(filled.size, 10_000)
        self.assertIsNotNone(tri_coords)
        self.assertIsNotNone(edge_coords)
        self.assertLess(flood_ms, 80.0)
        self.assertLess(write_ms, 50.0)

    def test_08_quad_ngon_and_point_domain_support(self):
        # Create a mesh with mixed quads and an n-gon, plus POINT domain BYTE_COLOR.
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        obj = bpy.context.active_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.subdivide(number_cuts=3)
        bpy.ops.object.mode_set(mode='OBJECT')

        me = obj.data
        # Subdivide one edge in bmesh to force an n-gon.
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.subdivide_edges(bm, edges=[list(bm.edges)[0]], cuts=1)
        bm.to_mesh(me)
        bm.free()

        ca = me.color_attributes.new(name="PointCol", type='BYTE_COLOR', domain='POINT')
        me.color_attributes.active_color = ca
        me.color_attributes.active = ca
        ca.data.foreach_set("color_srgb", np.ones(len(ca.data) * 4, dtype=np.float32))

        depsgraph = bpy.context.evaluated_depsgraph_get()
        cache = MeshFloodCache(obj, depsgraph, 'COLOR', 'POINT', 'BYTE_COLOR')
        self.assertFalse(cache.all_tris)

        hp, _, sf = cache.raycast_world(Vector((0.0, 0.0, 5.0)), Vector((0.0, 0.0, -1.0)))
        self.assertIsNotNone(sf)
        filled = compute_flood_faces(cache, sf, hp, radius=5.0, angle_deg=30.0, crease_resistance=1.0)
        self.assertGreater(filled.size, 8)
        elems = cache.faces_to_color_indices(filled)
        cache.apply_color_indices(elems, (0.2, 0.9, 0.3, 1.0))

        tri_coords, edge_coords = cache.extract_preview_geometry(filled)
        self.assertIsNotNone(tri_coords)
        self.assertIsNotNone(edge_coords)

        smoothed = smooth_jagged_borders(me, iterations=2, strength=0.75)
        self.assertGreater(smoothed, 0)

    def test_09_reset_canvas_and_sculpt_mode_poll(self):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2)
        obj = bpy.context.active_object
        props = bpy.context.scene.normal_flood_brush
        props.target_output = 'BOTH'

        res = bpy.ops.view3d.flood_reset_canvas()
        self.assertEqual(res, {'FINISHED'})
        me = obj.data
        self.assertIsNotNone(me.color_attributes.active)
        self.assertIsNotNone(me.color_attributes.active_color)
        self.assertIsNotNone(me.attributes.get(".sculpt_face_set"))

        # Check Sculpt Mode compatibility & smoothing poll.
        bpy.ops.object.mode_set(mode='SCULPT')
        self.assertTrue(bpy.ops.view3d.flood_smooth_jagged_borders.poll())
        bpy.ops.object.mode_set(mode='OBJECT')

    def test_10_texture_paint_flood_preserve_undo_and_smooth(self):
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        obj = bpy.context.active_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.subdivide(number_cuts=4)
        bpy.ops.object.mode_set(mode='OBJECT')

        me = obj.data
        self.assertGreater(len(me.uv_layers), 0)

        # Enter Texture Paint mode & create a 1024x1024 white Paint Texture.
        bpy.ops.object.mode_set(mode='TEXTURE_PAINT')
        bpy.context.scene.normal_flood_brush.texpaint_resolution = '1024'
        res_create = bpy.ops.view3d.flood_create_paint_image()
        self.assertEqual(res_create, {'FINISHED'})
        self.assertIn("W", get_active_shortcut_label(bpy.context))

        # Test Brush.color sync inside PAINT_TEXTURE.
        set_effective_fill_rgb(bpy.context, (0.9, 0.1, 0.1))
        rgba_tex = get_effective_fill_rgba(bpy.context)
        self.assertAlmostEqual(rgba_tex[0], 0.9, places=4)
        self.assertAlmostEqual(rgba_tex[1], 0.1, places=4)

        img, uv_layer, restrict_mat = resolve_texture_paint_target(bpy.context, obj)
        self.assertIsNotNone(img)
        self.assertIsNotNone(uv_layer)
        self.assertEqual(img.size[0], 1024)
        self.assertTrue(bpy.ops.view3d.flood_smooth_jagged_borders.poll())

        depsgraph = bpy.context.evaluated_depsgraph_get()
        cache = MeshFloodCache(
            obj=obj,
            depsgraph=depsgraph,
            target_output='TEXTURE',
            default_domain='CORNER',
            default_type='FLOAT_COLOR',
            tex_image=img,
            uv_layer=uv_layer,
            tex_seam_bleed=2,
            restrict_mat_index=restrict_mat,
        )
        self.assertFalse(cache.all_tris)  # Cube polygons are quads

        # Flood top face (+Z) and paint Red onto UV texture.
        hp_top, _, sf_top = cache.raycast_world(Vector((0.0, 0.0, 5.0)), Vector((0.0, 0.0, -1.0)))
        self.assertIsNotNone(sf_top)
        f_top = compute_flood_faces(
            cache, sf_top, hp_top, radius=10.0, angle_deg=30.0, crease_resistance=1.0, target_output='TEXTURE'
        )
        self.assertEqual(f_top.size, 25)
        stroke_mask = np.zeros(cache.tex_width * cache.tex_height, dtype=bool)
        pix_red, old_red = cache.apply_texture_faces(f_top, (1.0, 0.0, 0.0, 1.0), stroke_tex_mask=stroke_mask)
        self.assertGreater(pix_red.size, 1000)
        cache.tex_protected_mask.ravel()[pix_red] = True

        # Check top face UV centroid samples linear Red, side face samples White.
        face_tex = cache.get_face_tex_colors_linear()
        self.assertTrue(np.allclose(face_tex[sf_top, :3], [1.0, 0.0, 0.0], atol=0.02))
        hp_side, _, sf_side = cache.raycast_world(Vector((5.0, 0.0, 0.0)), Vector((-1.0, 0.0, 0.0)))
        self.assertIsNotNone(sf_side)
        self.assertTrue(np.allclose(face_tex[sf_side, :3], [1.0, 1.0, 1.0], atol=0.02))

        # Preserve Existing: flood across 95° threshold from side face (+X) with preserve_existing=True
        # must stop before entering top red face!
        f_side = compute_flood_faces(
            cache,
            sf_side,
            hp_side,
            radius=10.0,
            angle_deg=95.0,
            crease_resistance=1.0,
            preserve_existing=True,
            target_output='TEXTURE',
        )
        self.assertGreater(f_side.size, 0)
        self.assertEqual(len(set(f_side.tolist()) & set(f_top.tolist())), 0)

        pix_blue, old_blue = cache.apply_texture_faces(
            f_side, (0.0, 0.0, 1.0, 1.0), preserve_existing=True
        )
        self.assertGreater(pix_blue.size, 1000)
        # Pixel undo restores Blue pixels back to white while preserving Red top face.
        cache.restore_texture_pixels(pix_blue, old_blue)
        face_tex_after_undo = cache.get_face_tex_colors_linear()
        self.assertTrue(np.allclose(face_tex_after_undo[sf_side, :3], [1.0, 1.0, 1.0], atol=0.02))
        self.assertTrue(np.allclose(face_tex_after_undo[sf_top, :3], [1.0, 0.0, 0.0], atol=0.02))

        # Paint Blue onto adjacent quad strips via preserve_existing, then test Smooth Jagged Borders operator.
        cache.apply_texture_faces(f_side, (0.0, 0.0, 1.0, 1.0), preserve_existing=True)
        bpy.context.scene.normal_flood_brush.smooth_iterations = 2
        bpy.context.scene.normal_flood_brush.smooth_strength = 0.75
        res_smooth = bpy.ops.view3d.flood_smooth_jagged_borders()
        self.assertEqual(res_smooth, {'FINISHED'})

        # Revert_all restores original all-white texture cleanly.
        cache.revert_all()
        px_chk = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
        img.pixels.foreach_get(px_chk)
        self.assertTrue(np.allclose(px_chk, 1.0, atol=1e-4))

        bpy.ops.object.mode_set(mode='OBJECT')

    def test_11_texture_paint_multimaterial_and_linear_image(self):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0)
        obj = bpy.context.active_object
        me = obj.data

        mat0 = bpy.data.materials.new("MatTop")
        mat1 = bpy.data.materials.new("MatBottom")
        me.materials.append(mat0)
        me.materials.append(mat1)
        centers = np.empty(len(me.polygons) * 3, dtype=np.float32)
        me.polygons.foreach_get("center", centers)
        centers = centers.reshape((-1, 3))
        mat_assign = np.where(centers[:, 2] >= 0.0, 0, 1).astype(np.int32)
        me.polygons.foreach_set("material_index", mat_assign)

        # Linear (Non-Color) paint image test.
        img_lin = bpy.data.images.new("LinPaint", width=512, height=512, alpha=True)
        img_lin.colorspace_settings.name = 'Non-Color'
        img_lin.pixels.foreach_set(np.ones(512 * 512 * 4, dtype=np.float32))
        img_lin.update()

        depsgraph = bpy.context.evaluated_depsgraph_get()
        cache = MeshFloodCache(
            obj=obj,
            depsgraph=depsgraph,
            target_output='TEXTURE',
            default_domain='CORNER',
            default_type='FLOAT_COLOR',
            tex_image=img_lin,
            uv_layer=me.uv_layers.active,
            tex_seam_bleed=2,
            restrict_mat_index=0,
        )
        self.assertFalse(cache.tex_is_srgb)

        hp, _, sf = cache.raycast_world(Vector((0.0, 0.0, 5.0)), Vector((0.0, 0.0, -1.0)))
        filled = compute_flood_faces(
            cache, sf, hp, radius=10.0, angle_deg=180.0, crease_resistance=0.0, target_output='TEXTURE'
        )
        # Must cover ONLY material slot 0 even with 180° threshold & radius 10.
        self.assertTrue(np.all(mat_assign[filled] == 0))
        self.assertEqual(filled.size, int(np.count_nonzero(mat_assign == 0)))

        # Painting 0.25 linear stores exact 0.25 in Non-Color image without sRGB gamma encoding.
        pix, _ = cache.apply_texture_faces(filled, (0.25, 0.5, 0.75, 1.0))
        self.assertGreater(pix.size, 100)
        self.assertTrue(np.allclose(cache.tex_buf_flat[pix[0], :3], [0.25, 0.5, 0.75], atol=1e-4))

        # Uniformly filling all faces of Material 0 produces 0 internal boundaries (margins do not bleed in).
        mod_uniform = smooth_jagged_texture_borders(img_lin, cache.uv_covered_mask, iterations=1, strength=0.5)
        self.assertEqual(mod_uniform, 0)

        # Flood a smaller contrasting patch inside Material 0 so an internal boundary exists.
        patch = compute_flood_faces(
            cache, sf, hp, radius=0.5, angle_deg=180.0, crease_resistance=0.0, target_output='TEXTURE'
        )
        cache.apply_texture_faces(patch, (0.9, 0.1, 0.1, 1.0))
        mod_px = smooth_jagged_texture_borders(img_lin, cache.uv_covered_mask, iterations=1, strength=0.5)
        self.assertGreater(mod_px, 0)

    def test_12_texture_paint_one_million_plus_benchmark(self):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=9, radius=1.0)
        obj = bpy.context.active_object
        me = obj.data
        nf = len(me.polygons)
        self.assertGreaterEqual(nf, 1_000_000)

        img = bpy.data.images.new("Bench2K", width=2048, height=2048, alpha=True)
        img.pixels.foreach_set(np.ones(2048 * 2048 * 4, dtype=np.float32))
        img.update()

        t_init = time.perf_counter()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cache = MeshFloodCache(
            obj=obj,
            depsgraph=depsgraph,
            target_output='TEXTURE',
            default_domain='CORNER',
            default_type='FLOAT_COLOR',
            tex_image=img,
            uv_layer=me.uv_layers.active,
            tex_seam_bleed=2,
        )
        cache_s = time.perf_counter() - t_init

        hp, _, sf = cache.raycast_world(Vector((0.0, 0.0, 5.0)), Vector((0.0, 0.0, -1.0)))
        t_flood = time.perf_counter()
        filled = compute_flood_faces(
            cache, sf, hp, radius=0.35, angle_deg=30.0, crease_resistance=0.95, target_output='TEXTURE'
        )
        flood_ms = (time.perf_counter() - t_flood) * 1000.0

        t_write = time.perf_counter()
        pix, _ = cache.apply_texture_faces(filled, (0.9, 0.3, 0.1, 1.0))
        write_ms = (time.perf_counter() - t_write) * 1000.0

        t_smooth = time.perf_counter()
        smoothed = smooth_jagged_texture_borders(img, cache.uv_covered_mask, iterations=2, strength=0.75)
        smooth_ms = (time.perf_counter() - t_smooth) * 1000.0

        print(
            f"\n[1M+ TEXPAINT BENCHMARK] faces={nf:,} | tex=2048x2048 | cache={cache_s:.3f}s | "
            f"flood({filled.size:,} faces)={flood_ms:.2f}ms | uv_raster+upload({pix.size:,} px)={write_ms:.2f}ms | "
            f"tex_smooth({smoothed:,} px)={smooth_ms:.2f}ms"
        )
        self.assertGreater(filled.size, 10_000)
        self.assertGreater(pix.size, 1_000)

    def test_13_inverse_flood_cache_and_fill_unpainted(self):
        from normal_aware_flood_brush.mesh_cache import clear_geometry_cache

        clear_geometry_cache()
        # Subdivided cube: paint +X and -X ends, leaving the middle ring/band unreachable across 90° edges.
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        obj = bpy.context.active_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.subdivide(number_cuts=4)
        bpy.ops.object.mode_set(mode='OBJECT')
        me = obj.data

        bpy.ops.object.mode_set(mode='VERTEX_PAINT')
        bpy.ops.view3d.flood_reset_canvas()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cache = MeshFloodCache(obj, depsgraph, 'COLOR', 'CORNER', 'FLOAT_COLOR')
        stroke_ids = np.zeros(cache.num_faces, dtype=np.int32)

        # 1. Paint +X cap Red (30° stops strictly at +X face).
        hp_px, _, sf_px = cache.raycast_world(Vector((5.0, 0.0, 0.0)), Vector((-1.0, 0.0, 0.0)))
        f_px = compute_flood_faces(cache, sf_px, hp_px, radius=10.0, angle_deg=30.0, crease_resistance=1.0)
        cache.apply_color_indices(cache.faces_to_color_indices(f_px), (1.0, 0.0, 0.0, 1.0))
        stroke_ids[f_px] = 1

        # 2. Paint -X cap Blue.
        hp_nx, _, sf_nx = cache.raycast_world(Vector((-5.0, 0.0, 0.0)), Vector((1.0, 0.0, 0.0)))
        f_nx = compute_flood_faces(cache, sf_nx, hp_nx, radius=10.0, angle_deg=30.0, crease_resistance=1.0)
        cache.apply_color_indices(cache.faces_to_color_indices(f_nx), (0.0, 0.0, 1.0, 1.0))
        stroke_ids[f_nx] = 2

        # 3a. Inverse Flood [I] ALONE (preserve_existing=False!) has implicit Gap-Fill boundary protection:
        #     hitting painted Red +X cap edge selects ONLY unpainted middle faces while protecting +X/-X.
        hp_px_edge, _, sf_px_edge = cache.raycast_world(Vector((5.0, 0.0, 0.85)), Vector((-1.0, 0.0, 0.0)))
        f_inv_from_painted = compute_flood_faces(
            cache,
            sf_px_edge,
            hp_px_edge,
            radius=1.2,
            angle_deg=30.0,
            crease_resistance=1.0,
            preserve_existing=False,
            active_stroke_id=3,
            face_stroke_ids=stroke_ids,
            invert_flood=True,
        )
        self.assertGreater(f_inv_from_painted.size, 0)
        self.assertEqual(len(set(f_inv_from_painted.tolist()) & set(f_px.tolist())), 0)
        self.assertEqual(len(set(f_inv_from_painted.tolist()) & set(f_nx.tolist())), 0)

        # 3b. Cursor hovering directly in unpainted middle gap (+Z between +X Red and -X Blue) with I alone:
        #     selects ALL unpainted middle faces inside cursor radius (including seed face!) while protecting +X/-X.
        hp_top, _, sf_top = cache.raycast_world(Vector((0.8, 0.0, 5.0)), Vector((0.0, 0.0, -1.0)))
        f_mid_with_color_boundary = compute_flood_faces(
            cache,
            sf_top,
            hp_top,
            radius=1.2,
            angle_deg=30.0,
            crease_resistance=1.0,
            preserve_existing=False,
            face_stroke_ids=None,  # Cross-session test: detects Red/Blue painted colors even with no stroke_ids!
            invert_flood=True,
        )
        self.assertIn(int(sf_top), set(f_mid_with_color_boundary.tolist()))
        self.assertEqual(len(set(f_mid_with_color_boundary.tolist()) & set(f_px.tolist())), 0)
        self.assertEqual(len(set(f_mid_with_color_boundary.tolist()) & set(f_nx.tolist())), 0)

        # 3c. Session geometry cache reuse & exact vertex-move invalidation (before creating cube2).
        cache_hit = MeshFloodCache(obj, depsgraph, 'COLOR', 'CORNER', 'FLOAT_COLOR')
        self.assertIs(cache_hit.adj_offsets, cache.adj_offsets)
        old_co = me.vertices[0].co.copy()
        me.vertices[0].co.x += 0.25
        me.update()
        cache_miss = MeshFloodCache(obj, depsgraph, 'COLOR', 'CORNER', 'FLOAT_COLOR')
        self.assertIsNot(cache_miss.adj_offsets, cache.adj_offsets)
        me.vertices[0].co = old_co
        me.update()

        # 4. Blank region fallback (zero painted boundaries inside radius): inverts normal flood across crease.
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(10.0, 0.0, 0.0))
        cube2 = bpy.context.active_object
        bpy.ops.object.mode_set(mode='VERTEX_PAINT')
        bpy.ops.view3d.flood_reset_canvas()
        cache2 = MeshFloodCache(cube2, bpy.context.evaluated_depsgraph_get(), 'COLOR', 'CORNER', 'FLOAT_COLOR')
        hp2, _, sf2 = cache2.raycast_world(Vector((10.0, 0.0, 5.0)), Vector((0.0, 0.0, -1.0)))
        f_norm2 = compute_flood_faces(cache2, sf2, hp2, radius=3.0, angle_deg=30.0, crease_resistance=1.0, invert_flood=False)
        f_rev2 = compute_flood_faces(cache2, sf2, hp2, radius=3.0, angle_deg=30.0, crease_resistance=1.0, invert_flood=True)
        self.assertGreater(f_rev2.size, 0)
        self.assertEqual(len(set(f_norm2.tolist()) & set(f_rev2.tolist())), 0)
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.data.objects.remove(cube2, do_unlink=True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='VERTEX_PAINT')

        # 5. Standalone Fill Unpainted [U] & deferred commit in Texture Paint mode.
        bpy.ops.object.mode_set(mode='TEXTURE_PAINT')
        bpy.context.scene.normal_flood_brush.texpaint_resolution = '1024'
        self.assertEqual(bpy.ops.view3d.flood_create_paint_image(), {'FINISHED'})
        img, uv_layer, _ = resolve_texture_paint_target(bpy.context, obj)
        tcache = MeshFloodCache(
            obj, depsgraph, 'TEXTURE', 'CORNER', 'FLOAT_COLOR',
            tex_image=img, uv_layer=uv_layer, store_initial_tex=False,
        )
        tcache.apply_texture_faces(f_px, (1.0, 0.0, 0.0, 1.0), commit=False)
        # Buffered on CPU only until push_texture_buffer() is called; image pixels still white before push.
        px_before = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
        img.pixels.foreach_get(px_before)
        self.assertTrue(np.allclose(px_before, 1.0, atol=1e-4))
        tcache.push_texture_buffer()
        # Verify store_initial_tex=False revert_all() restores white canvas accurately after push.
        tcache.revert_all()
        px_chk = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
        img.pixels.foreach_get(px_chk)
        self.assertTrue(np.allclose(px_chk, 1.0, atol=1e-4))

        # Re-paint +X cap Red, then invoke view3d.flood_fill_unpainted to fill remaining faces Green.
        tcache.apply_texture_faces(f_px, (1.0, 0.0, 0.0, 1.0))
        set_effective_fill_rgb(bpy.context, (0.0, 0.85, 0.1))
        unp = find_unpainted_faces(tcache, target_output='TEXTURE')
        self.assertEqual(unp.size, cache.num_faces - f_px.size)
        self.assertEqual(bpy.ops.view3d.flood_fill_unpainted(), {'FINISHED'})
        tcache2 = MeshFloodCache(
            obj, depsgraph, 'TEXTURE', 'CORNER', 'FLOAT_COLOR',
            tex_image=img, uv_layer=uv_layer, store_initial_tex=False,
        )
        self.assertEqual(find_unpainted_faces(tcache2, target_output='TEXTURE').size, 0)
        clear_geometry_cache()
        bpy.ops.object.mode_set(mode='OBJECT')

    def test_14_ui_guard_and_palette_color_pick(self):
        from types import SimpleNamespace
        from normal_aware_flood_brush.operators import VIEW3D_OT_normal_flood_brush
        from normal_aware_flood_brush.overlay import FloodOverlayState

        class MockRegion:
            def __init__(self, rtype, x, y, w, h):
                self.type = rtype
                self.x = x
                self.y = y
                self.width = w
                self.height = h

        class MockArea:
            def __init__(self, x, y, w, h, regions):
                self.x = x
                self.y = y
                self.width = w
                self.height = h
                self.regions = regions

        class MockOp:
            def __init__(self, area):
                self._area = area
                self._window_reg = None
                self._rv3d = None
                self._cache = None
                self._overlay = None
                self._target_mode = 'COLOR'

            def _is_over_ui(self, mouse_x, mouse_y):
                return VIEW3D_OT_normal_flood_brush._is_over_ui(self, mouse_x, mouse_y)

            def _raycast_mouse(self, mouse_x, mouse_y):
                return VIEW3D_OT_normal_flood_brush._raycast_mouse(self, mouse_x, mouse_y)

            def _cycle_palette_color(self, context):
                return VIEW3D_OT_normal_flood_brush._cycle_palette_color(self, context)

            def _handle_color_pick(self, context, event):
                return VIEW3D_OT_normal_flood_brush._handle_color_pick(self, context, event)

            def _sync_overlay_props(self, context):
                pass

            def _redraw_viewport(self, context):
                pass

            def _sample_screen_pixel_linear(self, mouse_x, mouse_y):
                return None

        # 1. UI guard test: mouse coordinates over non-WINDOW regions (sidebar N-panel, toolbar, header)
        mock_regions = [
            MockRegion('WINDOW', 0, 0, 1000, 800),
            MockRegion('UI', 700, 50, 300, 750),
            MockRegion('TOOLS', 0, 50, 80, 750),
            MockRegion('HEADER', 0, 0, 1000, 50),
        ]
        mock_area = MockArea(0, 0, 1000, 800, mock_regions)
        mock_op = MockOp(mock_area)

        # Over UI sidebar (x=750, y=400) -> True
        self.assertTrue(mock_op._is_over_ui(750, 400))
        # Over Tools sidebar (x=40, y=400) -> True
        self.assertTrue(mock_op._is_over_ui(40, 400))
        # Over Header (x=500, y=25) -> True
        self.assertTrue(mock_op._is_over_ui(500, 25))
        # Outside area entirely (x=1100, y=400) -> True
        self.assertTrue(mock_op._is_over_ui(1100, 400))
        # In 3D canvas viewport (x=400, y=400) -> False
        self.assertFalse(mock_op._is_over_ui(400, 400))

        # 2. _raycast_mouse UI guard: when over UI, early returns (None, None, None)
        mock_op._window_reg = mock_regions[0]
        self.assertEqual(mock_op._raycast_mouse(750, 400), (None, None, None))
        self.assertEqual(mock_op._raycast_mouse(40, 400), (None, None, None))

        # 3. Palette cycling [C] with and without palette
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        obj = bpy.context.active_object
        bpy.ops.object.mode_set(mode='VERTEX_PAINT')

        mock_overlay = FloodOverlayState(mock_area)
        mock_op._overlay = mock_overlay

        ps = bpy.context.tool_settings.vertex_paint
        ps.palette = None
        mock_op._cycle_palette_color(bpy.context)
        self.assertIn("No active palette", mock_overlay._toast_text)

        # Create palette with 3 colors: Red, Green, Blue
        pal = bpy.data.palettes.new("TestCyclePalette")
        ps.palette = pal
        c1 = pal.colors.new()
        c1.color = (1.0, 0.0, 0.0)
        c2 = pal.colors.new()
        c2.color = (0.0, 1.0, 0.0)
        c3 = pal.colors.new()
        c3.color = (0.0, 0.0, 1.0)
        pal.colors.active = c1

        # Cycle once: advances to Green (c2)
        mock_op._cycle_palette_color(bpy.context)
        self.assertEqual(pal.colors.active, c2)
        r, g, b, _ = get_effective_fill_rgba(bpy.context)
        self.assertAlmostEqual(g, 1.0, places=3)
        self.assertIn("Palette [2/3]", mock_overlay._toast_text)

        # Cycle again: advances to Blue (c3)
        mock_op._cycle_palette_color(bpy.context)
        self.assertEqual(pal.colors.active, c3)
        r, g, b, _ = get_effective_fill_rgba(bpy.context)
        self.assertAlmostEqual(b, 1.0, places=3)

        # Cycle again: wraps around to Red (c1)
        mock_op._cycle_palette_color(bpy.context)
        self.assertEqual(pal.colors.active, c1)
        r, g, b, _ = get_effective_fill_rgba(bpy.context)
        self.assertAlmostEqual(r, 1.0, places=3)

        # 4. _handle_color_pick on 3D surface
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cache = MeshFloodCache(obj, depsgraph, 'COLOR', 'CORNER', 'FLOAT_COLOR')
        cache.ensure_color_buffer('CORNER', 'FLOAT_COLOR', create_if_missing=True)
        mock_op._cache = cache

        # Paint face 0 Yellow (1.0, 1.0, 0.0)
        cache.apply_color_indices(cache.faces_to_color_indices(np.array([0], dtype=np.int32)), (1.0, 1.0, 0.0, 1.0))

        mock_op._raycast_mouse = lambda mx, my: (Vector((0, 0, 0)), Vector((0, 0, 1)), 0)
        mock_event = SimpleNamespace(mouse_x=400, mouse_y=400)
        mock_op._handle_color_pick(bpy.context, mock_event)
        r, g, b, _ = get_effective_fill_rgba(bpy.context)
        self.assertAlmostEqual(r, 1.0, places=3)
        self.assertAlmostEqual(g, 1.0, places=3)
        self.assertAlmostEqual(b, 0.0, places=3)
        self.assertIn("Sampled surface", mock_overlay._toast_text)

        # 5. _handle_color_pick matching nearest palette swatch
        # Paint face 1 exact Blue (0.0, 0.0, 1.0) which matches c3 in palette
        cache.apply_color_indices(cache.faces_to_color_indices(np.array([1], dtype=np.int32)), (0.0, 0.0, 1.0, 1.0))
        mock_op._raycast_mouse = lambda mx, my: (Vector((0, 0, 0)), Vector((0, 0, 1)), 1)
        mock_op._handle_color_pick(bpy.context, mock_event)
        self.assertEqual(pal.colors.active, c3)
        self.assertIn("(palette)", mock_overlay._toast_text)

        # 6. _handle_color_pick in empty space (raycast None): cycles palette
        mock_op._raycast_mouse = lambda mx, my: (None, None, None)
        mock_op._sample_screen_pixel_linear = lambda mx, my: None
        mock_op._handle_color_pick(bpy.context, mock_event)
        self.assertEqual(pal.colors.active, c1)

        # Clean up
        bpy.data.palettes.remove(pal)
        bpy.ops.object.mode_set(mode='OBJECT')

    def test_15_middle_mouse_orbit_and_radius_resize(self):
        from types import SimpleNamespace
        from normal_aware_flood_brush.operators import VIEW3D_OT_normal_flood_brush
        from normal_aware_flood_brush.overlay import FloodOverlayState

        class MockRegion:
            def __init__(self, rtype, x, y, w, h):
                self.type = rtype
                self.x = x
                self.y = y
                self.width = w
                self.height = h

        class MockArea:
            def __init__(self, x, y, w, h, regions):
                self.x = x
                self.y = y
                self.width = w
                self.height = h
                self.regions = regions

            def tag_redraw(self):
                pass

        mock_area = MockArea(0, 0, 1000, 800, [MockRegion('WINDOW', 0, 0, 1000, 800)])

        class MockNavOp:
            def __init__(self, area):
                self._area = area
                self._window_reg = area.regions[0]
                self._rv3d = None
                self._cache = None
                self._overlay = FloodOverlayState(area)
                self._target_mode = 'COLOR'
                self._is_painting = False
                self._is_navigating = False
                self._is_resizing = False
                self._resize_locked_toggle = False
                self._resize_anchor_x = 0
                self._resize_start_radius = 0.5
                self._undo_stack = []

            def _is_over_ui(self, mx, my):
                return False

            def _enter_resize(self, context, event, locked_toggle=False):
                return VIEW3D_OT_normal_flood_brush._enter_resize(self, context, event, locked_toggle)

            def _exit_resize(self, context):
                return VIEW3D_OT_normal_flood_brush._exit_resize(self, context)

            def _sync_overlay_props(self, context):
                pass

            def _redraw_viewport(self, context):
                pass

            def _refresh_at_last_hit(self, context):
                pass

            def _update_from_mouse(self, context, mx, my, paint=False, commit_tex=True, **kwargs):
                pass

            def _end_stroke(self):
                pass

            def _handle_smooth_key(self, context, mesh_wide=False):
                pass

            def _cleanup(self, context):
                self._is_navigating = False

            def modal(self, context, event):
                return VIEW3D_OT_normal_flood_brush.modal(self, context, event)

        def make_event(etype, val='PRESS', mx=500, my=500, shift=False, ctrl=False, alt=False):
            return SimpleNamespace(
                type=etype,
                value=val,
                mouse_x=mx,
                mouse_y=my,
                shift=shift,
                ctrl=ctrl,
                alt=alt,
            )

        op = MockNavOp(mock_area)
        props = bpy.context.scene.normal_flood_brush
        props.fill_radius = 0.5

        # 1. Middle Mouse click: passes through to Blender for navigation
        res = op.modal(bpy.context, make_event('MIDDLEMOUSE', 'PRESS'))
        self.assertEqual(res, {'PASS_THROUGH'})

        # 2. Mouse move while in viewport (not painting): returns RUNNING_MODAL, updates preview live, NEVER resizes cursor!
        res = op.modal(bpy.context, make_event('MOUSEMOVE', 'NOTHING', mx=600, my=520))
        self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertAlmostEqual(props.fill_radius, 0.5, places=4)

        # 3. Middle Mouse release: passes through to Blender
        res = op.modal(bpy.context, make_event('MIDDLEMOUSE', 'RELEASE', mx=600, my=520))
        self.assertEqual(res, {'PASS_THROUGH'})
        self.assertAlmostEqual(props.fill_radius, 0.5, places=4)

        # 4. Shift key alone: does NOT enter resize mode (safe for Shift+MMB pan!)
        op.modal(bpy.context, make_event('LEFT_SHIFT', 'PRESS', shift=True))
        self.assertFalse(op._is_resizing)
        op.modal(bpy.context, make_event('RIGHT_SHIFT', 'PRESS', shift=True))
        self.assertFalse(op._is_resizing)

        # 5. Shift + MMB (Pan): passes through without resizing cursor
        res = op.modal(bpy.context, make_event('MIDDLEMOUSE', 'PRESS', shift=True))
        self.assertEqual(res, {'PASS_THROUGH'})
        self.assertFalse(op._is_resizing)
        res = op.modal(bpy.context, make_event('MOUSEMOVE', 'NOTHING', mx=700, my=400, shift=True))
        self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertAlmostEqual(props.fill_radius, 0.5, places=4)
        res = op.modal(bpy.context, make_event('MIDDLEMOUSE', 'RELEASE', mx=700, my=400, shift=True))
        self.assertEqual(res, {'PASS_THROUGH'})

        # 6. Right-Click: commits paint and exits modal!
        res = op.modal(bpy.context, make_event('RIGHTMOUSE', 'PRESS'))
        self.assertEqual(res, {'FINISHED'})

        # 7. E / Alt+S: triggers color sampling; S key triggers boundary smooth
        called_sample = []
        called_smooth = []
        op._handle_color_pick = lambda ctx, ev: (called_sample.append(True) or True)
        op._handle_smooth_key = lambda ctx, mesh_wide=False: (called_smooth.append(mesh_wide) or None)
        res_e = op.modal(bpy.context, make_event('E', 'PRESS'))
        self.assertEqual(res_e, {'RUNNING_MODAL'})
        self.assertEqual(len(called_sample), 1)

        res_alts = op.modal(bpy.context, make_event('S', 'PRESS', alt=True))
        self.assertEqual(res_alts, {'RUNNING_MODAL'})
        self.assertEqual(len(called_sample), 2)

        res_s = op.modal(bpy.context, make_event('S', 'PRESS'))
        self.assertEqual(res_s, {'RUNNING_MODAL'})
        self.assertTrue(props.smooth_tool)

        res_shifts = op.modal(bpy.context, make_event('S', 'PRESS', shift=True))
        self.assertEqual(res_shifts, {'RUNNING_MODAL'})
        self.assertEqual(len(called_smooth), 1)

        # 8. F / R key for dynamic brush resize
        props.fill_radius = 0.5
        res = op.modal(bpy.context, make_event('F', 'PRESS', mx=500, my=500))
        self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertTrue(op._is_resizing)

        # Moving mouse expands radius without heavy Dijkstra
        op.modal(bpy.context, make_event('MOUSEMOVE', 'NOTHING', mx=600, my=500))
        self.assertGreater(props.fill_radius, 0.5)

        # Left-Click confirms and locks the new radius
        res = op.modal(bpy.context, make_event('LEFTMOUSE', 'PRESS', mx=600, my=500))
        self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertFalse(op._is_resizing)
        new_locked_radius = props.fill_radius
        self.assertGreater(new_locked_radius, 0.5)

        # 9. Esc cancels resize and resets brush radius to default (0.25 m)
        props.fill_radius = 0.5
        op.modal(bpy.context, make_event('R', 'PRESS', mx=500, my=500))
        self.assertTrue(op._is_resizing)
        op.modal(bpy.context, make_event('MOUSEMOVE', 'NOTHING', mx=650, my=500))
        self.assertNotEqual(props.fill_radius, 0.5)
        res = op.modal(bpy.context, make_event('ESC', 'PRESS'))
        self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertFalse(op._is_resizing)
        self.assertAlmostEqual(props.fill_radius, 0.25, places=4)

    def test_16_fresh_mesh_color_attr_and_ui_guard(self):
        """Verify MeshFloodCache initializes without AttributeError on fresh mesh with 0 color attributes."""
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        self.assertEqual(len(cube.data.color_attributes), 0)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        # Initializing cache in COLOR mode must auto-create Color attribute without erroring on 'position'
        cache = MeshFloodCache(cube, depsgraph, 'COLOR', 'CORNER', 'BYTE_COLOR')
        self.assertIsNotNone(cache.color_attr)
        self.assertEqual(cache.color_attr.name, "Color")
        self.assertEqual(len(cache.color_buf), 24)

        # Initializing cache in FACE_SET mode on mesh without color attributes must succeed cleanly
        cube2 = bpy.context.active_object
        cache2 = MeshFloodCache(cube2, depsgraph, 'FACE_SET', 'CORNER', 'BYTE_COLOR')
        self.assertIsNotNone(cache2.face_set_attr)

    def test_17_flip_swatch_colors_on_x_key(self):
        """Verify pressing X switches foreground/background swatch color in paint mode and fallback."""
        from normal_aware_flood_brush.properties import (
            flip_brush_colors,
            get_active_brush_color_owner,
            get_effective_fill_rgba,
            get_effective_secondary_rgba,
        )

        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        props = bpy.context.scene.normal_flood_brush
        props.sync_brush_color = True

        # 1. Texture Paint mode: test flipping brush foreground and background
        bpy.ops.paint.texture_paint_toggle()
        owner = get_active_brush_color_owner(bpy.context)
        self.assertIsNotNone(owner)
        owner.color = (1.0, 0.1, 0.2)
        owner.secondary_color = (0.2, 0.9, 0.4)

        c_pri = get_effective_fill_rgba(bpy.context)
        c_sec = get_effective_secondary_rgba(bpy.context)
        self.assertAlmostEqual(c_pri[0], 1.0, places=2)
        self.assertAlmostEqual(c_pri[1], 0.1, places=2)
        self.assertAlmostEqual(c_sec[0], 0.2, places=2)
        self.assertAlmostEqual(c_sec[1], 0.9, places=2)

        # Flip colors via helper function (as X key does)
        flip_brush_colors(bpy.context)
        c_pri_flipped = get_effective_fill_rgba(bpy.context)
        c_sec_flipped = get_effective_secondary_rgba(bpy.context)
        self.assertAlmostEqual(c_pri_flipped[0], 0.2, places=2)
        self.assertAlmostEqual(c_pri_flipped[1], 0.9, places=2)
        self.assertAlmostEqual(c_sec_flipped[0], 1.0, places=2)
        self.assertAlmostEqual(c_sec_flipped[1], 0.1, places=2)

        # 2. Test modal operator X key press event handling
        mock_area = SimpleNamespace(
            x=0, y=0, width=1920, height=1080,
            regions=[SimpleNamespace(type='WINDOW', x=0, y=0, width=1920, height=1080, data=SimpleNamespace())],
            spaces=SimpleNamespace(active=SimpleNamespace(type='VIEW_3D', region_3d=SimpleNamespace())),
            tag_redraw=lambda: None,
        )

        class MockOp:
            def __init__(self, area):
                self._area = area
                self._window_reg = area.regions[0]
                self._rv3d = area.spaces.active.region_3d
                self._is_painting = False
                self._is_resizing = False
                self._is_navigating = False
                self._target_mode = 'TEXTURE'
                self._undo_stack = []
                self._overlay = FloodOverlayState(area)

            def _exit_resize(self, ctx):
                self._is_resizing = False

            def _sync_overlay_props(self, ctx):
                self._overlay.fill_rgba = get_effective_fill_rgba(ctx)
                self._overlay.secondary_rgba = get_effective_secondary_rgba(ctx)

            def _redraw_viewport(self, ctx):
                pass

            def _refresh_at_last_hit(self, ctx):
                pass

            def modal(self, ctx, event):
                return VIEW3D_OT_normal_flood_brush.modal(self, ctx, event)

        op = MockOp(mock_area)
        op._sync_overlay_props(bpy.context)
        before_modal_fill = op._overlay.fill_rgba

        ev_x = SimpleNamespace(type='X', value='PRESS', mouse_x=500, mouse_y=500, ctrl=False, alt=False, shift=False)
        res = op.modal(bpy.context, ev_x)
        self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertNotEqual(op._overlay.fill_rgba[:3], before_modal_fill[:3])
        self.assertAlmostEqual(op._overlay.fill_rgba[0], 1.0, places=2)

        # 3. Fallback properties flip when sync_brush_color is False
        bpy.ops.paint.texture_paint_toggle()  # exit texture paint to object mode
        props.sync_brush_color = False
        props.fill_color = (0.2, 0.4, 0.6, 1.0)
        props.secondary_fill_color = (0.8, 0.1, 0.3, 1.0)
        flip_brush_colors(bpy.context)
        self.assertAlmostEqual(props.fill_color[0], 0.8, places=2)
        self.assertAlmostEqual(props.secondary_fill_color[0], 0.2, places=2)

    def test_18_local_vertex_paint_smooth_on_s_key(self):
        """Verify pressing S in modal W mode performs localized vertex paint boundary smoothing and Ctrl+Z undoes it."""
        # 1. Create a subdivided plane mesh (10x10 cuts = 100 quads, 200 triangles)
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=11, y_subdivisions=11, size=2.0, enter_editmode=False)
        obj = bpy.context.active_object
        self.assertIsNotNone(obj)
        mesh = obj.data

        # 2. Add CORNER Color Attribute
        ca = mesh.color_attributes.new(name="Color", type='FLOAT_COLOR', domain='CORNER')
        mesh.color_attributes.active = ca

        # Paint left half (x < 0) pure red (1, 0, 0, 1) and right half (x >= 0) pure blue (0, 0, 1, 1)
        n_loops = len(mesh.loops)
        cols = np.zeros((n_loops, 4), dtype=np.float32)
        cols[:, 3] = 1.0
        loop_verts = np.empty(n_loops, dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_verts)
        vert_coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", vert_coords)
        vert_coords = vert_coords.reshape((-1, 3))
        left_mask = vert_coords[loop_verts, 0] < 0.0
        cols[left_mask, 0] = 1.0  # Red
        cols[~left_mask, 2] = 1.0  # Blue
        ca.data.foreach_set("color", cols.ravel())
        mesh.update()

        # 3. Build MeshFloodCache
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cache = MeshFloodCache(
            obj=obj,
            depsgraph=depsgraph,
            target_output='COLOR',
            default_domain='CORNER',
            default_type='FLOAT_COLOR',
        )

        # 4. Verify smooth_local_vertex_paint localized around origin (0, 0, 0)
        mod_indices, old_colors = smooth_local_vertex_paint(
            cache=cache,
            hit_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=0.4,
            iterations=2,
            strength=0.75,
        )
        self.assertGreater(mod_indices.size, 0)
        self.assertEqual(len(old_colors), mod_indices.size)

        # Check that modified colors have blended (both red and blue components present)
        smoothed_colors = cache.color_buf[mod_indices]
        self.assertTrue(np.any(smoothed_colors[:, 0] > 0.05))
        self.assertTrue(np.any(smoothed_colors[:, 2] > 0.05))

        # Check that faraway boundary corners outside radius were NOT modified
        faraway_loops = np.flatnonzero(np.abs(vert_coords[loop_verts, 1]) > 0.8)
        self.assertFalse(np.any(np.isin(faraway_loops, mod_indices)))

        # 5. Verify undo via cache.restore_color_indices
        cache.restore_color_indices(mod_indices, old_colors)
        restored_colors = cache.color_buf[mod_indices]
        np.testing.assert_allclose(restored_colors, old_colors, atol=1e-5)

        # 6. Test modal keymap handling in MockOp
        mock_area = SimpleNamespace(
            x=0, y=0, width=1920, height=1080,
            regions=[SimpleNamespace(type='WINDOW', x=0, y=0, width=1920, height=1080, data=SimpleNamespace())],
            spaces=SimpleNamespace(active=SimpleNamespace(type='VIEW_3D', region_3d=SimpleNamespace())),
            tag_redraw=lambda: None,
        )

        class MockModalOp:
            def __init__(self, c, area):
                self._cache = c
                self._target_mode = 'COLOR'
                self._undo_stack = []
                self._face_stroke_ids = np.zeros(c.num_faces, dtype=np.int32)
                self._overlay = FloodOverlayState(area)
                self._last_hit_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                self._last_hit_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                self._last_hit_face = 0
                self._active_stroke = None
                self._is_painting = False
                self._is_resizing = False
                self._area = area
                self._window_reg = area.regions[0]
                self._handle_view = None
                self._handle_pixel = None
                self._had_color_attr = True

            def _exit_resize(self, ctx):
                self._is_resizing = False

            def _end_stroke(self):
                pass

            def _refresh_at_last_hit(self, ctx):
                pass

            def _redraw_viewport(self, ctx):
                pass

            def _sync_overlay_props(self, ctx):
                pass

            def _handle_smooth_key(self, ctx, mesh_wide=False):
                return VIEW3D_OT_normal_flood_brush._handle_smooth_key(self, ctx, mesh_wide)

            def _undo_last_stroke(self, ctx):
                return VIEW3D_OT_normal_flood_brush._undo_last_stroke(self, ctx)

            def _handle_color_pick(self, ctx, event):
                self._overlay.set_toast("Sampled color", duration=1.0)
                return True

            def modal(self, ctx, event):
                return VIEW3D_OT_normal_flood_brush.modal(self, ctx, event)

        op = MockModalOp(cache, mock_area)
        props = bpy.context.scene.normal_flood_brush
        props.fill_radius = 0.4
        props.smooth_iterations = 2
        props.smooth_strength = 0.75

        # Press Shift+S in modal mode -> triggers one-shot boundary smoothing
        ev_shifts = SimpleNamespace(type='S', value='PRESS', mouse_x=500, mouse_y=500, ctrl=False, alt=False, shift=True)
        res_shifts = op.modal(bpy.context, ev_shifts)
        self.assertEqual(res_shifts, {'RUNNING_MODAL'})
        self.assertEqual(len(op._undo_stack), 1)
        self.assertIn("Smoothed", op._overlay._toast_text)

        # Press Ctrl+Z to undo
        ev_undo = SimpleNamespace(type='Z', value='PRESS', mouse_x=500, mouse_y=500, ctrl=True, alt=False, shift=False)
        res_undo = op.modal(bpy.context, ev_undo)
        self.assertEqual(res_undo, {'RUNNING_MODAL'})
        self.assertEqual(len(op._undo_stack), 0)
        self.assertIn("Undid stroke", op._overlay._toast_text)

        # Press S key in modal mode -> toggles Smooth Tool ON/OFF
        ev_s = SimpleNamespace(type='S', value='PRESS', mouse_x=500, mouse_y=500, ctrl=False, alt=False, shift=False)
        res_s = op.modal(bpy.context, ev_s)
        self.assertEqual(res_s, {'RUNNING_MODAL'})
        self.assertTrue(props.smooth_tool)
        self.assertIn("Smooth Tool: ON", op._overlay._toast_text)

        # Press S again -> toggles Smooth Tool OFF
        res_s2 = op.modal(bpy.context, ev_s)
        self.assertEqual(res_s2, {'RUNNING_MODAL'})
        self.assertFalse(props.smooth_tool)
        self.assertIn("Smooth Tool: OFF", op._overlay._toast_text)

        # Press E key for eyedropper sampling
        ev_e = SimpleNamespace(type='E', value='PRESS', mouse_x=500, mouse_y=500, ctrl=False, alt=False, shift=False)
        res_e = op.modal(bpy.context, ev_e)
        self.assertEqual(res_e, {'RUNNING_MODAL'})
        self.assertEqual(op._overlay._toast_text, "Sampled color")

        # Press Alt+S for eyedropper sampling
        ev_alts = SimpleNamespace(type='S', value='PRESS', mouse_x=500, mouse_y=500, ctrl=False, alt=True, shift=False)
        res_alts = op.modal(bpy.context, ev_alts)
        self.assertEqual(res_alts, {'RUNNING_MODAL'})
        self.assertEqual(op._overlay._toast_text, "Sampled color")

    def test_19_smooth_tool_toggle_and_boundary_smoothing(self):
        """Verify Smooth Tool property, boundary line smoothing, and automatic stroke smoothing."""
        # 1. Verify property registration and default
        props = bpy.context.scene.normal_flood_brush
        self.assertFalse(props.smooth_tool)

        # 2. Create quad grid mesh
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=15, y_subdivisions=15, size=2.0)
        obj = bpy.context.active_object
        mesh = obj.data
        ca = mesh.color_attributes.new(name="Color", type='FLOAT_COLOR', domain='CORNER')
        mesh.color_attributes.active = ca

        depsgraph = bpy.context.evaluated_depsgraph_get()
        cache = MeshFloodCache(
            obj=obj,
            depsgraph=depsgraph,
            target_output='COLOR',
            default_domain='CORNER',
            default_type='FLOAT_COLOR',
        )

        # 3. Compute flood faces with smooth_tool=False vs smooth_tool=True
        seed_face = 50
        hit_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        raw_faces = compute_flood_faces(
            cache=cache,
            seed_face=seed_face,
            hit_pos=hit_pos,
            radius=0.5,
            angle_deg=180.0,
            crease_resistance=0.0,
            smooth_tool=False,
        )
        self.assertGreater(raw_faces.size, 0)

        smooth_faces = compute_flood_faces(
            cache=cache,
            seed_face=seed_face,
            hit_pos=hit_pos,
            radius=0.5,
            angle_deg=180.0,
            crease_resistance=0.0,
            smooth_tool=True,
        )
        self.assertGreater(smooth_faces.size, 0)

        # 4. Verify boundary outline smoothing in extract_preview_geometry
        _tri_raw, edge_raw = cache.extract_preview_geometry(raw_faces, smooth_boundary=False)
        _tri_sm, edge_sm = cache.extract_preview_geometry(raw_faces, smooth_boundary=True)

        self.assertIsNotNone(edge_raw)
        self.assertIsNotNone(edge_sm)
        self.assertGreater(len(edge_raw), 0)
        self.assertGreater(len(edge_sm), 0)

        # The smoothed boundary has interpolated/subdivided points to eliminate 90-degree corners
        self.assertGreaterEqual(len(edge_sm), len(edge_raw))

        # 5. Verify Overlay state synchronization
        mock_area = SimpleNamespace(
            x=0, y=0, width=1920, height=1080,
            regions=[SimpleNamespace(type='WINDOW', x=0, y=0, width=1920, height=1080, data=SimpleNamespace())],
            spaces=SimpleNamespace(active=SimpleNamespace(type='VIEW_3D', region_3d=SimpleNamespace())),
            tag_redraw=lambda: None,
        )
        ov = FloodOverlayState(mock_area)
        self.assertFalse(ov.smooth_tool)
        ov.smooth_tool = True
        self.assertTrue(ov.smooth_tool)

    def test_20_weight_paint_mode_session_group_and_undo(self):
        """Verify Weight Paint mode support, auto session vertex group creation, weight assignment, and undo/revert."""
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        obj = bpy.context.active_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.subdivide(number_cuts=4)
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')

        props = bpy.context.scene.normal_flood_brush
        self.assertTrue(props.weight_auto_vertex_group)
        self.assertAlmostEqual(props.weight_value, 1.0)

        # 1. Verify target mode resolution
        self.assertEqual(normal_aware_flood_brush.properties.get_effective_target_mode(bpy.context), 'WEIGHT')

        # 2. Verify cache initialization in WEIGHT mode
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cache = MeshFloodCache(
            obj=obj,
            depsgraph=depsgraph,
            target_output='WEIGHT',
            default_domain='CORNER',
            default_type='FLOAT_COLOR',
        )
        self.assertGreater(cache.num_faces, 0)

        # 3. Compute normal-aware flood faces on one side of cube
        faces = compute_flood_faces(
            cache=cache,
            seed_face=0,
            hit_pos=cache.face_centers[0],
            radius=1.5,
            angle_deg=30.0,
            crease_resistance=0.95,
        )
        self.assertGreater(faces.size, 0)

        # 4. Convert faces to vertex indices
        vert_indices = cache.faces_to_vert_indices(faces)
        self.assertGreater(vert_indices.size, 0)

        # 5. Simulate session vertex group auto-creation
        self.assertEqual(len(obj.vertex_groups), 0)
        vg = obj.vertex_groups.new(name=f"Flood_{obj.name}")
        self.assertEqual(len(obj.vertex_groups), 1)

        # Apply weights
        vg.add(vert_indices.tolist(), 0.85, 'REPLACE')
        for vi in vert_indices:
            self.assertAlmostEqual(vg.weight(int(vi)), 0.85, places=4)

        # 6. Simulate undo: remove vertices
        vg.remove(vert_indices.tolist())
        for vi in vert_indices:
            with self.assertRaises(RuntimeError):
                vg.weight(int(vi))

        # 7. Simulate cancel on ESC: clean deletion of session-created group
        obj.vertex_groups.remove(vg)
        self.assertEqual(len(obj.vertex_groups), 0)

    def test_21_boundary_clipping_smooth_painting_and_resize_commit(self):
        """Verify boundary completeness, exact hit color sampling, smooth painting falloff, and resize commit."""
        # 1. Open boundary plane mesh test
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=5, y_subdivisions=5, size=2.0)
        obj = bpy.context.active_object
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cache = MeshFloodCache(
            obj=obj,
            depsgraph=depsgraph,
            target_output='COLOR',
            default_domain='POINT',
            default_type='FLOAT_COLOR',
        )

        # Flood select corner faces
        sel_faces = np.array([0, 1, 4, 5], dtype=np.int32)
        tri_coords, edge_coords = cache.extract_preview_geometry(sel_faces, smooth_boundary=True)
        self.assertIsNotNone(tri_coords)
        self.assertIsNotNone(edge_coords)
        self.assertGreater(edge_coords.size, 0)

        # 2. Verify incident face coverage computation
        verts, ratios = cache.faces_to_vert_incident_coverage(sel_faces)
        self.assertGreater(verts.size, 0)
        self.assertEqual(verts.size, ratios.size)
        self.assertTrue(np.all((ratios >= 0.0) & (ratios <= 1.0)))
        # In a 2x2 patch on an open grid, corner vertex has different coverage than boundary vertex
        self.assertTrue(np.any(ratios < 1.0))

        # 3. Exact hit color sampling
        col = cache.get_color_at_hit(0, cache.face_centers[0], 'COLOR')
        self.assertIsNotNone(col)
        self.assertEqual(len(col), 4)

        # 4. Weight paint solid hard brush with smooth boundary trimming
        obj.vertex_groups.new(name="TestWeightSmooth")
        vg = obj.vertex_groups.active
        weight_val = 1.0
        apply_verts = verts[ratios >= 0.35]
        self.assertGreater(apply_verts.size, 0)
        vg.add(apply_verts.tolist(), weight_val, 'REPLACE')
        for av in apply_verts:
            self.assertAlmostEqual(vg.weight(int(av)), 1.0, places=4)

        # 4b. Vertex paint CORNER domain smooth trimming (smooth hard brush on mesh faces)
        raw_loops = cache.faces_to_color_indices(sel_faces)
        in_valid_loops = np.isin(cache.loop_verts[raw_loops], apply_verts)
        trimmed_loops = raw_loops[in_valid_loops]
        self.assertGreater(trimmed_loops.size, 0)
        self.assertLess(trimmed_loops.size, raw_loops.size)

        # 5. Texture paint with smooth_boundary=True (100% solid hard brush write & zero interior gaps)
        img = bpy.data.images.new("SmoothTestImg", width=128, height=128, alpha=True)
        img.pixels.foreach_set([1.0] * (128 * 128 * 4))
        uv_layer = obj.data.uv_layers.new(name="UVMap")
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        t_cache = MeshFloodCache(
            obj=obj,
            depsgraph=depsgraph,
            target_output='TEXTURE',
            default_domain='CORNER',
            default_type='FLOAT_COLOR',
            tex_image=img,
            uv_layer=uv_layer,
        )
        fill_col = (0.2, 0.4, 0.8, 1.0)
        raw_idx, _ = t_cache.apply_texture_faces(
            sel_faces,
            fill_col,
            commit=False,
            smooth_boundary=False,
        )
        flat_idx, old_px = t_cache.apply_texture_faces(
            sel_faces,
            fill_col,
            commit=True,
            smooth_boundary=True,
        )
        self.assertGreater(flat_idx.size, 0)
        self.assertEqual(flat_idx.size, old_px.shape[0])
        # Zero unpainted gaps: all original flood pixels are 100% preserved in smoothed texture
        self.assertTrue(np.all(np.isin(raw_idx, flat_idx)))
        # Verify 100% solid write (hard brush, no partial alpha or checkered pixels)
        written_px = t_cache.tex_buf_flat[flat_idx]
        self.assertTrue(np.allclose(written_px[:, 3], 1.0, atol=1e-3))
        exp_rgb = linear_to_srgb(np.asarray(fill_col[:3], dtype=np.float32)) if t_cache.tex_is_srgb else np.asarray(fill_col[:3], dtype=np.float32)
        self.assertTrue(np.allclose(written_px[:, :3], exp_rgb, atol=0.01))
        # Zero variance: completely uniform solid hard brush fill
        self.assertTrue(np.allclose(written_px[:, :3], written_px[0, :3], atol=1e-3))

    def test_22_expand_contract_boundary_and_zero_neighbor_bleed(self):
        """Verify Expand/Contract boundary offset and zero bleeding onto neighboring mesh faces."""
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=20, y_subdivisions=20, size=2.0)
        obj = bpy.context.active_object
        dg = bpy.context.evaluated_depsgraph_get()
        cache = MeshFloodCache(
            obj=obj,
            depsgraph=dg,
            target_output='COLOR',
            default_domain='CORNER',
            default_type='FLOAT_COLOR',
        )

        centers = cache.face_centers
        base_faces = np.flatnonzero(np.linalg.norm(centers, axis=1) < 0.5).astype(np.int32)
        self.assertGreater(base_faces.size, 0)

        # 1. Expand boundary (+1, +2)
        exp_1 = offset_flood_boundary_faces(cache, base_faces, 1)
        exp_2 = offset_flood_boundary_faces(cache, base_faces, 2)
        self.assertGreater(len(exp_1), len(base_faces))
        self.assertGreater(len(exp_2), len(exp_1))
        self.assertTrue(np.all(np.isin(base_faces, exp_1)))

        # 2. Contract boundary (-1, -2)
        cnt_1 = offset_flood_boundary_faces(cache, base_faces, -1)
        cnt_2 = offset_flood_boundary_faces(cache, base_faces, -2)
        self.assertLess(len(cnt_1), len(base_faces))
        self.assertLess(len(cnt_2), len(cnt_1))
        self.assertTrue(np.all(np.isin(cnt_1, base_faces)))

        # 3. Offset = 0 returns identical
        same = offset_flood_boundary_faces(cache, base_faces, 0)
        self.assertEqual(len(same), len(base_faces))
        self.assertTrue(np.array_equal(same, base_faces))

        # 4. In compute_flood_faces: boundary_offset expands and contracts wavefront results
        hit = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        seed = int(base_faces[0])
        res_0 = compute_flood_faces(
            cache, seed, hit, radius=0.6, angle_deg=45.0, crease_resistance=0.95,
            preserve_existing=False, boundary_offset=0
        )
        res_p1 = compute_flood_faces(
            cache, seed, hit, radius=0.6, angle_deg=45.0, crease_resistance=0.95,
            preserve_existing=False, boundary_offset=1
        )
        res_m1 = compute_flood_faces(
            cache, seed, hit, radius=0.6, angle_deg=45.0, crease_resistance=0.95,
            preserve_existing=False, boundary_offset=-1
        )
        self.assertGreater(len(res_p1), len(res_0))
        self.assertLess(len(res_m1), len(res_0))

        # 5. Texture paint: zero bleed onto neighboring unselected faces
        img = bpy.data.images.new("NoBleedTestImg", width=128, height=128, alpha=True)
        img.pixels.foreach_set([0.0] * (128 * 128 * 4))
        uv_layer = obj.data.uv_layers.new(name="UVMap2")
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        t_cache = MeshFloodCache(
            obj=obj,
            depsgraph=depsgraph,
            target_output='TEXTURE',
            default_domain='CORNER',
            default_type='FLOAT_COLOR',
            tex_image=img,
            uv_layer=uv_layer,
        )
        fill_col = (1.0, 0.5, 0.0, 1.0)
        flat_idx, _ = t_cache.apply_texture_faces(base_faces, fill_col, commit=True, smooth_boundary=True)

        neighbor_faces = offset_flood_boundary_faces(cache, base_faces, 1)
        only_neighbor_faces = np.setdiff1d(neighbor_faces, base_faces)
        neighbor_idx, _ = t_cache.apply_texture_faces(only_neighbor_faces, (0.0, 1.0, 0.0, 1.0), commit=False, smooth_boundary=False)
        overlap = np.intersect1d(flat_idx, neighbor_idx)
        self.assertEqual(len(overlap), 0, "Paint must not bleed onto neighboring mesh faces!")

    def test_23_ctrl_scroll_weight_toggle_and_color_to_vertex_group(self):
        """Verify Ctrl+Scroll expand/contract, Weight Paint X toggle, and Color to Vertex Group conversion."""
        from types import SimpleNamespace
        from normal_aware_flood_brush.operators import (
            VIEW3D_OT_normal_flood_brush,
            VIEW3D_OT_flood_weight_toggle,
            VIEW3D_OT_flood_color_to_vertex_group,
        )
        from normal_aware_flood_brush.overlay import FloodOverlayState

        class MockRegion:
            def __init__(self, rtype, x, y, w, h):
                self.type = rtype
                self.x = x
                self.y = y
                self.width = w
                self.height = h

        class MockArea:
            def __init__(self, x, y, w, h, regions):
                self.type = 'VIEW_3D'
                self.x = x
                self.y = y
                self.width = w
                self.height = h
                self.regions = regions

            def tag_redraw(self):
                pass

        mock_area = MockArea(0, 0, 1000, 800, [MockRegion('WINDOW', 0, 0, 1000, 800)])

        class MockOp:
            def __init__(self, area):
                self._area = area
                self._window_reg = area.regions[0]
                self._rv3d = None
                self._cache = None
                self._overlay = FloodOverlayState(area)
                self._target_mode = 'WEIGHT'
                self._is_painting = False
                self._is_navigating = False
                self._is_resizing = False
                self._resize_locked_toggle = False
                self._resize_anchor_x = 0
                self._resize_start_radius = 0.5
                self._undo_stack = []

            def _is_over_ui(self, mx, my):
                return False

            def _sync_overlay_props(self, context):
                pass

            def _redraw_viewport(self, context):
                pass

            def _refresh_at_last_hit(self, context):
                pass

            def modal(self, context, event):
                return VIEW3D_OT_normal_flood_brush.modal(self, context, event)

        def make_event(etype, val='PRESS', mx=500, my=500, shift=False, ctrl=False, alt=False):
            return SimpleNamespace(
                type=etype,
                value=val,
                mouse_x=mx,
                mouse_y=my,
                shift=shift,
                ctrl=ctrl,
                alt=alt,
            )

        props = bpy.context.scene.normal_flood_brush
        op = MockOp(mock_area)

        # 1. Ctrl + Scroll increments and decrements props.boundary_offset
        props.boundary_offset = 0
        res = op.modal(bpy.context, make_event('WHEELUPMOUSE', ctrl=True))
        self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertEqual(props.boundary_offset, 1)

        res = op.modal(bpy.context, make_event('WHEELUPMOUSE', ctrl=True))
        self.assertEqual(props.boundary_offset, 2)

        res = op.modal(bpy.context, make_event('WHEELDOWNMOUSE', ctrl=True))
        self.assertEqual(props.boundary_offset, 1)

        res = op.modal(bpy.context, make_event('WHEELDOWNMOUSE', ctrl=True))
        self.assertEqual(props.boundary_offset, 0)

        # 1b. Regular scroll without Ctrl controls normal_angle_threshold
        props.normal_angle_threshold = 20.0
        res = op.modal(bpy.context, make_event('WHEELUPMOUSE', ctrl=False))
        self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertEqual(props.normal_angle_threshold, 22.0)

        # 1c. Wheel with Alt passes through to navigation
        res = op.modal(bpy.context, make_event('WHEELUPMOUSE', alt=True))
        self.assertEqual(res, {'PASS_THROUGH'})

        # 2. X key in Weight mode toggles weight between 1.0 and 0.0
        props.weight_value = 1.0
        res = op.modal(bpy.context, make_event('X', 'PRESS'))
        self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertAlmostEqual(props.weight_value, 0.0, places=4)

        res = op.modal(bpy.context, make_event('X', 'PRESS'))
        self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertAlmostEqual(props.weight_value, 1.0, places=4)

        # 3. Vanilla Blender Weight Paint toggle operator
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
        wp = bpy.context.tool_settings.weight_paint
        if wp and wp.brush:
            wp.brush.weight = 1.0
        props.weight_value = 1.0
        res = bpy.ops.view3d.flood_weight_toggle()
        self.assertEqual(res, {'FINISHED'})
        self.assertAlmostEqual(props.weight_value, 0.0, places=4)
        if wp and wp.brush:
            self.assertAlmostEqual(wp.brush.weight, 0.0, places=4)

        res = bpy.ops.view3d.flood_weight_toggle()
        self.assertEqual(res, {'FINISHED'})
        self.assertAlmostEqual(props.weight_value, 1.0, places=4)
        if wp and wp.brush:
            self.assertAlmostEqual(wp.brush.weight, 1.0, places=4)

        # 4. Color to Vertex Group conversion (CORNER domain Color Attribute)
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.mesh.primitive_cylinder_add(vertices=16)
        cyl = bpy.context.active_object
        mesh = cyl.data
        ca = mesh.color_attributes.new(name="TestCol", type='FLOAT_COLOR', domain='CORNER')
        loop_cols = np.zeros((len(mesh.loops), 4), dtype=np.float32)
        # Color top half blue (0, 0, 1, 1)
        loop_cols[:len(mesh.loops)//2] = [0.0, 0.0, 1.0, 1.0]
        ca.data.foreach_set('color', loop_cols.ravel())

        props.fill_color = (0.0, 0.0, 1.0, 1.0)
        res = bpy.ops.view3d.flood_color_to_vertex_group(color=(0.0, 0.0, 1.0, 1.0), group_name="Blue_Group")
        self.assertEqual(res, {'FINISHED'})
        self.assertIn("Blue_Group", cyl.vertex_groups)
        vg_blue = cyl.vertex_groups["Blue_Group"]
        self.assertEqual(cyl.vertex_groups.active, vg_blue)

        # Verify vertices with blue color have weight 1.0
        loop_verts = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get('vertex_index', loop_verts)
        blue_verts = np.unique(loop_verts[:len(mesh.loops)//2])
        self.assertGreater(blue_verts.size, 0)
        for bv in blue_verts:
            self.assertAlmostEqual(vg_blue.weight(int(bv)), 1.0, places=4)

        # 5. Color to Vertex Group conversion (POINT domain Color Attribute)
        ca_pt = mesh.color_attributes.new(name="PointCol", type='FLOAT_COLOR', domain='POINT')
        mesh.color_attributes.active_color = ca_pt
        pt_cols = np.zeros((len(mesh.vertices), 4), dtype=np.float32)
        pt_cols[:8] = [1.0, 0.5, 0.0, 1.0]
        ca_pt.data.foreach_set('color', pt_cols.ravel())

        res = bpy.ops.view3d.flood_color_to_vertex_group(color=(1.0, 0.5, 0.0, 1.0), group_name="Orange_Group")
        self.assertEqual(res, {'FINISHED'})
        self.assertIn("Orange_Group", cyl.vertex_groups)
        vg_orange = cyl.vertex_groups["Orange_Group"]
        for vi in range(8):
            self.assertAlmostEqual(vg_orange.weight(vi), 1.0, places=4)

        # 6. Color to Vertex Group conversion (UV Image Texture)
        bpy.ops.mesh.primitive_cylinder_add(vertices=16)
        tex_cyl = bpy.context.active_object
        tex_mesh = tex_cyl.data
        t_img = bpy.data.images.new("TexGroupImg", 64, 64)
        t_pix = np.zeros((64 * 64, 4), dtype=np.float32)
        t_pix[:] = [0.0, 1.0, 0.0, 1.0]  # Solid green
        t_img.pixels.foreach_set(t_pix.ravel())
        t_uv = tex_mesh.uv_layers.new(name="UVMap")
        bpy.context.view_layer.update()

        props.texpaint_target = 'TEXTURE'
        bpy.context.tool_settings.image_paint.canvas = t_img
        res = bpy.ops.view3d.flood_color_to_vertex_group(color=(0.0, 1.0, 0.0, 1.0), group_name="Green_Group")
        self.assertEqual(res, {'FINISHED'})
        self.assertIn("Green_Group", tex_cyl.vertex_groups)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestNormalAwareFloodBrush)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
