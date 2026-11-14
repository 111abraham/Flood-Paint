# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from .properties import get_mesh_active_color_attribute


# Single-entry pure-NumPy cache for heavy mesh geometry, CSR adjacency, and UV tile masks.
# Validated strictly via dimensions, matrix_world, vertex coordinates, and loop vertex indices
# so zero bpy.types.ID references are retained and edits/sculpts/UV edits never go stale.
_GEOM_CACHE: dict | None = None


def clear_geometry_cache():
    """Clear the cached pure-NumPy mesh geometry, adjacency, and UV arrays."""
    global _GEOM_CACHE
    _GEOM_CACHE = None


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB float32 values in [0, 1] to sRGB using IEC 61966-2-1."""
    c = np.clip(rgb, 0.0, 1.0)
    return np.where(
        c <= 0.0031308,
        c * 12.92,
        1.055 * np.power(np.maximum(c, 0.0031308), 1.0 / 2.4) - 0.055,
    ).astype(np.float32, copy=False)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB float32 values in [0, 1] to linear RGB using IEC 61966-2-1."""
    c = np.clip(rgb, 0.0, 1.0)
    return np.where(
        c <= 0.04045,
        c / 12.92,
        np.power((c + 0.055) / 1.055, 2.4),
    ).astype(np.float32, copy=False)


def rasterize_uv_triangles(
    tri_uvs_sub: np.ndarray,
    width: int,
    height: int,
    x_offset: int = 0,
    y_offset: int = 0,
    out_w: int | None = None,
    out_h: int | None = None,
) -> np.ndarray:
    """Rasterize normalized UV triangles (K, 3, 2) into a boolean pixel mask.

    Supports both small/medium triangles (batched vectorized) and large triangles (per-bbox grid),
    as well as sub-region crops via (x_offset, y_offset, out_w, out_h).
    """
    crop_w = width if out_w is None else out_w
    crop_h = height if out_h is None else out_h
    mask = np.zeros((crop_h, crop_w), dtype=bool)
    if tri_uvs_sub.size == 0 or crop_w <= 0 or crop_h <= 0:
        return mask

    px_uv = tri_uvs_sub.astype(np.float32, copy=True)
    px_uv[:, :, 0] = px_uv[:, :, 0] * width - x_offset
    px_uv[:, :, 1] = px_uv[:, :, 1] * height - y_offset

    px = px_uv[:, :, 0]
    py = px_uv[:, :, 1]
    x0 = np.clip(np.floor(px.min(axis=1)).astype(np.int32), 0, crop_w - 1)
    x1 = np.clip(np.ceil(px.max(axis=1)).astype(np.int32), 0, crop_w - 1)
    y0 = np.clip(np.floor(py.min(axis=1)).astype(np.int32), 0, crop_h - 1)
    y1 = np.clip(np.ceil(py.max(axis=1)).astype(np.int32), 0, crop_h - 1)

    pw = np.maximum(x1 - x0 + 1, 0)
    ph = np.maximum(y1 - y0 + 1, 0)
    areas = pw * ph

    ax, ay = px[:, 0], py[:, 0]
    bx, by = px[:, 1], py[:, 1]
    cx, cy = px[:, 2], py[:, 2]

    det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    valid = (areas > 0) & (np.abs(det) > 1e-6)
    if not np.any(valid):
        return mask

    # Keep only non-degenerate triangles and precompute edge coefficients once per triangle.
    ax, ay = ax[valid], ay[valid]
    bx, by = bx[valid], by[valid]
    cx, cy = cx[valid], cy[valid]
    x0, x1 = x0[valid], x1[valid]
    y0, y1 = y0[valid], y1[valid]
    pw, areas = pw[valid], areas[valid]
    sgn = np.where(det[valid] >= 0.0, np.float32(1.0), np.float32(-1.0))

    dx0, dy0 = bx - ax, by - ay
    dx1, dy1 = cx - bx, cy - by
    dx2, dy2 = ax - cx, ay - cy

    t0 = 1e-3 * (np.abs(dx0) + np.abs(dy0) + 1.0)
    t1 = 1e-3 * (np.abs(dx1) + np.abs(dy1) + 1.0)
    t2 = 1e-3 * (np.abs(dx2) + np.abs(dy2) + 1.0)

    u0 = -dy0 * sgn
    v0 = dx0 * sgn
    k0 = (ax * by - ay * bx) * sgn + t0

    u1 = -dy1 * sgn
    v1 = dx1 * sgn
    k1 = (bx * cy - by * cx) * sgn + t1

    u2 = -dy2 * sgn
    v2 = dx2 * sgn
    k2 = (cx * ay - cy * ax) * sgn + t2

    # 1. Fast path for 1x1 sub-pixel triangles (common on 1M+ meshes).
    single = areas == 1
    if np.any(single):
        sx = x0[single].astype(np.float32) + 0.5
        sy = y0[single].astype(np.float32) + 0.5
        hit_s = (
            (u0[single] * sx + v0[single] * sy + k0[single] >= 0.0)
            & (u1[single] * sx + v1[single] * sy + k1[single] >= 0.0)
            & (u2[single] * sx + v2[single] * sy + k2[single] >= 0.0)
        )
        if np.any(hit_s):
            mask[y0[single][hit_s], x0[single][hit_s]] = True

    # 2. Chunked vectorized path with progressive edge culling (1 < area <= 65536).
    small = (areas > 1) & (areas <= 65536)
    if np.any(small):
        sx0, sy0, spw, sarea = x0[small], y0[small], pw[small], areas[small]
        su0, sv0, sk0 = u0[small], v0[small], k0[small]
        su1, sv1, sk1 = u1[small], v1[small], k1[small]
        su2, sv2, sk2 = u2[small], v2[small], k2[small]

        n_small = sarea.shape[0]
        cum_all = np.cumsum(sarea)
        idx_start = 0
        while idx_start < n_small:
            base = int(cum_all[idx_start - 1]) if idx_start > 0 else 0
            idx_end = int(np.searchsorted(cum_all, base + 2_000_000, side='right'))
            if idx_end <= idx_start:
                idx_end = idx_start + 1

            c_area = sarea[idx_start:idx_end]
            c_pw = spw[idx_start:idx_end]
            c_x0 = sx0[idx_start:idx_end]
            c_y0 = sy0[idx_start:idx_end]
            cu0, cv0, ck0 = su0[idx_start:idx_end], sv0[idx_start:idx_end], sk0[idx_start:idx_end]
            cu1, cv1, ck1 = su1[idx_start:idx_end], sv1[idx_start:idx_end], sk1[idx_start:idx_end]
            cu2, cv2, ck2 = su2[idx_start:idx_end], sv2[idx_start:idx_end], sk2[idx_start:idx_end]

            p_total = int(c_area.sum())
            cum = np.cumsum(c_area)
            tri_rep = np.repeat(np.arange(idx_end - idx_start, dtype=np.int32), c_area)
            local_i = np.arange(p_total, dtype=np.int32) - (cum - c_area)[tri_rep]
            pw_rep = c_pw[tri_rep]
            gx = (c_x0[tri_rep] + (local_i % pw_rep)).astype(np.float32) + 0.5
            gy = (c_y0[tri_rep] + (local_i // pw_rep)).astype(np.float32) + 0.5

            hit = (cu0[tri_rep] * gx + cv0[tri_rep] * gy + ck0[tri_rep]) >= 0.0
            if np.any(hit):
                tr1 = tri_rep[hit]
                gx1 = gx[hit]
                gy1 = gy[hit]
                hit1 = (cu1[tr1] * gx1 + cv1[tr1] * gy1 + ck1[tr1]) >= 0.0
                if np.any(hit1):
                    tr2 = tr1[hit1]
                    gx2 = gx1[hit1]
                    gy2 = gy1[hit1]
                    hit2 = (cu2[tr2] * gx2 + cv2[tr2] * gy2 + ck2[tr2]) >= 0.0
                    if np.any(hit2):
                        mask[gy2[hit2].astype(np.int32), gx2[hit2].astype(np.int32)] = True

            idx_start = idx_end

    # 3. Vectorized slice path for very large triangles (area > 65536 pixels).
    large_indices = np.flatnonzero(areas > 65536)
    for ti in large_indices.tolist():
        tx0, tx1 = int(x0[ti]), int(x1[ti])
        ty0, ty1 = int(y0[ti]), int(y1[ti])
        gx = np.arange(tx0, tx1 + 1, dtype=np.float32)[None, :] + 0.5
        gy = np.arange(ty0, ty1 + 1, dtype=np.float32)[:, None] + 0.5
        inside = (
            (u0[ti] * gx + v0[ti] * gy + k0[ti] >= 0.0)
            & (u1[ti] * gx + v1[ti] * gy + k1[ti] >= 0.0)
            & (u2[ti] * gx + v2[ti] * gy + k2[ti] >= 0.0)
        )
        mask[ty0 : ty1 + 1, tx0 : tx1 + 1] |= inside

    return mask


class MeshFloodCache:
    """Pre-cached NumPy topology, normals, BVHTree, and Color/FaceSet/Texture buffers."""

    def __init__(
        self,
        obj,
        depsgraph,
        target_output: str,
        default_domain: str,
        default_type: str,
        tex_image=None,
        uv_layer=None,
        tex_seam_bleed: int = 2,
        restrict_mat_index: int | None = None,
        store_initial_tex: bool = True,
        build_adjacency: bool = True,
        build_bvh: bool = True,
    ):
        self.obj = obj
        self.mesh = obj.data
        mesh = self.mesh

        self.num_faces = len(mesh.polygons)
        self.num_verts = len(mesh.vertices)
        self.num_loops = len(mesh.loops)
        self.num_edges = len(mesh.edges)

        if self.num_faces == 0:
            raise ValueError("Active mesh has no faces.")

        # World/local transform helpers (keep positions origin-centered for float32 precision).
        self.matrix_world = obj.matrix_world.copy()
        self.inv_matrix_world = obj.matrix_world.inverted_safe()
        self.rot_scale_3x3 = np.asarray(self.matrix_world.to_3x3(), dtype=np.float32)
        self.inv_m3 = np.asarray(self.inv_matrix_world.to_3x3(), dtype=np.float32)
        matrix_key = tuple(np.asarray(self.matrix_world, dtype=np.float32).ravel().tolist())

        # Vertex positions in object local space (used directly in GPU POST_VIEW draw with matrix_world).
        vert_coords_flat = np.empty(self.num_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", vert_coords_flat)
        self.vert_coords = vert_coords_flat.reshape((self.num_verts, 3))

        vert_normals_flat = np.empty(self.num_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("normal", vert_normals_flat)
        self.vert_normals = vert_normals_flat.reshape((self.num_verts, 3))

        self.loop_verts = np.empty(self.num_loops, dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", self.loop_verts)

        global _GEOM_CACHE
        cached = _GEOM_CACHE
        geom_hit = (
            cached is not None
            and cached["mesh_name"] == mesh.name
            and cached["dims"] == (self.num_verts, self.num_faces, self.num_loops, self.num_edges)
            and cached["matrix_key"] == matrix_key
            and (not build_adjacency or cached["adj_offsets"] is not None)
            and np.array_equal(self.vert_coords, cached["vert_coords"])
            and np.array_equal(self.loop_verts, cached["loop_verts"])
        )

        if geom_hit:
            self.vert_normals = cached.get("vert_normals", self.vert_normals)
            self.face_centers = cached["face_centers"]
            self.face_normals = cached["face_normals"]
            self.poly_starts = cached["poly_starts"]
            self.poly_totals = cached["poly_totals"]
            self.loop_edges = cached["loop_edges"]
            self.edge_verts = cached["edge_verts"]
            self.all_tris = cached["all_tris"]
            self.tri_verts = cached["tri_verts"]
            self.tri_poly_idx = cached["tri_poly_idx"]
            self.tri_loops = cached["tri_loops"]
            self.poly_first_verts = cached["poly_first_verts"]
            self.adj_offsets = cached["adj_offsets"]
            self.adj_neighbors = cached["adj_neighbors"]
            self.adj_edges = cached["adj_edges"]
            self.adj_angles = cached["adj_angles"]
            self.adj_dists = cached["adj_dists"]
            self._geom_cache_entry = cached
        else:
            # Face centers (scaled/rotated relative to object origin) and world-space unit normals.
            centers_local = np.empty(self.num_faces * 3, dtype=np.float32)
            mesh.polygons.foreach_get("center", centers_local)
            self.face_centers = centers_local.reshape((self.num_faces, 3)) @ self.rot_scale_3x3.T

            normals_local = np.empty(self.num_faces * 3, dtype=np.float32)
            mesh.polygons.foreach_get("normal", normals_local)
            normals_world = normals_local.reshape((self.num_faces, 3)) @ self.inv_m3
            norm_len = np.linalg.norm(normals_world, axis=1, keepdims=True)
            valid_norm = norm_len[:, 0] > 1e-8
            normals_world[valid_norm] /= norm_len[valid_norm]
            normals_world[~valid_norm] = 0.0
            self.face_normals = normals_world

            # Polygon loop ranges & loop/edge topology (skip loop_total fetch on pure-tri meshes).
            self.poly_starts = np.empty(self.num_faces, dtype=np.int32)
            mesh.polygons.foreach_get("loop_start", self.poly_starts)
            if self.num_loops == self.num_faces * 3:
                self.poly_totals = np.full(self.num_faces, 3, dtype=np.int32)
            else:
                self.poly_totals = np.empty(self.num_faces, dtype=np.int32)
                mesh.polygons.foreach_get("loop_total", self.poly_totals)

            self.loop_edges = np.empty(self.num_loops, dtype=np.int32)
            mesh.loops.foreach_get("edge_index", self.loop_edges)

            edge_verts = np.empty(self.num_edges * 2, dtype=np.int32)
            mesh.edges.foreach_get("vertices", edge_verts)
            self.edge_verts = edge_verts.reshape((self.num_edges, 2))

            self.all_tris = bool(
                self.num_loops == self.num_faces * 3
                and np.all(self.poly_totals == 3)
                and (self.num_faces == 0 or self.poly_starts[0] == 0)
                and bool(np.all(self.poly_starts == np.arange(self.num_faces, dtype=np.int32) * 3))
            )

            # Triangulation cache for GPU preview overlay and Texture Paint UV rasterization.
            if self.all_tris:
                self.tri_verts = self.loop_verts.reshape((self.num_faces, 3))
                self.tri_poly_idx = np.arange(self.num_faces, dtype=np.int32)
                self.tri_loops = None
            else:
                mesh.calc_loop_triangles()
                num_tris = len(mesh.loop_triangles)
                tri_v = np.empty(num_tris * 3, dtype=np.int32)
                mesh.loop_triangles.foreach_get("vertices", tri_v)
                self.tri_verts = tri_v.reshape((num_tris, 3))
                self.tri_poly_idx = np.empty(num_tris, dtype=np.int32)
                mesh.loop_triangles.foreach_get("polygon_index", self.tri_poly_idx)
                tri_l = np.empty(num_tris * 3, dtype=np.int32)
                mesh.loop_triangles.foreach_get("loops", tri_l)
                self.tri_loops = tri_l.reshape((num_tris, 3))

            self.poly_first_verts = self.loop_verts[self.poly_starts]

            # Build CSR face-to-face adjacency graph via vectorized NumPy sorting on loop_edges.
            if build_adjacency:
                self._build_csr_adjacency()
            else:
                self.adj_offsets = None
                self.adj_neighbors = None
                self.adj_edges = None
                self.adj_angles = None
                self.adj_dists = None

            if build_adjacency:
                _GEOM_CACHE = {
                    "mesh_name": mesh.name,
                    "dims": (self.num_verts, self.num_faces, self.num_loops, self.num_edges),
                    "matrix_key": matrix_key,
                    "vert_coords": self.vert_coords,
                    "vert_normals": self.vert_normals,
                    "loop_verts": self.loop_verts,
                    "face_centers": self.face_centers,
                    "face_normals": self.face_normals,
                    "poly_starts": self.poly_starts,
                    "poly_totals": self.poly_totals,
                    "loop_edges": self.loop_edges,
                    "edge_verts": self.edge_verts,
                    "all_tris": self.all_tris,
                    "tri_verts": self.tri_verts,
                    "tri_poly_idx": self.tri_poly_idx,
                    "tri_loops": self.tri_loops,
                    "poly_first_verts": self.poly_first_verts,
                    "adj_offsets": self.adj_offsets,
                    "adj_neighbors": self.adj_neighbors,
                    "adj_edges": self.adj_edges,
                    "adj_angles": self.adj_angles,
                    "adj_dists": self.adj_dists,
                    "uv_cache": None,
                }
                self._geom_cache_entry = _GEOM_CACHE
            else:
                self._geom_cache_entry = None

        # Hidden face mask (.hide_poly + negative .sculpt_face_set).
        self.hidden_faces = np.zeros(self.num_faces, dtype=bool)
        hide_attr = mesh.attributes.get(".hide_poly")
        if hide_attr is not None and hide_attr.domain == 'FACE' and len(hide_attr.data) == self.num_faces:
            hide_buf = np.empty(self.num_faces, dtype=bool)
            hide_attr.data.foreach_get("value", hide_buf)
            self.hidden_faces |= hide_buf

        # Build BVHTree (fallback to FromPolygons if generative modifiers alter face count).
        if build_bvh:
            eval_obj = obj.evaluated_get(depsgraph)
            if len(eval_obj.data.polygons) == self.num_faces:
                self.bvh = BVHTree.FromObject(obj, depsgraph)
            else:
                verts_list = self.vert_coords.tolist()
                polys_list = [
                    self.loop_verts[s : s + t].tolist()
                    for s, t in zip(self.poly_starts.tolist(), self.poly_totals.tolist())
                ]
                self.bvh = BVHTree.FromPolygons(verts_list, polys_list)
        else:
            self.bvh = None

        # Attribute buffers.
        self.color_attr = None
        self.color_domain = None
        self.color_type = None
        self.color_buf = None
        self.srgb_buf = None
        self.initial_color_buf = None

        self.face_set_attr = None
        self.face_set_buf = None
        self.initial_face_set_buf = None
        self.had_face_set_attr = False
        self.next_face_set_id = 2

        # Texture Paint UV / Image buffers.
        self.tex_image = None
        self.tex_width = 0
        self.tex_height = 0
        self.tex_is_srgb = True
        self.tex_seam_bleed = max(0, min(8, int(tex_seam_bleed)))
        self.tex_buf = None
        self.tex_buf_flat = None
        self.initial_tex_buf = None
        self._tex_revert_history = []
        self.tri_uvs = None
        self.face_uv_centroids = None
        self.face_tex_colors = None
        self._uv_covered_mask = None
        self._uv_cov_tris = None
        self._uv_tri_px_bounds = None
        self._uv_tile_size = 256
        self._uv_tiles_done = None
        self.tex_protected_mask = None
        self.poly_mat_indices = None
        self.restrict_mat_index = restrict_mat_index

        self.ensure_face_set_buffer(create_if_missing=(target_output in {'FACE_SET', 'BOTH'}))
        if self.face_set_buf is not None:
            self.hidden_faces |= (self.face_set_buf < 0)

        if target_output == 'TEXTURE':
            self._init_texture_paint_cache(
                tex_image, uv_layer, restrict_mat_index, store_initial_tex=store_initial_tex
            )
        elif target_output in {'COLOR', 'BOTH'}:
            self.ensure_color_buffer(default_domain, default_type, create_if_missing=True)
        else:
            self.ensure_color_buffer(default_domain, default_type, create_if_missing=False)

        # Reusable scratch buffers for zero-allocation flood propagation and preview extraction.
        self.cost_buf = np.full(self.num_faces, np.inf, dtype=np.float32)
        self.in_mask_buf = np.zeros(self.num_faces, dtype=bool)
        self.painted_faces = np.zeros(self.num_faces, dtype=bool)

    @property
    def uv_covered_mask(self) -> np.ndarray | None:
        """Lazily build and return the full (H, W) UV-covered mask when explicitly requested."""
        if self._uv_covered_mask is None and self._uv_cov_tris is not None:
            self.ensure_full_uv_covered_mask()
        return self._uv_covered_mask

    def _init_texture_paint_cache(
        self,
        tex_image,
        uv_layer,
        restrict_mat_index: int | None,
        store_initial_tex: bool = True,
    ):
        if tex_image is None or uv_layer is None:
            raise ValueError("Texture Paint mode requires an active UV layer and paint Image.")
        w, h = int(tex_image.size[0]), int(tex_image.size[1])
        if w <= 0 or h <= 0:
            raise ValueError(f"Paint image '{tex_image.name}' has invalid dimensions ({w}x{h}).")

        self.tex_image = tex_image
        self.tex_width = w
        self.tex_height = h
        self.tex_is_srgb = "srgb" in tex_image.colorspace_settings.name.lower()

        px = np.empty(w * h * 4, dtype=np.float32)
        tex_image.pixels.foreach_get(px)
        self.tex_buf = px.reshape((h, w, 4))
        self.tex_buf_flat = self.tex_buf.reshape((w * h, 4))
        self.initial_tex_buf = self.tex_buf.copy() if store_initial_tex else None
        self._tex_revert_history = []
        self.tex_protected_mask = np.zeros((h, w), dtype=bool)
        self.face_tex_colors = None

        loop_uvs_flat = np.empty(self.num_loops * 2, dtype=np.float32)
        uv_layer.uv.foreach_get("vector", loop_uvs_flat)
        loop_uvs = loop_uvs_flat.reshape((self.num_loops, 2))

        geom_entry = getattr(self, "_geom_cache_entry", None)
        uvc = geom_entry.get("uv_cache") if geom_entry is not None else None
        uv_hit = (
            uvc is not None
            and uvc["uv_name"] == uv_layer.name
            and np.array_equal(loop_uvs, uvc["loop_uvs"])
        )

        if uv_hit:
            self.tri_uvs = uvc["tri_uvs"]
            self.face_uv_centroids = uvc["face_uv_centroids"]
        else:
            if self.all_tris:
                self.tri_uvs = loop_uvs.reshape((self.num_faces, 3, 2))
                self.face_uv_centroids = self.tri_uvs.mean(axis=1)
            else:
                self.tri_uvs = loop_uvs[self.tri_loops]
                # Per-polygon UV centroid via loop_faces mean.
                loop_faces = np.repeat(np.arange(self.num_faces, dtype=np.int32), self.poly_totals)
                inv_tot = 1.0 / np.maximum(self.poly_totals.astype(np.float32), 1.0)
                cu = np.bincount(loop_faces, weights=loop_uvs[:, 0], minlength=self.num_faces) * inv_tot
                cv = np.bincount(loop_faces, weights=loop_uvs[:, 1], minlength=self.num_faces) * inv_tot
                self.face_uv_centroids = np.column_stack((cu, cv)).astype(np.float32)

        nx = (w + self._uv_tile_size - 1) // self._uv_tile_size
        ny = (h + self._uv_tile_size - 1) // self._uv_tile_size
        cov_key = (w, h, restrict_mat_index)

        if uv_hit and uvc["cov_key"] == cov_key and restrict_mat_index is None:
            self._uv_cov_tris = uvc["_uv_cov_tris"]
            self._uv_covered_mask = uvc["_uv_covered_mask"]
            self._uv_tri_px_bounds = uvc["_uv_tri_px_bounds"]
            self._uv_tiles_done = uvc["_uv_tiles_done"]
        else:
            if restrict_mat_index is not None:
                mat_idx = np.empty(self.num_faces, dtype=np.int32)
                self.mesh.polygons.foreach_get("material_index", mat_idx)
                self.poly_mat_indices = mat_idx
                cov_tris = self.tri_uvs[mat_idx[self.tri_poly_idx] == restrict_mat_index]
            else:
                cov_tris = self.tri_uvs

            self._uv_cov_tris = cov_tris
            self._uv_covered_mask = np.zeros((h, w), dtype=bool)
            self._uv_tri_px_bounds = None
            self._uv_tiles_done = np.zeros((ny, nx), dtype=bool)
            self._ensure_uv_tri_bounds()

            if geom_entry is not None and restrict_mat_index is None:
                geom_entry["uv_cache"] = {
                    "uv_name": uv_layer.name,
                    "loop_uvs": loop_uvs,
                    "tri_uvs": self.tri_uvs,
                    "face_uv_centroids": self.face_uv_centroids,
                    "cov_key": cov_key,
                    "_uv_cov_tris": self._uv_cov_tris,
                    "_uv_tri_px_bounds": self._uv_tri_px_bounds,
                    "_uv_tiles_done": self._uv_tiles_done,
                    "_uv_covered_mask": self._uv_covered_mask,
                }

    def _ensure_uv_tri_bounds(self):
        if self._uv_tri_px_bounds is not None or self._uv_cov_tris is None:
            return
        w, h = self.tex_width, self.tex_height
        xs = self._uv_cov_tris[:, :, 0] * w
        ys = self._uv_cov_tris[:, :, 1] * h
        bx0 = np.clip(np.floor(xs.min(axis=1)).astype(np.int32), 0, w - 1)
        bx1 = np.clip(np.ceil(xs.max(axis=1)).astype(np.int32), 0, w - 1)
        by0 = np.clip(np.floor(ys.min(axis=1)).astype(np.int32), 0, h - 1)
        by1 = np.clip(np.ceil(ys.max(axis=1)).astype(np.int32), 0, h - 1)
        self._uv_tri_px_bounds = (bx0, bx1, by0, by1)

    def get_uv_covered_crop(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray | None:
        """Lazily rasterize and return the UV-covered boolean sub-mask for [y0:y1, x0:x1]."""
        if self._uv_cov_tris is None or self._uv_tiles_done is None:
            return None
        w, h = self.tex_width, self.tex_height
        x0 = max(0, min(w, int(x0)))
        x1 = max(x0, min(w, int(x1)))
        y0 = max(0, min(h, int(y0)))
        y1 = max(y0, min(h, int(y1)))
        if x1 <= x0 or y1 <= y0:
            return None

        ts = self._uv_tile_size
        tx0 = x0 // ts
        tx1 = (x1 - 1) // ts
        ty0 = y0 // ts
        ty1 = (y1 - 1) // ts

        if not np.all(self._uv_tiles_done[ty0 : ty1 + 1, tx0 : tx1 + 1]):
            if self._uv_covered_mask is None:
                self._uv_covered_mask = np.zeros((h, w), dtype=bool)
            self._ensure_uv_tri_bounds()
            bx0, bx1, by0, by1 = self._uv_tri_px_bounds

            # Shrink rectangle strictly to uncached tile rows/cols.
            sub_done = self._uv_tiles_done[ty0 : ty1 + 1, tx0 : tx1 + 1]
            missing_rows, missing_cols = np.nonzero(~sub_done)
            utx0 = tx0 + int(missing_cols.min())
            utx1 = tx0 + int(missing_cols.max())
            uty0 = ty0 + int(missing_rows.min())
            uty1 = ty0 + int(missing_rows.max())

            rx0 = utx0 * ts
            rx1 = min(w, (utx1 + 1) * ts)
            ry0 = uty0 * ts
            ry1 = min(h, (uty1 + 1) * ts)

            overlap = (bx1 >= rx0) & (bx0 < rx1) & (by1 >= ry0) & (by0 < ry1)
            if np.any(overlap):
                self._uv_covered_mask[ry0:ry1, rx0:rx1] |= rasterize_uv_triangles(
                    self._uv_cov_tris[overlap],
                    w,
                    h,
                    x_offset=rx0,
                    y_offset=ry0,
                    out_w=rx1 - rx0,
                    out_h=ry1 - ry0,
                )
            self._uv_tiles_done[uty0 : uty1 + 1, utx0 : utx1 + 1] = True

        return self._uv_covered_mask[y0:y1, x0:x1]

    def ensure_full_uv_covered_mask(self) -> np.ndarray | None:
        if self._uv_cov_tris is None:
            return None
        if self._uv_covered_mask is None:
            self._uv_covered_mask = rasterize_uv_triangles(self._uv_cov_tris, self.tex_width, self.tex_height)
            if self._uv_tiles_done is not None:
                self._uv_tiles_done.fill(True)
        elif self._uv_tiles_done is not None and not np.all(self._uv_tiles_done):
            self.get_uv_covered_crop(0, 0, self.tex_width, self.tex_height)
        return self._uv_covered_mask

    def _build_csr_adjacency(self):
        nf = self.num_faces
        if self.num_loops == 0 or self.num_edges == 0:
            self.adj_offsets = np.zeros(nf + 1, dtype=np.int32)
            self.adj_neighbors = np.empty(0, dtype=np.int32)
            self.adj_edges = np.empty(0, dtype=np.int32)
            self.adj_angles = np.empty(0, dtype=np.float32)
            self.adj_dists = np.empty(0, dtype=np.float32)
            return

        loop_faces = np.repeat(np.arange(nf, dtype=np.int32), self.poly_totals)
        order = np.argsort(self.loop_edges, kind='quicksort')
        se = self.loop_edges[order]
        sf = loop_faces[order]

        mask = (se[:-1] == se[1:]) & (sf[:-1] != sf[1:])
        u_parts = [sf[:-1][mask]]
        v_parts = [sf[1:][mask]]
        e_parts = [se[:-1][mask]]

        if se.size >= 3:
            non_manifold_pos = np.flatnonzero(se[:-2] == se[2:])
            if non_manifold_pos.size > 0:
                bad_edges = np.unique(se[non_manifold_pos])
                extra_u = []
                extra_v = []
                extra_e = []
                for edge_id in bad_edges.tolist():
                    left = int(np.searchsorted(se, edge_id, side='left'))
                    right = int(np.searchsorted(se, edge_id, side='right'))
                    faces = sf[left:right]
                    k = faces.size
                    for i in range(k):
                        for j in range(i + 1, k):
                            if faces[i] != faces[j]:
                                extra_u.append(int(faces[i]))
                                extra_v.append(int(faces[j]))
                                extra_e.append(int(edge_id))
                if extra_u:
                    u_parts.append(np.asarray(extra_u, dtype=np.int32))
                    v_parts.append(np.asarray(extra_v, dtype=np.int32))
                    e_parts.append(np.asarray(extra_e, dtype=np.int32))

        u = np.concatenate(u_parts)
        v = np.concatenate(v_parts)
        e = np.concatenate(e_parts)

        if u.size == 0:
            self.adj_offsets = np.zeros(nf + 1, dtype=np.int32)
            self.adj_neighbors = np.empty(0, dtype=np.int32)
            self.adj_edges = np.empty(0, dtype=np.int32)
            self.adj_angles = np.empty(0, dtype=np.float32)
            self.adj_dists = np.empty(0, dtype=np.float32)
            return

        all_u = np.concatenate((u, v))
        all_v = np.concatenate((v, u))
        all_e = np.concatenate((e, e))

        u_order = np.argsort(all_u, kind='quicksort')
        all_u = all_u[u_order]
        all_v = all_v[u_order]
        all_e = all_e[u_order]

        counts = np.bincount(all_u, minlength=nf).astype(np.int32)
        adj_offsets = np.zeros(nf + 1, dtype=np.int32)
        np.cumsum(counts, out=adj_offsets[1:])

        dots = np.einsum('ij,ij->i', self.face_normals[all_u], self.face_normals[all_v])
        np.clip(dots, -1.0, 1.0, out=dots)

        self.adj_offsets = adj_offsets
        self.adj_neighbors = all_v
        self.adj_edges = all_e
        self.adj_angles = np.arccos(dots).astype(np.float32)
        self.adj_dists = np.linalg.norm(self.face_centers[all_u] - self.face_centers[all_v], axis=1).astype(np.float32)

    def ensure_color_buffer(self, default_domain: str, default_type: str, create_if_missing: bool = True):
        if self.color_attr is not None:
            return self.color_attr

        mesh = self.mesh
        ca = get_mesh_active_color_attribute(mesh)

        if ca is None:
            if not create_if_missing:
                return None
            ca = mesh.color_attributes.new(name="Color", type=default_type, domain=default_domain)
            n_elems = len(ca.data)
            init_white = np.ones((n_elems, 4), dtype=np.float32)
            if ca.data_type == 'BYTE_COLOR':
                ca.data.foreach_set("color_srgb", init_white.ravel())
            else:
                ca.data.foreach_set("color", init_white.ravel())
            mesh.update_gpu_tag()
            mesh.update()

        mesh.color_attributes.active_color = ca
        mesh.color_attributes.active = ca
        if mesh.color_attributes.render_color_index < 0 and mesh.color_attributes.active_color_index >= 0:
            mesh.color_attributes.render_color_index = mesh.color_attributes.active_color_index

        self.color_attr = ca
        self.color_domain = ca.domain
        self.color_type = ca.data_type

        n_elems = len(ca.data)
        buf = np.empty(n_elems * 4, dtype=np.float32)
        ca.data.foreach_get("color", buf)
        self.color_buf = buf.reshape((n_elems, 4))
        self.initial_color_buf = self.color_buf.copy()

        if ca.data_type == 'BYTE_COLOR':
            srgb_buf = np.empty(n_elems * 4, dtype=np.float32)
            ca.data.foreach_get("color_srgb", srgb_buf)
            self.srgb_buf = srgb_buf.reshape((n_elems, 4))
        else:
            self.srgb_buf = None

        return ca

    def ensure_face_set_buffer(self, create_if_missing: bool = False):
        if self.face_set_buf is not None:
            return self.face_set_attr

        mesh = self.mesh
        fs_attr = mesh.attributes.get(".sculpt_face_set")
        if fs_attr is not None:
            self.had_face_set_attr = True
        elif create_if_missing:
            fs_attr = mesh.attributes.new(name=".sculpt_face_set", type='INT', domain='FACE')
            init_fs = np.ones(self.num_faces, dtype=np.int32)
            fs_attr.data.foreach_set("value", init_fs)
            self.had_face_set_attr = False
            mesh.update_gpu_tag()
            mesh.update()
        else:
            return None

        self.face_set_attr = fs_attr
        fs_buf = np.empty(self.num_faces, dtype=np.int32)
        fs_attr.data.foreach_get("value", fs_buf)
        self.face_set_buf = fs_buf
        self.initial_face_set_buf = fs_buf.copy()
        max_id = int(np.abs(fs_buf).max(initial=1))
        self.next_face_set_id = max(max_id + 1, 2)
        return fs_attr

    def faces_to_color_indices(self, face_indices: np.ndarray) -> np.ndarray:
        """Map polygon indices to active color attribute element indices (loops or vertices)."""
        if face_indices.size == 0:
            return np.empty(0, dtype=np.int32)

        if self.all_tris:
            starts = face_indices * 3
            loop_idx = np.column_stack((starts, starts + 1, starts + 2)).ravel()
        else:
            starts = self.poly_starts[face_indices]
            totals = self.poly_totals[face_indices]
            total_loops = int(totals.sum())
            if total_loops == 0:
                return np.empty(0, dtype=np.int32)
            cum = np.cumsum(totals)
            group_offsets = np.repeat(starts - (cum - totals), totals)
            loop_idx = np.arange(total_loops, dtype=np.int32) + group_offsets

        if self.color_domain == 'CORNER':
            return loop_idx
        return np.unique(self.loop_verts[loop_idx])

    def faces_to_vert_indices(self, face_indices: np.ndarray) -> np.ndarray:
        """Map polygon indices to unique mesh vertex indices."""
        if face_indices.size == 0:
            return np.empty(0, dtype=np.int32)
        if self.all_tris:
            starts = face_indices * 3
            loop_idx = np.column_stack((starts, starts + 1, starts + 2)).ravel()
        else:
            starts = self.poly_starts[face_indices]
            totals = self.poly_totals[face_indices]
            total_loops = int(totals.sum())
            if total_loops == 0:
                return np.empty(0, dtype=np.int32)
            cum = np.cumsum(totals)
            group_offsets = np.repeat(starts - (cum - totals), totals)
            loop_idx = np.arange(total_loops, dtype=np.int32) + group_offsets
        return np.unique(self.loop_verts[loop_idx])

    def faces_to_vert_incident_coverage(self, face_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return unique vertex indices in face_indices and their incident face selection ratio in [0, 1]."""
        if face_indices.size == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

        if self.all_tris:
            starts = face_indices * 3
            loop_idx = np.column_stack((starts, starts + 1, starts + 2)).ravel()
        else:
            starts = self.poly_starts[face_indices]
            totals = self.poly_totals[face_indices]
            total_loops = int(totals.sum())
            if total_loops == 0:
                return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
            cum = np.cumsum(totals)
            loop_idx = np.arange(total_loops, dtype=np.int32) + np.repeat(starts - (cum - totals), totals)

        sel_verts = self.loop_verts[loop_idx]
        sel_counts = np.bincount(sel_verts, minlength=self.num_verts)
        total_counts = np.bincount(self.loop_verts, minlength=self.num_verts)
        uniq_verts = np.unique(sel_verts)
        ratios = sel_counts[uniq_verts].astype(np.float32) / np.maximum(total_counts[uniq_verts].astype(np.float32), 1.0)
        return uniq_verts, np.clip(ratios, 0.0, 1.0)

    def get_face_colors_linear(self) -> np.ndarray | None:
        """Return an (N_F, 4) linear RGBA proxy array per polygon for barrier checks."""
        if self.color_buf is None:
            return None
        if self.color_domain == 'CORNER':
            return self.color_buf[self.poly_starts]
        return self.color_buf[self.poly_first_verts]

    def get_face_tex_colors_linear(self) -> np.ndarray | None:
        """Sample the UV centroid pixel of each polygon in linear RGBA (cached for speed)."""
        if self.tex_buf is None or self.face_uv_centroids is None:
            return None
        if self.face_tex_colors is None:
            px = np.clip((self.face_uv_centroids[:, 0] * self.tex_width).astype(np.int32), 0, self.tex_width - 1)
            py = np.clip((self.face_uv_centroids[:, 1] * self.tex_height).astype(np.int32), 0, self.tex_height - 1)
            sampled = self.tex_buf[py, px].copy()
            if self.tex_is_srgb:
                sampled[:, :3] = srgb_to_linear(sampled[:, :3])
            self.face_tex_colors = sampled
        return self.face_tex_colors

    def get_face_color_at(self, face_idx: int, target_output: str) -> np.ndarray | None:
        """O(1) single-face linear RGBA lookup for stroke anchor color."""
        if face_idx < 0 or face_idx >= self.num_faces:
            return None
        if target_output == 'TEXTURE':
            if self.face_tex_colors is not None:
                return self.face_tex_colors[face_idx].copy()
            if self.tex_buf is None or self.face_uv_centroids is None:
                return None
            u, v = self.face_uv_centroids[face_idx]
            px = max(0, min(self.tex_width - 1, int(u * self.tex_width)))
            py = max(0, min(self.tex_height - 1, int(v * self.tex_height)))
            col = self.tex_buf[py, px].copy()
            if self.tex_is_srgb:
                col[:3] = srgb_to_linear(col[:3])
            return col
        if self.color_buf is not None:
            if self.color_domain == 'CORNER':
                return self.color_buf[self.poly_starts[face_idx]].copy()
            return self.color_buf[self.poly_first_verts[face_idx]].copy()
        return None

    def get_color_at_hit(self, face_idx: int, hit_pos: np.ndarray, target_output: str) -> np.ndarray | None:
        """Sample exact color at the raycast hit position (barycentric UV pixel or vertex color)."""
        if face_idx < 0 or face_idx >= self.num_faces:
            return None

        if target_output == 'TEXTURE':
            if self.tex_buf is None or self.tri_uvs is None:
                return self.get_face_color_at(face_idx, target_output)
            if self.all_tris:
                t_indices = np.array([face_idx], dtype=np.int32)
            else:
                t_indices = np.flatnonzero(self.tri_poly_idx == face_idx).astype(np.int32)
            if t_indices.size == 0:
                return self.get_face_color_at(face_idx, target_output)

            t_verts = self.tri_verts[t_indices]
            t_coords = self.vert_coords[t_verts] @ self.rot_scale_3x3.T
            t_centers = np.mean(t_coords, axis=1)
            best_k = int(np.argmin(np.sum((t_centers - hit_pos) ** 2, axis=1)))
            best_ti = t_indices[best_k]

            v0 = t_coords[best_k, 0]
            v1 = t_coords[best_k, 1]
            v2 = t_coords[best_k, 2]
            e0 = v1 - v0
            e1 = v2 - v0
            ep = hit_pos - v0
            d00 = float(np.dot(e0, e0))
            d01 = float(np.dot(e0, e1))
            d11 = float(np.dot(e1, e1))
            dp0 = float(np.dot(ep, e0))
            dp1 = float(np.dot(ep, e1))
            denom = d00 * d11 - d01 * d01
            if abs(denom) > 1e-8:
                b1 = (d11 * dp0 - d01 * dp1) / denom
                b2 = (d00 * dp1 - d01 * dp0) / denom
                b0 = 1.0 - b1 - b2
                b0 = max(0.0, min(1.0, b0))
                b1 = max(0.0, min(1.0, b1))
                b2 = max(0.0, min(1.0, b2))
                b_sum = b0 + b1 + b2
                if b_sum > 1e-6:
                    b0 /= b_sum
                    b1 /= b_sum
                    b2 /= b_sum
            else:
                b0, b1, b2 = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0

            uvs = self.tri_uvs[best_ti]
            uv = b0 * uvs[0] + b1 * uvs[1] + b2 * uvs[2]
            px = max(0, min(self.tex_width - 1, int(uv[0] * self.tex_width)))
            py = max(0, min(self.tex_height - 1, int(uv[1] * self.tex_height)))
            col = self.tex_buf[py, px].copy()
            if self.tex_is_srgb:
                col[:3] = srgb_to_linear(col[:3])
            return col

        if self.color_buf is not None:
            if self.color_domain == 'CORNER':
                if self.all_tris:
                    t_idx = face_idx
                    t_loops = t_idx * 3 + np.arange(3, dtype=np.int32)
                else:
                    t_matches = np.flatnonzero(self.tri_poly_idx == face_idx)
                    if t_matches.size > 0:
                        t_idx = t_matches[0]
                        t_loops = self.tri_loops[t_idx]
                    else:
                        t_loops = np.arange(self.poly_starts[face_idx], self.poly_starts[face_idx] + self.poly_totals[face_idx], dtype=np.int32)
                loop_v = self.loop_verts[t_loops]
                loop_coords = self.vert_coords[loop_v] @ self.rot_scale_3x3.T
                closest_k = int(np.argmin(np.sum((loop_coords - hit_pos) ** 2, axis=1)))
                return self.color_buf[t_loops[closest_k]].copy()
            else:
                if self.all_tris:
                    poly_v = self.loop_verts[face_idx * 3 : face_idx * 3 + 3]
                else:
                    poly_v = self.loop_verts[self.poly_starts[face_idx] : self.poly_starts[face_idx] + self.poly_totals[face_idx]]
                poly_coords = self.vert_coords[poly_v] @ self.rot_scale_3x3.T
                closest_k = int(np.argmin(np.sum((poly_coords - hit_pos) ** 2, axis=1)))
                return self.color_buf[poly_v[closest_k]].copy()

        return None

    def push_color_buffer(self):
        """Write cached color buffer to mesh.color_attributes.active and tag GPU update."""
        if self.color_attr is None:
            return
        if self.color_type == 'BYTE_COLOR' and self.srgb_buf is not None:
            self.color_attr.data.foreach_set("color_srgb", self.srgb_buf.ravel())
        else:
            self.color_attr.data.foreach_set("color", self.color_buf.ravel())

    def apply_color_indices(self, indices: np.ndarray, rgba_linear: tuple[float, float, float, float]):
        """Update color buffer at given element indices and push to Blender."""
        if indices.size == 0 or self.color_buf is None:
            return
        col = np.asarray(rgba_linear, dtype=np.float32)
        self.color_buf[indices] = col
        if self.srgb_buf is not None:
            srgb = np.empty(4, dtype=np.float32)
            srgb[:3] = linear_to_srgb(col[:3])
            srgb[3] = np.clip(col[3], 0.0, 1.0)
            self.srgb_buf[indices] = srgb
        self.push_color_buffer()

    def restore_color_indices(self, indices: np.ndarray, old_linear: np.ndarray):
        """Restore color buffer slice for undo and push to Blender."""
        if indices.size == 0 or self.color_buf is None:
            return
        self.color_buf[indices] = old_linear
        if self.srgb_buf is not None:
            srgb = np.empty_like(old_linear)
            srgb[:, :3] = linear_to_srgb(old_linear[:, :3])
            srgb[:, 3] = np.clip(old_linear[:, 3], 0.0, 1.0)
            self.srgb_buf[indices] = srgb
        self.push_color_buffer()

    def apply_face_set_indices(self, face_indices: np.ndarray, fs_id: int):
        """Update Sculpt Face Set buffer at given face indices and push to Blender."""
        if face_indices.size == 0 or self.face_set_buf is None:
            return
        self.face_set_buf[face_indices] = fs_id
        self.face_set_attr.data.foreach_set("value", self.face_set_buf)

    def restore_face_set_indices(self, face_indices: np.ndarray, old_fs: np.ndarray):
        """Restore Sculpt Face Set slice for undo and push to Blender."""
        if face_indices.size == 0 or self.face_set_buf is None:
            return
        self.face_set_buf[face_indices] = old_fs
        self.face_set_attr.data.foreach_set("value", self.face_set_buf)

    def _ensure_uv_tiles_for_coords(self, gy: np.ndarray, gx: np.ndarray):
        """Lazily rasterize only the individual 256x256 UV tiles touched by (gy, gx)."""
        if self._uv_cov_tris is None or self._uv_tiles_done is None or gy.size == 0:
            return
        w, h = self.tex_width, self.tex_height
        ts = self._uv_tile_size
        ny, nx = self._uv_tiles_done.shape
        t_keys = np.unique((gy // ts) * nx + (gx // ts))
        done_flat = self._uv_tiles_done.ravel()
        missing = t_keys[~done_flat[t_keys]]
        if missing.size == 0:
            return

        if self._uv_covered_mask is None:
            self._uv_covered_mask = np.zeros((h, w), dtype=bool)
        self._ensure_uv_tri_bounds()
        bx0, bx1, by0, by1 = self._uv_tri_px_bounds

        for tk in missing.tolist():
            ty = tk // nx
            tx = tk % nx
            rx0 = tx * ts
            rx1 = min(w, rx0 + ts)
            ry0 = ty * ts
            ry1 = min(h, ry0 + ts)
            overlap = (bx1 >= rx0) & (bx0 < rx1) & (by1 >= ry0) & (by0 < ry1)
            if np.any(overlap):
                self._uv_covered_mask[ry0:ry1, rx0:rx1] |= rasterize_uv_triangles(
                    self._uv_cov_tris[overlap],
                    w,
                    h,
                    x_offset=rx0,
                    y_offset=ry0,
                    out_w=rx1 - rx0,
                    out_h=ry1 - ry0,
                )
            done_flat[tk] = True

    def push_texture_buffer(self):
        """Push CPU texture buffer to Blender Image and tag depsgraph/GPU for live viewport refresh."""
        if self.tex_image is None or self.tex_buf_flat is None:
            return
        self.tex_image.pixels.foreach_set(self.tex_buf_flat.ravel())
        self.tex_image.update()
        if hasattr(self.tex_image, "gl_free"):
            self.tex_image.gl_free()
        if hasattr(self.tex_image, "update_tag"):
            self.tex_image.update_tag()
        if getattr(self, "mesh", None) is not None:
            self.mesh.update_gpu_tag()
            self.mesh.update()

    def apply_texture_faces(
        self,
        face_indices: np.ndarray,
        rgba_linear: tuple[float, float, float, float],
        preserve_existing: bool = False,
        stroke_tex_mask: np.ndarray | None = None,
        commit: bool = True,
        smooth_boundary: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize flooded faces onto the active Texture Paint image and return undo delta."""
        if face_indices.size == 0 or self.tex_buf is None or self.tri_uvs is None:
            return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)

        if self.all_tris:
            sel_uvs = self.tri_uvs[face_indices]
        else:
            self.in_mask_buf[face_indices] = True
            tri_mask = self.in_mask_buf[self.tri_poly_idx]
            sel_uvs = self.tri_uvs[tri_mask]
            self.in_mask_buf[face_indices] = False

        if sel_uvs.size == 0:
            return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)

        w, h = self.tex_width, self.tex_height
        pad = self.tex_seam_bleed + 1
        x_min = max(0, int(np.floor(sel_uvs[:, :, 0].min() * w)) - pad)
        x_max = min(w - 1, int(np.ceil(sel_uvs[:, :, 0].max() * w)) + pad)
        y_min = max(0, int(np.floor(sel_uvs[:, :, 1].min() * h)) - pad)
        y_max = min(h - 1, int(np.ceil(sel_uvs[:, :, 1].max() * h)) + pad)

        crop_w = x_max - x_min + 1
        crop_h = y_max - y_min + 1
        if crop_w <= 0 or crop_h <= 0:
            return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)

        crop_mask = rasterize_uv_triangles(
            sel_uvs, w, h, x_offset=x_min, y_offset=y_min, out_w=crop_w, out_h=crop_h
        )
        if not np.any(crop_mask):
            return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)

        # Dilate strictly on the thin fringe outside crop_mask into uncovered UV margin pixels.
        if self.tex_seam_bleed > 0:
            dilated = crop_mask.copy()
            for _ in range(self.tex_seam_bleed):
                nxt = dilated.copy()
                if crop_h > 1:
                    nxt[1:, :] |= dilated[:-1, :]
                    nxt[:-1, :] |= dilated[1:, :]
                if crop_w > 1:
                    nxt[:, 1:] |= dilated[:, :-1]
                    nxt[:, :-1] |= dilated[:, 1:]
                dilated = nxt
            fringe = dilated & ~crop_mask
            if np.any(fringe):
                fy, fx = np.nonzero(fringe)
                gy = y_min + fy.astype(np.int32)
                gx = x_min + fx.astype(np.int32)
                self._ensure_uv_tiles_for_coords(gy, gx)
                if self._uv_covered_mask is not None:
                    uncov = ~self._uv_covered_mask[gy, gx]
                    if np.any(uncov):
                        crop_mask[fy[uncov], fx[uncov]] = True
                else:
                    crop_mask |= fringe

        if preserve_existing and self.tex_protected_mask is not None:
            crop_mask &= ~self.tex_protected_mask[y_min : y_max + 1, x_min : x_max + 1]

        if not np.any(crop_mask):
            return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)

        ry, rx = np.nonzero(crop_mask)
        flat_idx = ((y_min + ry.astype(np.int32)) * w + (x_min + rx.astype(np.int32))).astype(np.int32)

        if flat_idx.size == 0:
            return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)

        weights = None
        if smooth_boundary and crop_h > 2 and crop_w > 2:
            m = crop_mask.astype(np.float32)
            cov = (
                0.25 * m[1:-1, 1:-1]
                + 0.125 * (m[:-2, 1:-1] + m[2:, 1:-1] + m[1:-1, :-2] + m[1:-1, 2:])
                + 0.0625 * (m[:-2, :-2] + m[:-2, 2:] + m[2:, :-2] + m[2:, 2:])
            )
            w_arr = np.ones((crop_h, crop_w), dtype=np.float32)
            w_arr[1:-1, 1:-1] = np.clip(cov * 1.5, 0.0, 1.0)
            weights = w_arr[ry, rx]

        if stroke_tex_mask is not None:
            valid_mask = ~stroke_tex_mask[flat_idx]
            flat_idx = flat_idx[valid_mask]
            if weights is not None:
                weights = weights[valid_mask]
            if flat_idx.size == 0:
                return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)
            stroke_tex_mask[flat_idx] = True

        old_pixels = self.tex_buf_flat[flat_idx].copy()
        if self.initial_tex_buf is None:
            self._tex_revert_history.append((flat_idx, old_pixels))

        write_lin = np.asarray(rgba_linear, dtype=np.float32)
        write_col = write_lin
        if self.tex_is_srgb:
            write_col = write_lin.copy()
            write_col[:3] = linear_to_srgb(write_col[:3])
        write_col = write_col.copy() if write_col is write_lin else write_col
        write_col[3] = np.clip(write_col[3], 0.0, 1.0)

        if weights is not None:
            w_fac = weights[:, None]
            self.tex_buf_flat[flat_idx] = old_pixels * (1.0 - w_fac) + write_col * w_fac
        else:
            self.tex_buf_flat[flat_idx] = write_col
        if self.face_tex_colors is not None:
            lin_upd = write_lin.copy()
            lin_upd[3] = write_col[3]
            self.face_tex_colors[face_indices] = lin_upd

        if commit:
            self.push_texture_buffer()
        return flat_idx, old_pixels

    def restore_texture_pixels(
        self,
        flat_indices: np.ndarray,
        old_pixels: np.ndarray,
        faces: np.ndarray | None = None,
    ):
        """Restore texture image pixels from undo delta."""
        if flat_indices.size == 0 or self.tex_buf_flat is None or self.tex_image is None:
            return
        self.tex_buf_flat[flat_indices] = old_pixels
        if self.face_tex_colors is not None:
            if faces is not None and faces.size > 0 and self.face_uv_centroids is not None:
                px = np.clip((self.face_uv_centroids[faces, 0] * self.tex_width).astype(np.int32), 0, self.tex_width - 1)
                py = np.clip((self.face_uv_centroids[faces, 1] * self.tex_height).astype(np.int32), 0, self.tex_height - 1)
                sampled = self.tex_buf[py, px].copy()
                if self.tex_is_srgb:
                    sampled[:, :3] = srgb_to_linear(sampled[:, :3])
                self.face_tex_colors[faces] = sampled
            else:
                self.face_tex_colors = None
        self.push_texture_buffer()

    def revert_all(self):
        """Revert all mesh attributes and texture pixels back to their pre-modal state."""
        if self.color_attr is not None and self.initial_color_buf is not None:
            self.color_buf[:] = self.initial_color_buf
            if self.srgb_buf is not None:
                self.srgb_buf[:, :3] = linear_to_srgb(self.initial_color_buf[:, :3])
                self.srgb_buf[:, 3] = np.clip(self.initial_color_buf[:, 3], 0.0, 1.0)
            self.push_color_buffer()

        if self.face_set_attr is not None:
            if not self.had_face_set_attr:
                self.mesh.attributes.remove(self.face_set_attr)
                self.face_set_attr = None
                self.face_set_buf = None
            elif self.initial_face_set_buf is not None:
                self.face_set_buf[:] = self.initial_face_set_buf
                self.face_set_attr.data.foreach_set("value", self.face_set_buf)

        if self.tex_image is not None:
            if self.initial_tex_buf is not None:
                self.tex_buf[:] = self.initial_tex_buf
                self.push_texture_buffer()
            elif self._tex_revert_history:
                for idx_arr, old_px in reversed(self._tex_revert_history):
                    self.tex_buf_flat[idx_arr] = old_px
                self._tex_revert_history.clear()
                self.push_texture_buffer()
            self.face_tex_colors = None

        self.mesh.update_gpu_tag()
        self.mesh.update()

    def raycast_world(self, ray_origin_world: Vector, ray_dir_world: Vector):
        """Raycast a world-space ray into local BVHTree, returning (hit_pos_scaled_local, hit_normal_world, face_idx)."""
        origin_local = self.inv_matrix_world @ ray_origin_world
        dir_local = (self.inv_matrix_world.to_3x3() @ ray_dir_world).normalized()
        loc_local, _norm_local, face_idx, _dist = self.bvh.ray_cast(origin_local, dir_local)
        if face_idx is None or face_idx < 0 or face_idx >= self.num_faces:
            return None, None, None
        if self.hidden_faces[face_idx]:
            return None, None, None
        hit_pos = np.asarray(self.matrix_world.to_3x3() @ loc_local, dtype=np.float32)
        hit_normal = self.face_normals[face_idx]
        return hit_pos, hit_normal, int(face_idx)

    def extract_preview_geometry(
        self,
        face_indices: np.ndarray,
        max_fill_tris: int = 50000,
        smooth_boundary: bool = False,
    ):
        """Extract local-space triangle vertices and boundary line segment vertices for GPU preview."""
        if face_indices.size == 0:
            return None, None

        in_mask = self.in_mask_buf
        in_mask[face_indices] = True

        tri_coords = None
        if self.all_tris:
            if face_indices.size <= max_fill_tris:
                sel_tris = self.tri_verts[face_indices]
                tri_coords = self.vert_coords[sel_tris].reshape((-1, 3))
        else:
            tri_mask = in_mask[self.tri_poly_idx]
            if int(np.count_nonzero(tri_mask)) <= max_fill_tris:
                sel_tris = self.tri_verts[tri_mask]
                tri_coords = self.vert_coords[sel_tris].reshape((-1, 3))

        edge_coords = None
        if self.all_tris:
            starts = face_indices * 3
            loop_idx = np.column_stack((starts, starts + 1, starts + 2)).ravel()
        else:
            starts = self.poly_starts[face_indices]
            totals = self.poly_totals[face_indices]
            total_loops = int(totals.sum())
            if total_loops > 0:
                cum = np.cumsum(totals)
                loop_idx = np.arange(total_loops, dtype=np.int32) + np.repeat(starts - (cum - totals), totals)
            else:
                loop_idx = np.empty(0, dtype=np.int32)

        if loop_idx.size > 0:
            f_edges = self.loop_edges[loop_idx]
            edge_counts = np.bincount(f_edges, minlength=self.num_edges)
            b_edges = np.flatnonzero(edge_counts == 1).astype(np.int32)
            if b_edges.size > 0:
                if smooth_boundary:
                    edge_coords = self._smooth_boundary_edges(b_edges)
                else:
                    edge_coords = self.vert_coords[self.edge_verts[b_edges]].reshape((-1, 3))

        in_mask[face_indices] = False
        return tri_coords, edge_coords

    def _smooth_boundary_edges(self, b_edges: np.ndarray, iterations: int = 2) -> np.ndarray:
        """Smooth boundary edge segments using localized neighborhood vertex averaging.

        Guarantees 100% of boundary edges are preserved with zero clipping on complex topologies,
        symmetry seams, non-manifold boundaries, or character meshes.
        """
        if b_edges.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        raw_edges = self.edge_verts[b_edges]
        if raw_edges.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Unique undirected boundary edges
        edges = np.unique(np.sort(raw_edges, axis=1), axis=0)
        n_e = len(edges)
        if n_e < 3:
            return self.vert_coords[edges].reshape((-1, 3))

        # Unique boundary vertices and local index mapping
        b_verts, inv = np.unique(edges, return_inverse=True)
        n_bv = len(b_verts)

        orig_coords = self.vert_coords[b_verts]
        b_coords = orig_coords.copy()
        vn = self.vert_normals[b_verts]

        edge_u = inv[:, 0]
        edge_v = inv[:, 1]

        degree = np.bincount(edge_u, minlength=n_bv) + np.bincount(edge_v, minlength=n_bv)
        valid_deg = np.maximum(degree.astype(np.float32), 1.0)

        # Multi-pass Laplacian smoothing on boundary vertex coordinates
        for _ in range(iterations):
            sum_pos = np.zeros((n_bv, 3), dtype=np.float32)
            for c in range(3):
                np.add.at(sum_pos[:, c], edge_u, b_coords[edge_v, c])
                np.add.at(sum_pos[:, c], edge_v, b_coords[edge_u, c])
            avg_nb = sum_pos / valid_deg[:, None]
            b_coords = 0.5 * b_coords + 0.5 * avg_nb

        # Prevent boundary vertices from sinking into the mesh interior:
        disp = b_coords - orig_coords
        inward_dot = np.sum(disp * vn, axis=1)
        b_coords = b_coords - np.minimum(inward_dot, 0.0)[:, None] * vn + vn * 0.0003

        p0 = b_coords[edge_u]
        p1 = b_coords[edge_v]
        edge_vn = 0.5 * (vn[edge_u] + vn[edge_v])
        pmid = 0.5 * (p0 + p1) + edge_vn * 0.0002

        # Subdivide each edge into 2 segments: (p0 -> pmid) and (pmid -> p1)
        # This doubles the resolution and rounds all stair-steps into smooth curves.
        smooth_coords = np.empty((n_e * 4, 3), dtype=np.float32)
        smooth_coords[0::4] = p0
        smooth_coords[1::4] = pmid
        smooth_coords[2::4] = pmid
        smooth_coords[3::4] = p1

        return smooth_coords
