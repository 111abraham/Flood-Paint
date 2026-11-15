# SPDX-License-Identifier: GPL-3.0-or-later

import math
import numpy as np

from .mesh_cache import MeshFloodCache, linear_to_srgb, srgb_to_linear
from .properties import get_mesh_active_color_attribute


def _run_wavefront_flood(
    cache: MeshFloodCache,
    seed_face: int,
    hit_pos: np.ndarray,
    radius: float,
    angle_deg: float,
    crease_resistance: float,
    preserve_existing: bool = False,
    target_output: str = 'COLOR',
    active_stroke_id: int = 0,
    face_stroke_ids: np.ndarray | None = None,
    anchor_color: np.ndarray | None = None,
    anchor_fs: int | None = None,
    color_tol: float = 0.025,
) -> np.ndarray:
    """Core vectorized localized Dijkstra propagation from seed_face within radius."""
    if seed_face < 0 or seed_face >= cache.num_faces:
        return np.empty(0, dtype=np.int32)
    if cache.hidden_faces[seed_face]:
        return np.empty(0, dtype=np.int32)
    if (
        target_output == 'TEXTURE'
        and cache.poly_mat_indices is not None
        and cache.restrict_mat_index is not None
        and int(cache.poly_mat_indices[seed_face]) != int(cache.restrict_mat_index)
    ):
        return np.empty(0, dtype=np.int32)

    radius = max(float(radius), 1e-5)
    r2 = radius * radius
    angle_rad = max(math.radians(float(angle_deg)), 1e-4)
    crease_resistance = max(0.0, min(1.0, float(crease_resistance)))
    hard_stop = crease_resistance >= 0.99
    penalty_scale = (crease_resistance / max(1.0 - crease_resistance, 0.02)) * 8.0
    min_step = max(radius * 1e-4, 1e-6)

    # Pre-resolve barrier arrays if Preserve Existing is enabled.
    face_colors = None
    face_sets = None
    if preserve_existing:
        if face_stroke_ids is not None:
            sid = int(face_stroke_ids[seed_face])
            if sid != 0 and sid != active_stroke_id:
                return np.empty(0, dtype=np.int32)

        if target_output == 'TEXTURE':
            face_colors = cache.get_face_tex_colors_linear()
        elif target_output in {'COLOR', 'BOTH'} or cache.color_buf is not None:
            face_colors = cache.get_face_colors_linear()

        if face_colors is not None:
            if anchor_color is None:
                anchor_color = face_colors[seed_face]
            elif face_stroke_ids is None or face_stroke_ids[seed_face] == 0:
                if float(np.max(np.abs(face_colors[seed_face, :3] - anchor_color[:3]))) > color_tol:
                    return np.empty(0, dtype=np.int32)

        if target_output in {'FACE_SET', 'BOTH'} or cache.face_set_buf is not None:
            face_sets = cache.face_set_buf
            if anchor_fs is None and face_sets is not None:
                anchor_fs = int(face_sets[seed_face])
            elif (
                anchor_fs is not None
                and face_sets is not None
                and (face_stroke_ids is None or face_stroke_ids[seed_face] == 0)
            ):
                if int(face_sets[seed_face]) != int(anchor_fs):
                    return np.empty(0, dtype=np.int32)

    cost_buf = cache.cost_buf
    adj_offsets = cache.adj_offsets
    adj_neighbors = cache.adj_neighbors
    adj_angles = cache.adj_angles
    adj_dists = cache.adj_dists
    face_centers = cache.face_centers
    hidden_faces = cache.hidden_faces

    cost_buf[seed_face] = 0.0
    frontier = np.array([seed_face], dtype=np.int32)
    touched = [frontier]

    steps = 0
    max_steps = 4096

    while frontier.size > 0 and steps < max_steps:
        steps += 1
        starts = adj_offsets[frontier]
        ends = adj_offsets[frontier + 1]
        counts = ends - starts
        total = int(counts.sum())
        if total == 0:
            break

        cum = np.cumsum(counts)
        group_offsets = np.repeat(starts - (cum - counts), counts)
        edge_idx = np.arange(total, dtype=np.int32) + group_offsets

        nbs = adj_neighbors[edge_idx]
        angles = adj_angles[edge_idx]

        # 1. Topology/threshold/spatial pruning mask before computing full traversal weights.
        ok = ~hidden_faces[nbs]
        if (
            target_output == 'TEXTURE'
            and cache.poly_mat_indices is not None
            and cache.restrict_mat_index is not None
        ):
            ok &= (cache.poly_mat_indices[nbs] == cache.restrict_mat_index)
        if hard_stop:
            ok &= (angles <= angle_rad)

        if not np.any(ok):
            break

        # Filter to spatial sphere around hit_pos.
        diff = face_centers[nbs] - hit_pos
        ok &= (np.einsum('ij,ij->i', diff, diff) <= r2)

        # Preserve-existing segment barriers.
        if preserve_existing:
            if face_stroke_ids is not None:
                nb_sids = face_stroke_ids[nbs]
                same_stroke = (nb_sids == active_stroke_id) & (active_stroke_id != 0)
                untouched = (nb_sids == 0)
                seg_ok = untouched
                if face_colors is not None and anchor_color is not None:
                    seg_ok = seg_ok & (np.max(np.abs(face_colors[nbs, :3] - anchor_color[:3]), axis=1) <= color_tol)
                if face_sets is not None and anchor_fs is not None:
                    seg_ok = seg_ok & (face_sets[nbs] == anchor_fs)
                ok &= (same_stroke | seg_ok)
            else:
                if face_colors is not None and anchor_color is not None:
                    ok &= (np.max(np.abs(face_colors[nbs, :3] - anchor_color[:3]), axis=1) <= color_tol)
                if face_sets is not None and anchor_fs is not None:
                    ok &= (face_sets[nbs] == anchor_fs)

        if not np.any(ok):
            break

        nbs_ok = nbs[ok]
        angles_ok = angles[ok]
        dists_ok = adj_dists[edge_idx[ok]]
        parent_costs_ok = np.repeat(cost_buf[frontier], counts)[ok]

        # 2. Normal-change weighted edge traversal cost.
        if crease_resistance <= 1e-4:
            step_costs = np.maximum(dists_ok, min_step)
        else:
            ratio = angles_ok / angle_rad
            if hard_stop:
                over = np.maximum(ratio - 1.0, 0.0)
                mult = np.where(ratio <= 1.0, 1.0, 1.0 + penalty_scale * (over * over + over))
            else:
                over = np.maximum(ratio - 1.0, 0.0)
                mult = np.where(
                    ratio <= 1.0,
                    1.0 + 0.5 * crease_resistance * (ratio * ratio),
                    1.0 + 0.5 * crease_resistance + penalty_scale * (over * over + over),
                )
            step_costs = np.maximum(dists_ok, min_step) * mult

        cand_cost = parent_costs_ok + step_costs
        valid = (cand_cost <= radius) & (cand_cost < cost_buf[nbs_ok])
        if not np.any(valid):
            break

        nbs_v = nbs_ok[valid]
        cand_v = cand_cost[valid]

        # 3. Deterministic minimum-cost deduplication per neighbor face.
        if nbs_v.size > 1:
            order = np.lexsort((cand_v, nbs_v))
            nbs_s = nbs_v[order]
            cand_s = cand_v[order]
            is_first = np.empty(nbs_s.shape[0], dtype=bool)
            is_first[0] = True
            is_first[1:] = nbs_s[1:] != nbs_s[:-1]
            uniq_n = nbs_s[is_first]
            uniq_c = cand_s[is_first]
        else:
            uniq_n = nbs_v
            uniq_c = cand_v

        improved = uniq_c < cost_buf[uniq_n]
        if not np.any(improved):
            break

        next_f = uniq_n[improved]
        next_c = uniq_c[improved]
        newly_touched = next_f[np.isinf(cost_buf[next_f])]
        if newly_touched.size > 0:
            touched.append(newly_touched)
        cost_buf[next_f] = next_c
        frontier = next_f

    all_faces = np.concatenate(touched)
    cost_buf[all_faces] = np.inf
    return all_faces


def _spatial_sphere_faces(
    cache: MeshFloodCache,
    seed_face: int,
    hit_pos: np.ndarray,
    radius: float,
    target_output: str = 'COLOR',
) -> np.ndarray:
    """Fast unweighted BFS gathering all connected faces within Euclidean radius around hit_pos."""
    if seed_face < 0 or seed_face >= cache.num_faces:
        return np.empty(0, dtype=np.int32)
    if cache.hidden_faces[seed_face]:
        return np.empty(0, dtype=np.int32)
    if (
        target_output == 'TEXTURE'
        and cache.poly_mat_indices is not None
        and cache.restrict_mat_index is not None
        and int(cache.poly_mat_indices[seed_face]) != int(cache.restrict_mat_index)
    ):
        return np.empty(0, dtype=np.int32)

    radius = max(float(radius), 1e-5)
    r2 = radius * radius
    cost_buf = cache.cost_buf
    adj_offsets = cache.adj_offsets
    adj_neighbors = cache.adj_neighbors
    face_centers = cache.face_centers
    hidden_faces = cache.hidden_faces
    hit_center = np.asarray(hit_pos, dtype=np.float32)

    restrict_mat = (
        cache.poly_mat_indices is not None
        and cache.restrict_mat_index is not None
        and target_output == 'TEXTURE'
    )
    req_mat = int(cache.restrict_mat_index) if restrict_mat else -1

    cost_buf[seed_face] = 0.0
    frontier = np.array([seed_face], dtype=np.int32)
    touched = [frontier]

    steps = 0
    max_steps = 8192
    while frontier.size > 0 and steps < max_steps:
        steps += 1
        starts = adj_offsets[frontier]
        ends = adj_offsets[frontier + 1]
        counts = ends - starts
        total_edges = int(counts.sum())
        if total_edges == 0:
            break

        if frontier.size == 1:
            edge_idx = np.arange(starts[0], ends[0], dtype=np.int32)
        else:
            cum = np.cumsum(counts)
            base = cum - counts
            offsets_in_seg = np.arange(total_edges, dtype=np.int32) - np.repeat(base, counts)
            edge_idx = np.repeat(starts, counts) + offsets_in_seg

        nbs = adj_neighbors[edge_idx]
        valid = np.isinf(cost_buf[nbs]) & (~hidden_faces[nbs])
        if restrict_mat:
            valid &= (cache.poly_mat_indices[nbs] == req_mat)
        if not np.any(valid):
            break

        nbs_v = nbs[valid]
        diff = face_centers[nbs_v] - hit_center
        in_rad = np.einsum("ij,ij->i", diff, diff) <= r2
        if not np.any(in_rad):
            break

        nbs_r = np.unique(nbs_v[in_rad])
        cost_buf[nbs_r] = 0.0
        touched.append(nbs_r)
        frontier = nbs_r

    all_faces = np.concatenate(touched)
    cost_buf[all_faces] = np.inf
    return all_faces


def compute_flood_faces(
    cache: MeshFloodCache,
    seed_face: int,
    hit_pos: np.ndarray,
    radius: float,
    angle_deg: float,
    crease_resistance: float,
    preserve_existing: bool = False,
    target_output: str = 'COLOR',
    active_stroke_id: int = 0,
    face_stroke_ids: np.ndarray | None = None,
    anchor_color: np.ndarray | None = None,
    anchor_fs: int | None = None,
    color_tol: float = 0.025,
    invert_flood: bool = False,
    smooth_tool: bool = False,
    boundary_offset: int = 0,
) -> np.ndarray:
    """Propagate a normal-aware localized wavefront from seed_face within radius.

    When ``invert_flood=True``, Gap-Fill painted-boundary detection is enabled automatically:
    it detects painted regions inside the brush radius (stroke IDs, Face Sets, or painted colors
    bordering unpainted canvas) and inverts that selection to fill only the trapped/unpainted
    middle faces inside the cursor. If no painted regions exist inside the brush radius, it
    inverts the normal-aware wavefront across sharp angle creases inside the brush radius.

    When ``smooth_tool=True``, boundary stair-stepping and jagged corners are smoothed.
    When ``boundary_offset != 0``, the boundary is expanded (> 0) or contracted (< 0) by face rings.
    """
    if not invert_flood:
        res = _run_wavefront_flood(
            cache=cache,
            seed_face=seed_face,
            hit_pos=hit_pos,
            radius=radius,
            angle_deg=angle_deg,
            crease_resistance=crease_resistance,
            preserve_existing=preserve_existing,
            target_output=target_output,
            active_stroke_id=active_stroke_id,
            face_stroke_ids=face_stroke_ids,
            anchor_color=anchor_color,
            anchor_fs=anchor_fs,
            color_tol=color_tol,
        )
        if boundary_offset != 0 and res.size > 0:
            res = offset_flood_boundary_faces(cache, res, boundary_offset)
        return res

    # Inverse + implicit Gap-Fill mode: gather all connected faces inside Euclidean brush radius.
    sphere_faces = _spatial_sphere_faces(
        cache=cache,
        seed_face=seed_face,
        hit_pos=hit_pos,
        radius=radius,
        target_output=target_output,
    )
    if sphere_faces.size == 0:
        return sphere_faces

    # Detect already-painted boundary regions inside sphere_faces (excluding active stroke).
    painted = np.zeros(sphere_faces.size, dtype=bool)
    sids = face_stroke_ids[sphere_faces] if face_stroke_ids is not None else None
    active_mask = (
        (sids == active_stroke_id)
        if (sids is not None and active_stroke_id != 0)
        else np.zeros(sphere_faces.size, dtype=bool)
    )
    if sids is not None:
        painted |= (sids != 0) & (~active_mask)

    if target_output in {'FACE_SET', 'BOTH'} and cache.face_set_buf is not None:
        painted |= (cache.face_set_buf[sphere_faces] != 1) & (~active_mask)

    fc = None
    if target_output == 'TEXTURE':
        fc = cache.get_face_tex_colors_linear()
    elif target_output in {'COLOR', 'BOTH'}:
        fc = cache.get_face_colors_linear()

    if fc is not None:
        is_white = np.max(np.abs(fc[sphere_faces, :3] - 1.0), axis=1) <= color_tol
        non_white = (~is_white) & (~active_mask)
        if np.any(non_white) and (np.any(is_white) or np.any(active_mask)):
            painted |= non_white

    # If painted boundaries exist inside cursor, invert the painted boundary selection:
    # protect all painted walls and select all trapped/unpainted middle faces inside cursor.
    if np.any(painted):
        res = sphere_faces[~painted]
        if boundary_offset != 0 and res.size > 0:
            res = offset_flood_boundary_faces(cache, res, boundary_offset)
        return res

    # Blank/unpainted region fallback: invert the normal-angle flood across creases inside radius.
    normal_faces = _run_wavefront_flood(
        cache=cache,
        seed_face=seed_face,
        hit_pos=hit_pos,
        radius=radius,
        angle_deg=angle_deg,
        crease_resistance=crease_resistance,
        preserve_existing=False,
        target_output=target_output,
    )
    in_mask = cache.in_mask_buf
    in_mask[normal_faces] = True
    rev_faces = sphere_faces[~in_mask[sphere_faces]]
    in_mask[normal_faces] = False
    if boundary_offset != 0 and rev_faces.size > 0:
        rev_faces = offset_flood_boundary_faces(cache, rev_faces, boundary_offset)
    return rev_faces


def offset_flood_boundary_faces(
    cache: MeshFloodCache,
    faces: np.ndarray,
    offset: int,
) -> np.ndarray:
    """Expand (offset > 0) or contract (offset < 0) the flood face selection by |offset| polygon rings."""
    if faces.size == 0 or offset == 0 or cache.adj_offsets is None or cache.adj_neighbors is None:
        return faces

    adj_offsets = cache.adj_offsets
    adj_neighbors = cache.adj_neighbors
    num_faces = cache.num_faces

    in_mask = np.zeros(num_faces, dtype=bool)
    in_mask[faces] = True

    if offset > 0:
        for _ in range(offset):
            sel_faces = np.flatnonzero(in_mask).astype(np.int32)
            if sel_faces.size == 0:
                break
            starts = adj_offsets[sel_faces]
            ends = adj_offsets[sel_faces + 1]
            counts = ends - starts
            total = int(counts.sum())
            if total == 0:
                break
            cum = np.cumsum(counts)
            edge_idx = np.arange(total, dtype=np.int32) + np.repeat(starts - (cum - counts), counts)
            nbs = adj_neighbors[edge_idx]
            valid_nbs = nbs[~cache.hidden_faces[nbs]]
            in_mask[valid_nbs] = True
    else:
        for _ in range(abs(offset)):
            sel_faces = np.flatnonzero(in_mask).astype(np.int32)
            if sel_faces.size == 0:
                break
            starts = adj_offsets[sel_faces]
            ends = adj_offsets[sel_faces + 1]
            counts = ends - starts
            total = int(counts.sum())
            if total == 0:
                break
            cum = np.cumsum(counts)
            face_idx = np.repeat(sel_faces, counts)
            edge_idx = np.arange(total, dtype=np.int32) + np.repeat(starts - (cum - counts), counts)
            nbs = adj_neighbors[edge_idx]
            nb_sel = np.bincount(face_idx, weights=in_mask[nbs], minlength=num_faces)
            boundary_faces = sel_faces[nb_sel[sel_faces] < counts]
            in_mask[boundary_faces] = False

    return np.flatnonzero(in_mask).astype(np.int32)


def smooth_flood_boundary_faces(
    cache: MeshFloodCache,
    faces: np.ndarray,
    hit_pos: np.ndarray,
    radius: float,
    iterations: int = 1,
) -> np.ndarray:
    """Smooth the jagged stair-stepped boundary of the flood face selection.

    Prunes isolated single-quad stair bumps and fills single-quad stair notches within the brush radius.
    """
    if faces.size == 0 or cache.adj_offsets is None or cache.adj_neighbors is None:
        return faces

    r2 = (float(radius) * 1.05) ** 2
    in_mask = np.zeros(cache.num_faces, dtype=bool)
    in_mask[faces] = True

    diff = cache.face_centers - hit_pos
    in_radius = (np.einsum('ij,ij->i', diff, diff) <= r2) & (~cache.hidden_faces)

    adj_offsets = cache.adj_offsets
    adj_neighbors = cache.adj_neighbors

    for _ in range(iterations):
        starts = adj_offsets[:-1]
        ends = adj_offsets[1:]
        counts = ends - starts
        total = int(counts.sum())
        if total == 0:
            break
        cum = np.cumsum(counts)
        face_idx = np.repeat(np.arange(cache.num_faces, dtype=np.int32), counts)
        edge_idx = np.arange(total, dtype=np.int32) + np.repeat(starts - (cum - counts), counts)
        nbs = adj_neighbors[edge_idx]

        nb_sel = np.bincount(face_idx, weights=in_mask[nbs], minlength=cache.num_faces)
        nb_total = np.maximum(counts.astype(np.float32), 1.0)
        ratio = nb_sel / nb_total

        prune = in_mask & in_radius & (ratio <= 0.35) & (counts >= 3)
        fill = (~in_mask) & in_radius & (ratio >= 0.65) & (counts >= 3)

        if not np.any(prune) and not np.any(fill):
            break

        in_mask[prune] = False
        in_mask[fill] = True

    return np.flatnonzero(in_mask).astype(np.int32)


def find_unpainted_faces(
    cache: MeshFloodCache,
    target_output: str = 'COLOR',
    face_stroke_ids: np.ndarray | None = None,
    base_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    color_tol: float = 0.025,
) -> np.ndarray:
    """Return indices of remaining unpainted faces (matching canvas white / Face Set 1)."""
    ok = ~cache.hidden_faces
    if (
        target_output == 'TEXTURE'
        and cache.poly_mat_indices is not None
        and cache.restrict_mat_index is not None
    ):
        ok &= (cache.poly_mat_indices == cache.restrict_mat_index)

    if face_stroke_ids is not None:
        ok &= (face_stroke_ids == 0)

    base = np.asarray(base_rgb, dtype=np.float32)
    if target_output == 'TEXTURE':
        fc = cache.get_face_tex_colors_linear()
        if fc is not None:
            ok &= (np.max(np.abs(fc[:, :3] - base), axis=1) <= color_tol)
    elif target_output in {'COLOR', 'BOTH'} and cache.color_buf is not None:
        fc = cache.get_face_colors_linear()
        if fc is not None:
            ok &= (np.max(np.abs(fc[:, :3] - base), axis=1) <= color_tol)

    if target_output in {'FACE_SET', 'BOTH'} and cache.face_set_buf is not None:
        ok &= (cache.face_set_buf == 1)

    return np.flatnonzero(ok).astype(np.int32)


def smooth_jagged_borders(mesh, iterations: int = 2, strength: float = 0.75, color_threshold: float = 0.015) -> int:
    """Run fast vectorized neighborhood color averaging strictly along color boundaries.

    Returns the number of modified color attribute elements.
    """
    ca = get_mesh_active_color_attribute(mesh)
    if ca is None:
        return 0

    nv = len(mesh.vertices)
    ne = len(mesh.edges)
    nl = len(mesh.loops)
    n_elems = len(ca.data)
    if nv == 0 or n_elems == 0:
        return 0

    colors_flat = np.empty(n_elems * 4, dtype=np.float32)
    ca.data.foreach_get("color", colors_flat)
    colors = colors_flat.reshape((n_elems, 4))

    edge_verts = np.empty(ne * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edge_verts)
    edge_verts = edge_verts.reshape((ne, 2))
    e0 = edge_verts[:, 0] if ne > 0 else np.empty(0, dtype=np.int32)
    e1 = edge_verts[:, 1] if ne > 0 else np.empty(0, dtype=np.int32)

    strength = max(0.0, min(1.0, float(strength)))
    iterations = max(1, int(iterations))
    modified_mask = np.zeros(n_elems, dtype=bool)

    if ca.domain == 'CORNER':
        loop_verts = np.empty(nl, dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_verts)
        v_counts = np.maximum(np.bincount(loop_verts, minlength=nv).astype(np.float32), 1.0)

        for _ in range(iterations):
            # Average incident corner colors onto each vertex.
            v_mean = np.empty((nv, 3), dtype=np.float32)
            for c in range(3):
                v_mean[:, c] = np.bincount(loop_verts, weights=colors[:, c], minlength=nv) / v_counts

            # Boundary vertices: corner disagreement at vertex OR color step across an incident edge.
            loop_diff = np.max(np.abs(colors[:, :3] - v_mean[loop_verts]), axis=1)
            bvert = np.bincount(loop_verts, weights=(loop_diff > color_threshold), minlength=nv) > 0
            if ne > 0:
                edge_diff = np.max(np.abs(v_mean[e0] - v_mean[e1]), axis=1) > color_threshold
                if np.any(edge_diff):
                    bvert[e0[edge_diff]] = True
                    bvert[e1[edge_diff]] = True

            if not np.any(bvert):
                break

            # Smooth only boundary vertices using their 1-ring edge neighbors.
            if ne > 0:
                be = bvert[e0] | bvert[e1]
                if np.any(be):
                    be0 = e0[be]
                    be1 = e1[be]
                    v_sum = v_mean.copy()
                    v_cnt = np.ones(nv, dtype=np.float32)
                    v_cnt += np.bincount(be0, minlength=nv) + np.bincount(be1, minlength=nv)
                    for c in range(3):
                        v_sum[:, c] += np.bincount(be0, weights=v_mean[be1, c], minlength=nv)
                        v_sum[:, c] += np.bincount(be1, weights=v_mean[be0, c], minlength=nv)
                    v_mean[bvert] = v_sum[bvert] / v_cnt[bvert, None]

            bloop = bvert[loop_verts]
            modified_mask |= bloop
            colors[bloop, :3] = (
                (1.0 - strength) * colors[bloop, :3]
                + strength * v_mean[loop_verts[bloop]]
            )

    else:
        # POINT domain: smooth vertices along color-step edges.
        if ne == 0:
            return 0

        for _ in range(iterations):
            edge_diff = np.max(np.abs(colors[e0, :3] - colors[e1, :3]), axis=1) > color_threshold
            if not np.any(edge_diff):
                break
            bvert = np.zeros(nv, dtype=bool)
            bvert[e0[edge_diff]] = True
            bvert[e1[edge_diff]] = True

            be = bvert[e0] | bvert[e1]
            be0 = e0[be]
            be1 = e1[be]
            v_sum = colors[:, :3].copy()
            v_cnt = np.ones(nv, dtype=np.float32)
            v_cnt += np.bincount(be0, minlength=nv) + np.bincount(be1, minlength=nv)
            for c in range(3):
                v_sum[:, c] += np.bincount(be0, weights=colors[be1, c], minlength=nv)
                v_sum[:, c] += np.bincount(be1, weights=colors[be0, c], minlength=nv)
            v_smooth = v_sum[bvert] / v_cnt[bvert, None]

            modified_mask |= bvert
            colors[bvert, :3] = (1.0 - strength) * colors[bvert, :3] + strength * v_smooth

    mod_count = int(np.count_nonzero(modified_mask))
    if mod_count > 0:
        if ca.data_type == 'BYTE_COLOR':
            srgb = np.empty_like(colors)
            srgb[:, :3] = linear_to_srgb(colors[:, :3])
            srgb[:, 3] = np.clip(colors[:, 3], 0.0, 1.0)
            ca.data.foreach_set("color_srgb", srgb.ravel())
        else:
            ca.data.foreach_set("color", colors.ravel())
        mesh.update_gpu_tag()
        mesh.update()

    return mod_count


def smooth_jagged_texture_borders(
    image,
    uv_covered_mask: np.ndarray | None = None,
    iterations: int = 2,
    strength: float = 0.75,
    color_threshold: float = 0.02,
    seam_bleed: int = 2,
) -> int:
    """Smooth high-contrast color borders in a Texture Paint image via vectorized NumPy.

    When ``uv_covered_mask`` (shape ``(H, W)`` bool) is supplied, color smoothing is
    strictly confined to covered UV pixels so empty UV margin background does not wash
    out UV seams, and modified border pixels are re-dilated into uncovered seam margin.
    Returns the number of modified covered pixels.
    """
    if image is None or image.size[0] <= 0 or image.size[1] <= 0:
        return 0

    width, height = int(image.size[0]), int(image.size[1])
    total_px = width * height
    flat = np.empty(total_px * 4, dtype=np.float32)
    image.pixels.foreach_get(flat)
    buf = flat.reshape((height, width, 4))

    cs_name = getattr(getattr(image, "colorspace_settings", None), "name", "sRGB")
    is_srgb = "srgb" in cs_name.lower()
    work = srgb_to_linear(buf[:, :, :3]) if is_srgb else buf[:, :, :3].copy()

    cov = uv_covered_mask
    if cov is not None and cov.shape != (height, width):
        cov = None

    strength = max(0.0, min(1.0, float(strength)))
    iterations = max(1, int(iterations))
    threshold = float(color_threshold)
    modified_mask = np.zeros((height, width), dtype=bool)

    half_pairs = ((1, 0), (0, 1), (1, 1), (1, -1))
    neighbors = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 0.70710677), (-1, 1, 0.70710677),
        (1, -1, 0.70710677), (1, 1, 0.70710677),
    )

    for _ in range(iterations):
        is_boundary = np.zeros((height, width), dtype=bool)
        for dy, dx in half_pairs:
            y0_a = max(0, -dy)
            y1_a = height - max(0, dy)
            x0_a = max(0, -dx)
            x1_a = width - max(0, dx)

            y0_b = max(0, dy)
            y1_b = height - max(0, -dy)
            x0_b = max(0, dx)
            x1_b = width - max(0, -dx)

            diff = np.max(
                np.abs(work[y0_a:y1_a, x0_a:x1_a] - work[y0_b:y1_b, x0_b:x1_b]),
                axis=2,
            ) > threshold
            if cov is not None:
                diff &= cov[y0_a:y1_a, x0_a:x1_a] & cov[y0_b:y1_b, x0_b:x1_b]
            is_boundary[y0_a:y1_a, x0_a:x1_a] |= diff
            is_boundary[y0_b:y1_b, x0_b:x1_b] |= diff

        by, bx = np.nonzero(is_boundary)
        if by.size == 0:
            break

        modified_mask[by, bx] = True
        nb_sum = work[by, bx].copy()
        nb_w = np.ones(by.size, dtype=np.float32)

        for dy, dx, wt in neighbors:
            ny = by + dy
            nx = bx + dx
            ok = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
            if cov is not None:
                ok_idx = np.nonzero(ok)[0]
                ok[ok_idx] = cov[ny[ok_idx], nx[ok_idx]]
            if not np.any(ok):
                continue
            nb_sum[ok] += work[ny[ok], nx[ok]] * wt
            nb_w[ok] += wt

        work[by, bx] = (1.0 - strength) * work[by, bx] + strength * (nb_sum / nb_w[:, None])

    mod_count = int(np.count_nonzero(modified_mask))
    if mod_count == 0:
        return 0

    buf[:, :, :3] = np.clip(linear_to_srgb(work) if is_srgb else work, 0.0, 1.0)

    # Propagate smoothed UV edge colors into the seam-bleed margin around modified pixels.
    if cov is not None and seam_bleed > 0:
        dilated_mask = cov.copy()
        fy, fx = np.nonzero(modified_mask)
        cur_y = fy.astype(np.int32)
        cur_x = fx.astype(np.int32)
        cur_px = buf[cur_y, cur_x]
        for _ in range(int(seam_bleed)):
            if cur_y.size == 0:
                break
            cand_y_list = []
            cand_x_list = []
            cand_p_list = []
            for dy, dx, _ in neighbors[:4]:
                ny = cur_y + dy
                nx = cur_x + dx
                inb = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
                if not np.any(inb):
                    continue
                ny_i = ny[inb]
                nx_i = nx[inb]
                need = ~dilated_mask[ny_i, nx_i]
                if np.any(need):
                    cand_y_list.append(ny_i[need])
                    cand_x_list.append(nx_i[need])
                    cand_p_list.append(cur_px[inb][need])
            if not cand_y_list:
                break
            ny_all = np.concatenate(cand_y_list)
            nx_all = np.concatenate(cand_x_list)
            p_all = np.concatenate(cand_p_list)
            flat_id = ny_all.astype(np.int64) * width + nx_all.astype(np.int64)
            _, uniq_idx = np.unique(flat_id, return_index=True)
            cur_y = ny_all[uniq_idx]
            cur_x = nx_all[uniq_idx]
            cur_px = p_all[uniq_idx]
            dilated_mask[cur_y, cur_x] = True
            buf[cur_y, cur_x] = cur_px

    image.pixels.foreach_set(buf.ravel())
    if hasattr(image, "update"):
        image.update()
    try:
        image.gl_free()
    except Exception:
        pass
    if hasattr(image, "update_tag"):
        image.update_tag()
    return mod_count


def smooth_local_vertex_paint(
    cache: MeshFloodCache,
    hit_pos: np.ndarray | None = None,
    radius: float | None = None,
    iterations: int = 2,
    strength: float = 0.75,
    color_threshold: float = 0.015,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth jagged color attribute boundaries locally around hit_pos (or mesh-wide if hit_pos is None).

    Operates directly on cached NumPy buffers with zero reallocation overhead.
    Returns:
        (modified_indices, old_colors) where modified_indices is an (M,) int32 array and
        old_colors is an (M, 4) float32 array of their previous linear RGBA values.
    """
    if cache.color_buf is None or cache.color_attr is None:
        return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)

    nv = cache.num_verts
    ne = cache.num_edges
    nl = cache.num_loops
    colors = cache.color_buf
    n_elems = len(colors)
    if nv == 0 or n_elems == 0:
        return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)

    # Spatial sphere filter around hit_pos in metric space.
    if hit_pos is not None and radius is not None and radius > 0.0:
        vert_pos_metric = cache.vert_coords @ cache.rot_scale_3x3.T
        diff = vert_pos_metric - hit_pos
        dist_sq = np.sum(diff * diff, axis=1)
        r_limit = float(radius) * 1.05
        in_radius = dist_sq <= (r_limit * r_limit)
        if not np.any(in_radius):
            return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)
    else:
        in_radius = np.ones(nv, dtype=bool)

    strength = max(0.0, min(1.0, float(strength)))
    iterations = max(1, int(iterations))
    threshold = float(color_threshold)
    work_colors = colors.copy()

    e0 = cache.edge_verts[:, 0] if ne > 0 else np.empty(0, dtype=np.int32)
    e1 = cache.edge_verts[:, 1] if ne > 0 else np.empty(0, dtype=np.int32)

    if cache.color_domain == 'CORNER':
        loop_verts = cache.loop_verts
        v_counts = np.maximum(np.bincount(loop_verts, minlength=nv).astype(np.float32), 1.0)

        for _ in range(iterations):
            # Average incident corner colors onto each vertex.
            v_mean = np.empty((nv, 3), dtype=np.float32)
            for c in range(3):
                v_mean[:, c] = np.bincount(loop_verts, weights=work_colors[:, c], minlength=nv) / v_counts

            # Boundary vertices: corner disagreement at vertex OR color step across an incident edge.
            loop_diff = np.max(np.abs(work_colors[:, :3] - v_mean[loop_verts]), axis=1)
            bvert = np.bincount(loop_verts, weights=(loop_diff > threshold), minlength=nv) > 0
            if ne > 0:
                edge_diff = np.max(np.abs(v_mean[e0] - v_mean[e1]), axis=1) > threshold
                if np.any(edge_diff):
                    bvert[e0[edge_diff]] = True
                    bvert[e1[edge_diff]] = True

            bvert &= in_radius
            if not np.any(bvert):
                break

            # Smooth only boundary vertices using their 1-ring edge neighbors.
            if ne > 0:
                be = bvert[e0] | bvert[e1]
                if np.any(be):
                    be0 = e0[be]
                    be1 = e1[be]
                    v_sum = v_mean.copy()
                    v_cnt = np.ones(nv, dtype=np.float32)
                    v_cnt += np.bincount(be0, minlength=nv) + np.bincount(be1, minlength=nv)
                    for c in range(3):
                        v_sum[:, c] += np.bincount(be0, weights=v_mean[be1, c], minlength=nv)
                        v_sum[:, c] += np.bincount(be1, weights=v_mean[be0, c], minlength=nv)
                    v_mean[bvert] = v_sum[bvert] / v_cnt[bvert, None]

            bloop = bvert[loop_verts]
            work_colors[bloop, :3] = (
                (1.0 - strength) * work_colors[bloop, :3]
                + strength * v_mean[loop_verts[bloop]]
            )

    else:
        # POINT domain: smooth vertices along color-step edges.
        if ne == 0:
            return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)

        for _ in range(iterations):
            edge_diff = np.max(np.abs(work_colors[e0, :3] - work_colors[e1, :3]), axis=1) > threshold
            if not np.any(edge_diff):
                break
            bvert = np.zeros(nv, dtype=bool)
            bvert[e0[edge_diff]] = True
            bvert[e1[edge_diff]] = True
            bvert &= in_radius
            if not np.any(bvert):
                break

            be = bvert[e0] | bvert[e1]
            be0 = e0[be]
            be1 = e1[be]
            v_sum = work_colors[:, :3].copy()
            v_cnt = np.ones(nv, dtype=np.float32)
            v_cnt += np.bincount(be0, minlength=nv) + np.bincount(be1, minlength=nv)
            for c in range(3):
                v_sum[:, c] += np.bincount(be0, weights=work_colors[be1, c], minlength=nv)
                v_sum[:, c] += np.bincount(be1, weights=work_colors[be0, c], minlength=nv)
            v_smooth = v_sum[bvert] / v_cnt[bvert, None]

            work_colors[bvert, :3] = (1.0 - strength) * work_colors[bvert, :3] + strength * v_smooth

    diff_mask = np.any(np.abs(work_colors[:, :3] - colors[:, :3]) > 1e-5, axis=1)
    mod_indices = np.flatnonzero(diff_mask).astype(np.int32)
    if mod_indices.size == 0:
        return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)

    old_colors = colors[mod_indices].copy()
    colors[mod_indices] = work_colors[mod_indices]
    if cache.srgb_buf is not None:
        srgb = np.empty_like(work_colors[mod_indices])
        srgb[:, :3] = linear_to_srgb(work_colors[mod_indices, :3])
        srgb[:, 3] = np.clip(work_colors[mod_indices, 3], 0.0, 1.0)
        cache.srgb_buf[mod_indices] = srgb

    cache.push_color_buffer()
    cache.mesh.update_gpu_tag()
    cache.mesh.update()

    return mod_indices, old_colors


def smooth_local_texture_paint(
    cache: MeshFloodCache,
    hit_pos: np.ndarray | None = None,
    radius: float | None = None,
    iterations: int = 2,
    strength: float = 0.75,
    color_threshold: float = 0.02,
    seam_bleed: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth jagged texture paint borders locally around hit_pos or image-wide.

    Returns:
        (modified_pixels_flat, old_pixels_linear)
    """
    if cache.tex_buf is None or cache.tex_image is None:
        return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)

    w, h = cache.tex_width, cache.tex_height
    if w <= 0 or h <= 0:
        return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)

    old_tex = cache.tex_buf.copy()

    # Determine region of interest mask
    cov = cache.uv_covered_mask
    roi_mask = None
    if hit_pos is not None and radius is not None and radius > 0.0 and cache.tri_uvs is not None:
        vert_pos_metric = cache.vert_coords @ cache.rot_scale_3x3.T
        diff = vert_pos_metric - hit_pos
        dist_sq = np.sum(diff * diff, axis=1)
        r_limit = float(radius) * 1.05
        in_radius = dist_sq <= (r_limit * r_limit)
        if cache.all_tris:
            tri_in_r = (
                in_radius[cache.tri_verts[:, 0]]
                | in_radius[cache.tri_verts[:, 1]]
                | in_radius[cache.tri_verts[:, 2]]
            )
            sel_uvs = cache.tri_uvs[tri_in_r]
        else:
            poly_in_r = np.zeros(cache.num_faces, dtype=bool)
            for f in range(cache.num_faces):
                s = cache.poly_starts[f]
                t = cache.poly_totals[f]
                poly_in_r[f] = np.any(in_radius[cache.loop_verts[s : s + t]])
            tri_in_r = poly_in_r[cache.tri_poly_idx]
            sel_uvs = cache.tri_uvs[tri_in_r]

        if sel_uvs.size > 0:
            pad = max(1, int(seam_bleed) + 2)
            u_min = max(0, int(np.floor(sel_uvs[:, :, 0].min() * w)) - pad)
            u_max = min(w - 1, int(np.ceil(sel_uvs[:, :, 0].max() * w)) + pad)
            v_min = max(0, int(np.floor(sel_uvs[:, :, 1].min() * h)) - pad)
            v_max = min(h - 1, int(np.ceil(sel_uvs[:, :, 1].max() * h)) + pad)
            roi_mask = np.zeros((h, w), dtype=bool)
            roi_mask[v_min : v_max + 1, u_min : u_max + 1] = True
            if cov is not None:
                roi_mask &= cov

    if roi_mask is None:
        roi_mask = cov

    # Run texture border smoothing
    mod_count = smooth_jagged_texture_borders(
        image=cache.tex_image,
        uv_covered_mask=roi_mask,
        iterations=iterations,
        strength=strength,
        color_threshold=color_threshold,
        seam_bleed=seam_bleed,
    )

    if mod_count == 0:
        return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)

    # Re-sync CPU texture buffer from image
    flat = np.empty(w * h * 4, dtype=np.float32)
    cache.tex_image.pixels.foreach_get(flat)
    cache.tex_buf[:, :, :] = flat.reshape((h, w, 4))
    if cache.tex_buf_flat is not None:
        cache.tex_buf_flat[:] = flat

    diff = np.any(np.abs(cache.tex_buf - old_tex) > 1e-4, axis=2)
    pix = np.flatnonzero(diff).astype(np.int32)
    if pix.size == 0:
        return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)

    old_pix = old_tex.reshape((-1, 4))[pix]
    return pix, old_pix
