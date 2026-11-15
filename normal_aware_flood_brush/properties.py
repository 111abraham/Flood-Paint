# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
)


def get_paint_settings(context):
    """Return the active mode's paint settings (Sculpt, Vertex Paint, Texture Paint, or Weight Paint), or None."""
    ts = getattr(context, "tool_settings", None)
    if ts is None:
        return None
    mode = context.mode
    if mode == 'SCULPT':
        return getattr(ts, "sculpt", None)
    if mode == 'PAINT_VERTEX':
        return getattr(ts, "vertex_paint", None)
    if mode == 'PAINT_TEXTURE':
        return getattr(ts, "image_paint", None)
    if mode == 'PAINT_WEIGHT':
        return getattr(ts, "weight_paint", None)
    return None


def get_mesh_active_color_attribute(mesh):
    """Safely return the active ColorAttribute on a mesh, or None.

    Never access mesh.color_attributes.active directly, because in Blender
    RNA that returns mesh.attributes.active (e.g. 'position' FloatVectorAttribute),
    which is not a color attribute and has no .color property.
    """
    if mesh is None or not hasattr(mesh, "color_attributes") or len(mesh.color_attributes) == 0:
        return None
    ca = mesh.color_attributes.active_color
    if ca is not None and ca.name in mesh.color_attributes:
        return ca
    return mesh.color_attributes[0]


def get_active_brush_color_owner(context):
    """Return the RNA struct holding the active brush color (UnifiedPaintSettings or Brush)."""
    paint_settings = get_paint_settings(context)
    if paint_settings is None:
        return None
    ups = getattr(paint_settings, "unified_paint_settings", None)
    if ups is not None and getattr(ups, "use_unified_color", False):
        return ups
    return getattr(paint_settings, "brush", None)


def get_effective_fill_rgba(context):
    """Return active fill color as a 4-tuple of linear RGBA floats in [0.0, 1.0]."""
    props = context.scene.normal_flood_brush
    if props.sync_brush_color:
        owner = get_active_brush_color_owner(context)
        if owner is not None and hasattr(owner, "color"):
            c = owner.color
            return (float(c[0]), float(c[1]), float(c[2]), float(props.fill_alpha))
    fc = props.fill_color
    return (float(fc[0]), float(fc[1]), float(fc[2]), float(props.fill_alpha))


def get_effective_secondary_rgba(context):
    """Return active background/secondary color as a 4-tuple of linear RGBA floats in [0.0, 1.0]."""
    props = context.scene.normal_flood_brush
    if props.sync_brush_color:
        owner = get_active_brush_color_owner(context)
        if owner is not None and hasattr(owner, "secondary_color"):
            c = owner.secondary_color
            return (float(c[0]), float(c[1]), float(c[2]), float(props.fill_alpha))
    if hasattr(props, "secondary_fill_color"):
        sc = props.secondary_fill_color
        return (float(sc[0]), float(sc[1]), float(sc[2]), float(props.fill_alpha))
    return (1.0, 1.0, 1.0, float(props.fill_alpha))


def set_effective_fill_rgb(context, rgb):
    """Set the active fill RGB color on both addon properties and the active Blender brush."""
    props = context.scene.normal_flood_brush
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    props.fill_color = (r, g, b, props.fill_alpha)
    if props.sync_brush_color:
        owner = get_active_brush_color_owner(context)
        if owner is not None and hasattr(owner, "color"):
            owner.color = (r, g, b)


def flip_brush_colors(context):
    """Switch/flip the foreground (primary) and background (secondary) brush colors in Blender."""
    props = context.scene.normal_flood_brush
    paint_settings = get_paint_settings(context)
    owner = get_active_brush_color_owner(context)

    flipped_native = False
    try:
        if hasattr(bpy.ops.paint, "brush_colors_flip"):
            res = bpy.ops.paint.brush_colors_flip()
            if 'FINISHED' in res:
                flipped_native = True
    except Exception:
        pass

    if not flipped_native and owner is not None:
        if hasattr(owner, "color") and hasattr(owner, "secondary_color"):
            c1 = (float(owner.color[0]), float(owner.color[1]), float(owner.color[2]))
            c2 = (float(owner.secondary_color[0]), float(owner.secondary_color[1]), float(owner.secondary_color[2]))
            owner.color = c2
            owner.secondary_color = c1

    if paint_settings is not None:
        brush = getattr(paint_settings, "brush", None)
        ups = getattr(paint_settings, "unified_paint_settings", None)
        if ups is not None and getattr(ups, "use_unified_color", False) and brush is not None:
            if hasattr(brush, "color") and hasattr(brush, "secondary_color") and not flipped_native:
                c1 = (float(brush.color[0]), float(brush.color[1]), float(brush.color[2]))
                c2 = (float(brush.secondary_color[0]), float(brush.secondary_color[1]), float(brush.secondary_color[2]))
                brush.color = c2
                brush.secondary_color = c1

    if hasattr(props, "secondary_fill_color"):
        c_pri = tuple(props.fill_color)
        c_sec = tuple(props.secondary_fill_color)
        props.fill_color = c_sec
        props.secondary_fill_color = c_pri

    owner = get_active_brush_color_owner(context)
    if owner is not None and hasattr(owner, "color"):
        props.fill_color = (float(owner.color[0]), float(owner.color[1]), float(owner.color[2]), props.fill_alpha)

    return get_effective_fill_rgba(context)


def resolve_texture_paint_target(context, obj):
    """Resolve (image, uv_layer, restrict_material_index) for Texture Paint mode."""
    if obj is None or obj.type != 'MESH':
        return None, None, None
    mesh = obj.data
    ip = getattr(context.tool_settings, "image_paint", None)
    if ip is None:
        return None, mesh.uv_layers.active, None

    if ip.mode == 'IMAGE':
        return ip.canvas, mesh.uv_layers.active, None

    # MATERIAL mode: resolve active material's paint slot image & UV layer.
    mat = obj.active_material
    mat_idx = obj.active_material_index if len(obj.material_slots) > 1 else None
    if mat is not None and len(mat.texture_paint_images) > 0:
        slot_idx = max(0, min(mat.paint_active_slot, len(mat.texture_paint_images) - 1))
        img = mat.texture_paint_images[slot_idx]
        slot = mat.texture_paint_slots[slot_idx]
        uv_name = getattr(slot, "uv_layer", "")
        uv_layer = mesh.uv_layers.get(uv_name) if uv_name else mesh.uv_layers.active
        return img, uv_layer, mat_idx

    # Fallback if material has an active ShaderNodeTexImage before slot cache refresh.
    nt = getattr(mat, "node_tree", None) if mat is not None else None
    if nt is not None:
        active_node = nt.nodes.active
        if active_node is not None and active_node.type == 'TEX_IMAGE' and active_node.image is not None:
            return active_node.image, mesh.uv_layers.active, mat_idx
        for node in nt.nodes:
            if node.type == 'TEX_IMAGE' and node.image is not None:
                return node.image, mesh.uv_layers.active, mat_idx

    return ip.canvas, mesh.uv_layers.active, mat_idx


def get_effective_target_mode(context) -> str:
    """Return the active target output ('WEIGHT', 'TEXTURE', 'COLOR', 'FACE_SET', or 'BOTH')."""
    if context.mode == 'PAINT_WEIGHT':
        return 'WEIGHT'
    props = context.scene.normal_flood_brush
    if context.mode == 'PAINT_TEXTURE' and props.texpaint_target == 'TEXTURE':
        return 'TEXTURE'
    return props.target_output


def get_effective_weight_value(context) -> float:
    """Return the weight value to paint in Weight Paint mode."""
    props = context.scene.normal_flood_brush
    ts = getattr(context, "tool_settings", None)
    if ts is not None:
        ups = getattr(ts, "unified_paint_settings", None)
        if ups is not None and getattr(ups, "use_unified_weight", False):
            return float(getattr(ups, "weight", props.weight_value))
        wp = getattr(ts, "weight_paint", None)
        if wp is not None and getattr(wp, "brush", None) is not None:
            return float(getattr(wp.brush, "weight", props.weight_value))
    return float(props.weight_value)


class NormalFloodBrushProperties(bpy.types.PropertyGroup):
    fill_radius: FloatProperty(
        name="Max Fill Radius",
        description="Maximum geodesic/spatial radius (in scene units) for the flood propagation",
        default=0.25,
        min=0.0005,
        max=50.0,
        soft_min=0.005,
        soft_max=5.0,
        subtype='DISTANCE',
        unit='LENGTH',
    )
    normal_angle_threshold: FloatProperty(
        name="Normal Angle Threshold",
        description="Dihedral normal angle limit in degrees before propagation slows down or stops at creases",
        default=30.0,
        min=0.0,
        max=180.0,
        soft_min=1.0,
        soft_max=120.0,
        precision=1,
    )
    crease_resistance: FloatProperty(
        name="Crease Resistance",
        description=(
            "Controls how strongly normal changes resist flood propagation "
            "(1.0 = hard stop at threshold; lower values allow gradual slowdown across creases)"
        ),
        default=0.95,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )
    target_output: EnumProperty(
        name="Target Output",
        description="Where flood brush strokes write segment data in Sculpt / Vertex Paint / Object modes",
        items=[
            ('COLOR', "Color Attribute", "Paint directly to the active Mesh Color Attribute (for .PLY / .OBJ / .3MF export)"),
            ('FACE_SET', "Face Sets", "Generate unique Sculpt Face Sets per flood stroke"),
            ('BOTH', "Color & Face Sets", "Paint to the active Color Attribute and generate Sculpt Face Sets simultaneously"),
        ],
        default='COLOR',
    )
    texpaint_target: EnumProperty(
        name="Texture Paint Target",
        description="Where flood brush strokes write when invoked inside Texture Paint Mode",
        items=[
            ('TEXTURE', "UV Image Texture", "Rasterize flooded 3D triangles onto the active Texture Paint image"),
            ('COLOR', "Color Attribute", "Paint onto the mesh Color Attribute instead of a UV image"),
        ],
        default='TEXTURE',
    )
    texpaint_resolution: EnumProperty(
        name="New Texture Resolution",
        description="Resolution used when creating a new Paint Texture from the Flood Paint panel",
        items=[
            ('1024', "1024 x 1024", "1K texture"),
            ('2048', "2048 x 2048", "2K texture"),
            ('4096', "4096 x 4096", "4K texture"),
        ],
        default='2048',
    )
    preserve_existing: BoolProperty(
        name="Preserve Existing",
        description="Prevent new flood fills from overwriting previously painted segments or distinct Face Sets (Hotkey: P)",
        default=False,
    )
    invert_flood: BoolProperty(
        name="Inverse Flood",
        description=(
            "Enables Gap-Fill boundary detection while inverting the selection inside the brush radius — "
            "detects painted borders and selects only the unpainted/trapped middle faces inside the cursor (Hotkey: I)"
        ),
        default=False,
    )
    smooth_tool: BoolProperty(
        name="Smooth Tool",
        description="Smooth the jagged stair-stepped boundary edges of the flood cursor preview and paint strokes (Hotkey: S)",
        default=False,
    )
    boundary_offset: IntProperty(
        name="Expand / Contract",
        description="Expand (positive) or contract (negative) the flood boundary by polygon rings (default: 0; Hotkeys: [ / ])",
        default=0,
        min=-50,
        max=50,
    )
    weight_auto_vertex_group: BoolProperty(
        name="New Group per Session",
        description="Automatically create a separate vertex group for every weight painting session until Esc is pressed",
        default=True,
    )
    weight_value: FloatProperty(
        name="Weight",
        description="Weight value applied to vertices in Weight Paint mode",
        default=1.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )
    sync_brush_color: BoolProperty(
        name="Sync Active Brush Color",
        description="Keep the Fill Color synchronized with Blender's active Sculpt/Vertex/Texture Paint brush color",
        default=True,
    )
    fill_color: FloatVectorProperty(
        name="Fill Color",
        description="Fallback linear RGBA fill color (used when no active paint brush is available)",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.9, 0.25, 0.2, 1.0),
    )
    secondary_fill_color: FloatVectorProperty(
        name="Secondary Color",
        description="Secondary/background linear RGBA color (swapped with Fill Color via X)",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
    )
    fill_alpha: FloatProperty(
        name="Fill Alpha",
        description="Alpha channel value written to Color Attributes or Image Textures",
        min=0.0,
        max=1.0,
        default=1.0,
        subtype='FACTOR',
    )
    default_color_domain: EnumProperty(
        name="New Attribute Domain",
        description="Domain used when automatically creating a new Color Attribute",
        items=[
            ('CORNER', "Face Corner", "Per-face-corner colors for razor-sharp seam edges (recommended for coloring-book fills)"),
            ('POINT', "Vertex", "Per-vertex colors for lightweight storage and smooth transitions"),
        ],
        default='CORNER',
    )
    default_color_type: EnumProperty(
        name="New Attribute Format",
        description="Data type used when automatically creating a new Color Attribute",
        items=[
            ('FLOAT_COLOR', "Float Color (Fastest 1M+)", "32-bit float RGBA — fastest real-time updates on 1M+ triangle meshes"),
            ('BYTE_COLOR', "Byte Color (8-bit)", "8-bit sRGB RGBA — compact memory footprint"),
        ],
        default='FLOAT_COLOR',
    )
    smooth_iterations: IntProperty(
        name="Smooth Iterations",
        description="Number of neighborhood averaging passes along color boundaries",
        default=2,
        min=1,
        max=20,
    )
    smooth_strength: FloatProperty(
        name="Smooth Strength",
        description="Blend factor per smoothing pass along detected color boundaries",
        default=0.75,
        min=0.05,
        max=1.0,
        subtype='FACTOR',
    )
    show_preview_overlay: BoolProperty(
        name="Live Hover Preview",
        description="Render a real-time 3D preview overlay of the flooded region and boundary under the cursor",
        default=True,
    )


classes = (
    NormalFloodBrushProperties,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.normal_flood_brush = bpy.props.PointerProperty(type=NormalFloodBrushProperties)


def unregister():
    if hasattr(bpy.types.Scene, "normal_flood_brush"):
        del bpy.types.Scene.normal_flood_brush
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
