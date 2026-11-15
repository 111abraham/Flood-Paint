# SPDX-License-Identifier: GPL-3.0-or-later

import bpy

from .preferences import get_active_shortcut_label
from .properties import (
    get_active_brush_color_owner,
    get_effective_target_mode,
    get_mesh_active_color_attribute,
    get_paint_settings,
    resolve_texture_paint_target,
)


class VIEW3D_PT_normal_flood_brush(bpy.types.Panel):
    """N-Panel UI for Normal-Aware Flood Brush"""

    bl_label = "Normal-Aware Flood Brush"
    bl_idname = "VIEW3D_PT_normal_flood_brush"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Flood Paint"

    def draw(self, context):
        layout = self.layout
        props = context.scene.normal_flood_brush
        obj = context.active_object

        # Launch modal operator button.
        shortcut = get_active_shortcut_label(context)
        top_col = layout.column(align=True)
        top_col.scale_y = 1.4
        top_col.operator(
            "view3d.normal_flood_brush",
            text=f"Start Flood Brush ({shortcut})",
            icon='BRUSH_DATA',
        )

        if obj is None or obj.type != 'MESH':
            layout.label(text="Select a Mesh object to paint.", icon='INFO')
        elif context.mode == 'SCULPT' and getattr(obj, "use_dynamic_topology_sculpting", False):
            layout.label(text="Disable Dyntopo to use Flood Brush.", icon='ERROR')
        elif context.mode not in {'SCULPT', 'PAINT_VERTEX', 'PAINT_TEXTURE', 'PAINT_WEIGHT', 'OBJECT'}:
            layout.label(text="Switch to Sculpt, Vertex, Texture, Weight Paint, or Object Mode.", icon='INFO')

        # Active Fill Color (or Weight Settings in Weight Paint mode).
        if context.mode == 'PAINT_WEIGHT':
            wp_box = layout.box()
            wp_box.label(text="Weight Paint Session", icon='WPAINT_HLT')
            wp_box.prop(props, "weight_auto_vertex_group", text="New Group per Session", icon='GROUP_VERTEX')
            wp_box.prop(props, "weight_value", text="Paint Weight", slider=True)
            if obj is not None and obj.type == 'MESH':
                vg = obj.vertex_groups.active
                if vg is not None:
                    wp_box.label(text=f"Active Group: {vg.name}", icon='VERTEXSEL')
                else:
                    wp_box.label(text="No active group (auto-creates on start)", icon='INFO')
            vg_box = wp_box.box()
            vg_box.label(text="Color to Vertex Group", icon='GROUP_VERTEX')
            vg_box.prop(props, "color_match_mode", text="Mode")
            if props.color_match_mode in {'HSV', 'RGB'}:
                vg_box.prop(props, "color_tolerance", text="Tolerance", slider=True)
            vg_row = vg_box.row(align=True)
            vg_row.scale_y = 1.2
            vg_row.operator(
                "view3d.flood_color_to_vertex_group",
                text="Color to Vertex Group",
                icon='GROUP_VERTEX',
            )
        else:
            color_box = layout.box()
            header_row = color_box.row(align=True)
            header_row.label(text="Active Fill Color", icon='COLOR')
            header_row.prop(props, "sync_brush_color", text="Sync Brush", toggle=True)

            paint_settings = get_paint_settings(context)
            color_owner = get_active_brush_color_owner(context) if props.sync_brush_color else None

            if color_owner is not None and hasattr(color_owner, "color"):
                color_box.template_color_picker(color_owner, "color", value_slider=True)
                row = color_box.row(align=True)
                row.prop(color_owner, "color", text="")
                if hasattr(color_owner, "secondary_color"):
                    row.prop(color_owner, "secondary_color", text="")
                row.operator("view3d.flood_flip_colors", text="", icon='ARROW_LEFTRIGHT')
                ups = getattr(paint_settings, "unified_paint_settings", None)
                if ups is not None:
                    row.prop(ups, "use_unified_color", text="", icon='BRUSHES_ALL')
                color_box.prop(props, "fill_alpha", slider=True)
                if paint_settings is not None and hasattr(paint_settings, "palette"):
                    color_box.template_ID(paint_settings, "palette", new="palette.new")
                    if paint_settings.palette:
                        color_box.template_palette(paint_settings, "palette")
            else:
                color_box.template_color_picker(props, "fill_color", value_slider=True)
                row = color_box.row(align=True)
                row.prop(props, "fill_color", text="")
                row.prop(props, "secondary_fill_color", text="")
                row.operator("view3d.flood_flip_colors", text="", icon='ARROW_LEFTRIGHT')
                if props.sync_brush_color and context.mode not in {'SCULPT', 'PAINT_VERTEX', 'PAINT_TEXTURE'}:
                    color_box.label(text="Enter a Paint or Sculpt mode to sync brush color.", icon='INFO')

            vg_box = color_box.box()
            vg_box.label(text="Color to Vertex Group", icon='GROUP_VERTEX')
            vg_box.prop(props, "color_match_mode", text="Mode")
            if props.color_match_mode in {'HSV', 'RGB'}:
                vg_box.prop(props, "color_tolerance", text="Tolerance", slider=True)
            vg_row = vg_box.row(align=True)
            vg_row.scale_y = 1.2
            vg_row.operator(
                "view3d.flood_color_to_vertex_group",
                text="Color to Vertex Group",
                icon='GROUP_VERTEX',
            )

        # Flood propagation parameters.
        flood_box = layout.box()
        flood_box.label(text="Flood Propagation", icon='MOD_SMOOTH')
        col = flood_box.column(align=True)
        col.prop(props, "fill_radius", text="Max Fill Radius")
        col.prop(props, "normal_angle_threshold", text="Normal Angle (°)", slider=True)
        col.prop(props, "crease_resistance", text="Crease Resistance", slider=True)

        row = flood_box.row(align=True)
        row.prop(
            props,
            "preserve_existing",
            text="Preserve Existing [P]",
            icon='LOCKED' if props.preserve_existing else 'UNLOCKED',
            toggle=True,
        )
        row.prop(props, "show_preview_overlay", text="Preview", icon='HIDE_OFF', toggle=True)

        gap_row = flood_box.row(align=True)
        gap_row.prop(props, "invert_flood", text="Inverse [I]", icon='ARROW_LEFTRIGHT', toggle=True)
        gap_row.prop(props, "smooth_tool", text="Smooth Tool [S]", icon='MOD_SMOOTH', toggle=True)

        fill_row = flood_box.row(align=True)
        fill_row.operator(
            "view3d.flood_fill_unpainted",
            text="Fill Unpainted [U]",
            icon='SHADING_BBOX',
        )

        offset_row = flood_box.row(align=True)
        offset_row.prop(props, "boundary_offset", text="Expand / Contract")

        # Target Output & Attribute / Texture settings.
        out_box = layout.box()
        target_mode = get_effective_target_mode(context)
        if target_mode == 'WEIGHT':
            out_box.label(text="Weight Paint Target", icon='WPAINT_HLT')
            out_box.prop(props, "weight_auto_vertex_group", text="New Group per Session")
            out_box.prop(props, "weight_value", text="Paint Weight", slider=True)
            if obj is not None and obj.type == 'MESH':
                vg = obj.vertex_groups.active
                if vg is not None:
                    out_box.label(text=f"Active Vertex Group: {vg.name}", icon='VERTEXSEL')
                else:
                    out_box.label(text="No active group (will auto-create on start)", icon='INFO')
        elif context.mode == 'PAINT_TEXTURE':
            out_box.label(text="Texture Paint Target", icon='IMAGE_DATA')
            row = out_box.row(align=True)
            row.prop(props, "texpaint_target", expand=True)
        else:
            out_box.label(text="Target Output (3D Print / Sculpt)", icon='GROUP_VCOL')

        if target_mode == 'WEIGHT':
            pass
        elif target_mode == 'TEXTURE':
            if obj is not None and obj.type == 'MESH':
                tex_img, uv_layer, _ = resolve_texture_paint_target(context, obj)
                if uv_layer is None:
                    out_box.label(text="No UV Map on mesh (unwrap first).", icon='ERROR')
                elif tex_img is None:
                    sub = out_box.column(align=True)
                    sub.label(text="No active Paint Texture found:", icon='INFO')
                    sub.prop(props, "texpaint_resolution", text="Size")
                    sub.operator("view3d.flood_create_paint_image", text="Create Paint Texture", icon='ADD')
                else:
                    out_box.label(
                        text=f"Image: {tex_img.name} ({tex_img.size[0]}x{tex_img.size[1]})",
                        icon='IMAGE',
                    )
                    out_box.label(text=f"UV Map: {uv_layer.name}", icon='UV')
            util_row = out_box.row(align=True)
            util_row.operator("view3d.flood_setup_shading", text="Show Texture", icon='SHADING_TEXTURE')
            util_row.operator("view3d.flood_reset_canvas", text="Reset Canvas", icon='FILE_REFRESH')
        else:
            row = out_box.row(align=True)
            row.prop(props, "target_output", expand=True)

            if obj is not None and obj.type == 'MESH':
                mesh = obj.data
                ca = get_mesh_active_color_attribute(mesh)
                if ca is not None:
                    info = out_box.row(align=True)
                    info.label(text=f"Active: {ca.name} ({ca.data_type}, {ca.domain})")
                else:
                    sub = out_box.column(align=True)
                    sub.label(text="No Color Attribute (auto-created on paint):", icon='INFO')
                    sub.prop(props, "default_color_domain", text="Domain")
                    sub.prop(props, "default_color_type", text="Format")

            util_row = out_box.row(align=True)
            util_row.operator("view3d.flood_setup_shading", text="Show Colors", icon='SHADING_SOLID')
            util_row.operator("view3d.flood_reset_canvas", text="Reset Canvas", icon='FILE_REFRESH')

        # Color 3D Printing Post-Processing.
        print_box = layout.box()
        print_box.label(text="3D Print Border Cleanup", icon='MOD_BEVEL')
        btn_row = print_box.row(align=True)
        btn_row.scale_y = 1.25
        btn_row.operator(
            "view3d.flood_smooth_jagged_borders",
            text="Smooth Jagged Borders",
            icon='SMOOTHCURVE',
        )
        pcol = print_box.column(align=True)
        pcol.prop(props, "smooth_iterations", text="Iterations")
        pcol.prop(props, "smooth_strength", text="Strength", slider=True)


class VIEW3D_PT_normal_flood_brush_shortcuts(bpy.types.Panel):
    """Collapsible hotkey reference for the modal Flood Brush"""

    bl_label = "Modal Controls"
    bl_idname = "VIEW3D_PT_normal_flood_brush_shortcuts"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Flood Paint"
    bl_parent_id = "VIEW3D_PT_normal_flood_brush"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        col = self.layout.column(align=True)
        col.label(text="Hover: Preview flood region", icon='MOUSE_MOVE')
        col.label(text="LMB Click / Drag: Flood paint", icon='MOUSE_LMB_DRAG')
        col.label(text="RMB / Enter / Space: Commit paint & exit", icon='CHECKMARK')
        col.label(text="X: Flip swatch color (Weight: toggle 1/0)", icon='ARROW_LEFTRIGHT')
        col.label(text="S: Toggle Smooth Tool (smooth jagged edges)", icon='MOD_SMOOTH')
        col.label(text="Shift+S: Smooth boundary mesh-wide", icon='MOD_SMOOTH')
        col.label(text="E (or Alt+S): Sample color (eyedropper)", icon='EYEDROPPER')
        col.label(text="C: Cycle palette swatches", icon='COLOR')
        col.label(text="F (or R) + Move: Resize radius", icon='ARROW_LEFTRIGHT')
        col.label(text="MMB / Shift+MMB: Orbit / Pan viewport", icon='VIEW_PAN')
        col.label(text="Scroll Wheel or [ / ]: Angle limit", icon='MOUSE_MMB_SCROLL')
        col.label(text="Ctrl+Scroll or Ctrl+[ / ]: Expand/Contract", icon='DRIVER_DISTANCE')
        col.label(text="P: Toggle Preserve Existing", icon='LOCKED')
        col.label(text="I: Toggle Inverse Flood (in radius)", icon='ARROW_LEFTRIGHT')
        col.label(text="U: Fill all remaining unpainted gaps", icon='SHADING_BBOX')
        col.label(text="Ctrl+Z / Backspace: Undo stroke", icon='LOOP_BACK')
        col.label(text="Esc: Cancel & revert session", icon='CANCEL')


classes = (
    VIEW3D_PT_normal_flood_brush,
    VIEW3D_PT_normal_flood_brush_shortcuts,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
