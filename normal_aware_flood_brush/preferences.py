# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
import rna_keymap_ui


KEYMAP_SPECS = (
    ("Sculpt", 'EMPTY', 'WINDOW'),
    ("Vertex Paint", 'EMPTY', 'WINDOW'),
    ("Image Paint", 'EMPTY', 'WINDOW'),
    ("Weight Paint", 'EMPTY', 'WINDOW'),
)

_addon_keymaps: list[tuple[bpy.types.KeyMap, bpy.types.KeyMapItem]] = []


def register_keymaps():
    _addon_keymaps.clear()
    wm = bpy.context.window_manager
    if wm is None:
        return
    kc = wm.keyconfigs.addon
    if kc is None:
        return

    for km_name, space_type, region_type in KEYMAP_SPECS:
        km = kc.keymaps.new(name=km_name, space_type=space_type, region_type=region_type)
        kmi = km.keymap_items.new(
            "view3d.normal_flood_brush",
            type='W',
            value='PRESS',
        )
        _addon_keymaps.append((km, kmi))

    # Weight Paint mode: X key toggles weight between 1.0 and 0.0 in vanilla Blender
    km_wp = kc.keymaps.new(name="Weight Paint", space_type='EMPTY', region_type='WINDOW')
    kmi_x = km_wp.keymap_items.new(
        "view3d.flood_weight_toggle",
        type='X',
        value='PRESS',
    )
    _addon_keymaps.append((km_wp, kmi_x))


def unregister_keymaps():
    for km, kmi in _addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    _addon_keymaps.clear()


def get_active_shortcut_label(context) -> str:
    """Return a readable string for the current user-configured shortcut in the active mode."""
    wm = context.window_manager
    if wm is None or wm.keyconfigs.user is None:
        return "W"
    mode_km_map = {
        'PAINT_VERTEX': "Vertex Paint",
        'PAINT_TEXTURE': "Image Paint",
        'PAINT_WEIGHT': "Weight Paint",
    }
    target_km = mode_km_map.get(context.mode, "Sculpt")
    km = wm.keyconfigs.user.keymaps.get(target_km)
    if km is None:
        return "W"
    for kmi in km.keymap_items:
        if kmi.idname == "view3d.normal_flood_brush" and kmi.active:
            return kmi.to_string()
    return "W"


class NormalFloodBrushPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    def draw(self, context):
        layout = self.layout
        layout.label(text="Shortcuts (Sculpt, Vertex Paint & Texture Paint Modes):", icon='KEYINGSET')

        wm = context.window_manager
        kc = wm.keyconfigs.user if wm is not None else None
        if kc is None:
            layout.label(text="Keyconfig unavailable in current context.")
            return

        col = layout.column(align=True)
        for km_name, _space, _reg in KEYMAP_SPECS:
            km = kc.keymaps.get(km_name)
            if km is None:
                continue
            found = False
            for kmi in km.keymap_items:
                if kmi.idname in {"view3d.normal_flood_brush", "view3d.flood_weight_toggle"}:
                    box = col.box()
                    action_name = "Start Flood Brush" if kmi.idname == "view3d.normal_flood_brush" else "Toggle Weight (1/0)"
                    box.label(text=f"{km_name} — {action_name}:")
                    rna_keymap_ui.draw_kmi([], kc, km, kmi, box, 0)
                    found = True
            if not found:
                col.label(text=f"{km_name}: (Not bound — restore in Keymap Preferences)")


classes = (
    NormalFloodBrushPreferences,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    register_keymaps()


def unregister():
    unregister_keymaps()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
