# SPDX-License-Identifier: GPL-3.0-or-later

bl_info = {
    "name": "Normal-Aware Flood Brush",
    "author": "Antigravity",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > Flood Paint (W in Sculpt / Vertex / Texture Paint)",
    "description": (
        "Interactive normal-aware flood fill brush for coloring-book style painting "
        "on 1M+ triangle meshes for color 3D printing"
    ),
    "warning": "",
    "category": "Paint",
}

if "bpy" in locals():
    import importlib
    from . import mesh_cache, flood_engine, overlay, properties, preferences, operators, ui
    importlib.reload(mesh_cache)
    importlib.reload(flood_engine)
    importlib.reload(overlay)
    importlib.reload(properties)
    importlib.reload(preferences)
    importlib.reload(operators)
    importlib.reload(ui)
else:
    import bpy
    from . import mesh_cache, flood_engine, overlay, properties, preferences, operators, ui

_SUBMODULES = (
    mesh_cache,
    flood_engine,
    overlay,
    properties,
    preferences,
    operators,
    ui,
)


from bpy.app.handlers import persistent


@persistent
def _on_load_pre(_dummy):
    mesh_cache.clear_geometry_cache()


def register():
    for mod in _SUBMODULES:
        if hasattr(mod, "register"):
            mod.register()
    if _on_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_on_load_pre)


def unregister():
    mesh_cache.clear_geometry_cache()
    if _on_load_pre in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_on_load_pre)
    for mod in reversed(_SUBMODULES):
        if hasattr(mod, "unregister"):
            mod.unregister()


if __name__ == "__main__":
    register()
