# Normal-Aware Flood Brush (Blender 5.2 Add-on)

An interactive, vectorized NumPy flood-fill painting tool for **Blender 5.2**, built for coloring-book style segmentation on high-density meshes (**1,000,000+ triangles**) targeting color 3D printing (`.PLY`, `.OBJ`, `.3MF`), **Texture Paint** UV images, and **Sculpt Face Sets**.

## Features

- **Zero-Lag 1M+ Face Architecture:** Pre-caches CSR face adjacency, dihedral angles, face centers/normals, and `BVHTree` in vectorized NumPy when invoked (reusing verified pure-NumPy topology/UV caches on subsequent `W` invocations for fast re-entry). Wavefront Dijkstra flood propagation runs in ~10–17 ms on 1.31M-triangle meshes, and Texture Paint strokes invalidate GPU image textures (`gl_free()`) for live 3D viewport updates while dragging.
- **Normal-Change Resistance:** Prevents colors from bleeding across clothing seams, folds, and sharp creases using a configurable `Normal Angle Threshold` (default `30°`) and `Crease Resistance` falloff slider.
- **Interactive Modal Brush (`view3d.normal_flood_brush`):**
  - Default shortcut: bare **`W`** in **Sculpt Mode**, **Vertex Paint Mode**, and **Texture Paint Mode** (also launchable from **Object Mode** via the N-Panel).
  - Live hover preview showing the exact flood region, boundary outline, and 3D surface brush ring via GPU overlays.
  - Real-time `blf` HUD readout for Angle Threshold, Fill Radius, Crease Resistance, and Preserve mode.
  - Compact delta-slice stroke undo stack (`Ctrl+Z` / `Backspace`) inside the modal session plus full Blender Undo (`Ctrl+Z`) on confirm.
- **Sculpt, Vertex Paint, Texture Paint, & Weight Paint Support:**
  - Bidirectional sync with Blender's active Sculpt / Vertex Paint / Texture Paint brush color (`UnifiedPaintSettings` & `Brush`).
  - **Weight Paint Mode (`PAINT_WEIGHT`):** Normal-aware expanding selection on mesh vertices with configurable target weight value (0.0 to 1.0). Includes a **New Group per Session** toggle switch (ON by default) which automatically allocates a clean separate vertex group for every painting session until `Esc` is pressed. If cancelled via `Esc`, the session vertex group is cleanly deleted; if confirmed via `RMB`/`Enter`/`Space`, the vertex group is saved with all painted weights.
  - **Sculpt / Vertex Paint / Object Modes:** Writes to **Color Attributes** (`FLOAT_COLOR` / `BYTE_COLOR`, `CORNER` / `POINT`), **Sculpt Face Sets** (`.sculpt_face_set`), or **Both** simultaneously.
  - **Texture Paint Mode:** Rasterizes flooded 3D triangles onto the active UV Image Texture (`UV Image Texture` mode with `seam_bleed` UV margin dilation and multi-material boundary isolation, or optional fallback to `Color Attribute`), plus a one-click **Create Paint Texture** button when starting on an untextured mesh.
  - **Smooth Tool (`S`):** Toggles boundary smoothing for the cursor preview and flood borders. Features a 100% boundary-preserving Laplacian smoother that never clips boundaries or drops segments along symmetry seams, saddle points, or character meshes, while keeping texture painting completely freeze-free and responsive.
  - One-click **Smooth Jagged Borders** operator runs a localized, vectorized boundary color-averaging pass (across mesh loops/vertices for Color Attributes or UV-masked texel borders for Texture Paint images) to remove stair-stepping without washing out flat regions or UV seams.

## Installation (Blender 5.2)

1. Open **Blender 5.2**.
2. Go to **Edit > Preferences > Add-ons**.
3. Click the **drop-down arrow (`⌄`)** in the top-right corner of the Add-ons preferences panel and select **Install from Disk...**.
4. Choose `normal_aware_flood_brush.zip` (located in this repository).
5. Enable the checkbox next to **Paint: Normal-Aware Flood Brush**.
6. *(Optional)* Expand the add-on entry in Preferences to customize the default `W` shortcut for **Sculpt Mode**, **Vertex Paint Mode**, and **Texture Paint Mode (`Image Paint`)**.

## Usage

1. Select a Mesh object and switch to **Sculpt Mode**, **Vertex Paint Mode**, **Texture Paint Mode**, or **Weight Paint Mode** (also works from **Object Mode** via the N-Panel).
   - *Texture Paint note:* Ensure your mesh has a UV Map and an active paint texture (or click **Create Paint Texture** in the **Flood Paint** N-Panel).
2. Open the 3D Viewport sidebar (`N`) and select the **Flood Paint** tab.
3. Press **`W`** in the 3D Viewport (or click **Start Flood Brush (W)**).
4. **Controls while the brush is active:**
   - **Move Mouse (Hover):** Raycasts and previews the flood fill region under the cursor (automatically disabled when cursor moves over the N-panel / UI to prevent accidental painting behind the addon).
   - **Left-Click / Drag:** Flood-fills continuously as you sweep across the surface (clicking on the N-panel / sidebar / palette passes through to UI controls safely without painting).
   - **Right-Click (RMB), `Enter`, or `Space`:** **Commit Paint & Exit Modal** — cleanly confirms all painted strokes and exits the brush.
   - **`X`:** **Flip Foreground/Background Color Swatches** — instantly swaps Blender's primary and secondary brush colors in real-time.
   - **`S`:** **Smooth Tool [S]** — toggles the Smooth Tool ON/OFF to eliminate jagged stair-stepped quad/tri edges on the cursor preview and automatically apply anti-aliased smoothing to paint stroke boundaries (press `S` to toggle; `Shift + S` runs an on-demand mesh-wide smooth pass; fully undoable via `Ctrl + Z`).
   - **`E` or `Alt + S`:** **Sample Color (Eyedropper)** — sample color from 3D surface, N-panel palette swatch, or color picker into active brush swatch.
   - **`C`:** Cycle forward through active palette swatches.
   - **`F` or `R`:** Toggle dynamic **Brush Radius** resize mode — move mouse to expand/contract radius; Left-Click, Enter, Space, `F`, or `R` locks the new radius; Right-Click or `Esc` cancels and restores initial radius (can also be adjusted anytime via the N-Panel slider).
   - **Scroll Wheel or `[` / `]`:** Adjusts `Normal Angle Threshold` (`0°`–`180°`). Hold `Alt` for `0.5°` fine angle adjustments.
   - **`P`:** Toggle **Preserve Existing** to prevent new flood strokes from overwriting previously painted colors, UV texture regions, or Face Sets (protected segments remain outlined in red with status feedback so preview never vanishes).
   - **`I`:** Toggle **Inverse Flood** (automatically includes **Gap-Fill** boundary detection): detects already-painted borders inside the brush radius and inverts that selection so only the trapped/unpainted middle faces inside the cursor are painted (great for dense corners or seams between two painted colors). When no painted borders are inside the cursor yet, it inverts the normal-angle flood across creases inside the cursor radius.
   - **`U`:** **Fill Remaining Unpainted** across the mesh in one click (also available via the **Fill Unpainted [U]** button in the **Flood Paint** N-Panel).
   - **`Ctrl + Z` or `Backspace`:** Undo the last flood stroke cleanly.
   - **`Middle Mouse Drag` (`MMB`), `Shift + MMB` (pan), `Ctrl + MMB` (zoom), `Shift/Ctrl/Alt + Wheel`:** Full native 3D Viewport navigation without leaving the tool; hover preview remains live and updates continuously during and after orbiting without disappearing.
   - **`Esc`:** Cancel and revert all strokes made during the session (or cancel radius resize if currently resizing).
5. Click **Smooth Jagged Borders** in the **Flood Paint** N-Panel before saving textures or exporting meshes (`.PLY` / `.OBJ`) for color 3D printing.

## Verification

Run the automated test suite and 1.31M-triangle benchmark with Blender 5.2 headless:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --factory-startup --background --python "tests\test_flood_brush.py"
```
