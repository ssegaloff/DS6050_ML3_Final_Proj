"""
generate_border.py

Generates a flat rectangular border frame with tick marks to accompany
the hyperparameter fitness surface print (generate_surface_stl.py).

The frame sits at the base of the terrain and extends outward by BORDER_W mm
on each side. Tick marks are raised on the top face of the frame at key
lr0 and momentum values.

After loading in Bambu Slicer, use the built-in Text tool to add labels:
    Right-click flat surface -> Add Text
    Recommended labels:
        South face outer wall  ->  "lr0"   (learning rate axis)
        West face outer wall   ->  "momentum"
        North or East face     ->  "fitness (height)"
        Corner / top           ->  "Hyperparameter Sweep — Team Toothless"

Output:
    stl_output/border.stl

Usage:
    python generate_border.py

Requirements:
    pip install numpy
"""

import struct
import numpy as np
from pathlib import Path


# ── Configuration — must match generate_surface_stl.py ───────────────────────

PRINT_SIZE = 100.0   # mm — terrain footprint (square)
BASE_H     = 3.0     # mm — terrain base height (border sits here)

BORDER_W   = 10.0    # mm — width of border apron on each side
BORDER_H   = 3.0     # mm — thickness of border slab

# Tick mark geometry
TICK_W     = 1.0              # mm — tick width
TICK_L     = BORDER_W * 0.28  # mm — tick length (~28% of border width; scales with BORDER_W)
TICK_H     = 1.5              # mm — tick height above border top face

# Axis ranges — must match generate_surface_stl.py data
LR0_MIN, LR0_MAX = 0.010, 0.0356
MOM_MIN, MOM_MAX = 0.700, 0.9370

# Tick positions
LR0_TICKS = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035]
MOM_TICKS  = [0.70,  0.75,  0.80,  0.85,  0.90]

OUTPUT_DIR = Path("stl_output")


# ── STL writer ─────────────────────────────────────────────────────────────────

def tri_normal(v0, v1, v2):
    a = np.array(v1, dtype=float) - v0
    b = np.array(v2, dtype=float) - v0
    n = np.cross(a, b)
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else np.array([0., 0., 1.])

def write_stl(triangles, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(triangles)))
        for v0, v1, v2 in triangles:
            n = tri_normal(v0, v1, v2)
            f.write(struct.pack('<fff', *n))
            f.write(struct.pack('<fff', *np.array(v0, dtype=float)))
            f.write(struct.pack('<fff', *np.array(v1, dtype=float)))
            f.write(struct.pack('<fff', *np.array(v2, dtype=float)))
            f.write(struct.pack('<H', 0))
    print(f"Saved {len(triangles):,} triangles -> {path.name}")


# ── Box primitive ──────────────────────────────────────────────────────────────

def make_box(x0, y0, z0, x1, y1, z1):
    """Solid axis-aligned box as 12 triangles with outward normals."""
    tris = []
    # Bottom (-Z)
    tris.append(((x0,y0,z0), (x1,y1,z0), (x1,y0,z0)))
    tris.append(((x0,y0,z0), (x0,y1,z0), (x1,y1,z0)))
    # Top (+Z)
    tris.append(((x0,y0,z1), (x1,y0,z1), (x1,y1,z1)))
    tris.append(((x0,y0,z1), (x1,y1,z1), (x0,y1,z1)))
    # South (-Y)
    tris.append(((x0,y0,z0), (x1,y0,z0), (x1,y0,z1)))
    tris.append(((x0,y0,z0), (x1,y0,z1), (x0,y0,z1)))
    # North (+Y)
    tris.append(((x0,y1,z0), (x1,y1,z1), (x1,y1,z0)))
    tris.append(((x0,y1,z0), (x0,y1,z1), (x1,y1,z1)))
    # West (-X)
    tris.append(((x0,y0,z0), (x0,y0,z1), (x0,y1,z1)))
    tris.append(((x0,y0,z0), (x0,y1,z1), (x0,y1,z0)))
    # East (+X)
    tris.append(((x1,y0,z0), (x1,y1,z0), (x1,y1,z1)))
    tris.append(((x1,y0,z0), (x1,y1,z1), (x1,y0,z1)))
    return tris


# ── Border geometry ────────────────────────────────────────────────────────────

def make_border():
    tris = []

    BW  = BORDER_W
    PS  = PRINT_SIZE
    Z0  = BASE_H              # bottom of border (aligns with terrain base)
    Z1  = BASE_H + BORDER_H   # top of border slab
    ZT  = Z1 + TICK_H         # top of tick marks

    # ── Frame slabs (4 pieces, corners included in south and north) ───────────

    # South slab (includes SW and SE corners)
    tris.extend(make_box(-BW, -BW, Z0,  PS+BW, 0,    Z1))
    # North slab (includes NW and NE corners)
    tris.extend(make_box(-BW, PS,  Z0,  PS+BW, PS+BW, Z1))
    # West slab
    tris.extend(make_box(-BW, 0,   Z0,  0,     PS,    Z1))
    # East slab
    tris.extend(make_box(PS,  0,   Z0,  PS+BW, PS,    Z1))

    # ── lr0 tick marks on south border top face ───────────────────────────────
    # Ticks run parallel to Y (extending from y=0 inward toward y=-TICK_L)
    for lr0_val in LR0_TICKS:
        x = ((lr0_val - LR0_MIN) / (LR0_MAX - LR0_MIN)) * PS
        tris.extend(make_box(
            x - TICK_W/2, -TICK_L, Z1,
            x + TICK_W/2,  0,      ZT
        ))

    # ── momentum tick marks on west border top face ───────────────────────────
    # Ticks run parallel to X (extending from x=0 inward toward x=-TICK_L)
    for mom_val in MOM_TICKS:
        y = ((mom_val - MOM_MIN) / (MOM_MAX - MOM_MIN)) * PS
        tris.extend(make_box(
            -TICK_L, y - TICK_W/2, Z1,
            0,       y + TICK_W/2, ZT
        ))

    # ── Corner notch markers (small raised squares at axis origins) ───────────
    # SW corner — origin of both axes
    tris.extend(make_box(-4, -4, Z1, -1, -1, ZT + 0.5))

    return tris


# ── Generate and save ──────────────────────────────────────────────────────────

tris = make_border()
write_stl(tris, OUTPUT_DIR / "border.stl")

print(f"""
── Bambu Slicer text label guide ───────────────────────────────────
After importing border.stl alongside your 4 band STLs:

1. Right-click the south outer face (front wall) -> Add Text
       Text:  "lr0"
       Size:  ~6 mm,  Emboss depth: 0.6 mm

2. Right-click the west outer face (left wall) -> Add Text
       Text:  "momentum"
       Size:  ~6 mm,  Emboss depth: 0.6 mm

3. Right-click the east outer face (right wall) -> Add Text
       Text:  "fitness (height)"
       Size:  ~5 mm,  Emboss depth: 0.6 mm

4. Optionally on north outer face -> Add Text
       Text:  "Hyperparameter Sweep — Team Toothless"
       Size:  ~5 mm,  Emboss depth: 0.6 mm

── Tick mark positions (for Bambu Slicer text placement) ────────────
South face — lr0 axis (X position from left edge of terrain):
{"".join(f"  {v:.3f}  →  {((v - LR0_MIN) / (LR0_MAX - LR0_MIN)) * PRINT_SIZE:.1f} mm from left{chr(10)}" for v in LR0_TICKS)}
West face — momentum axis (Y position from front edge of terrain):
{"".join(f"  {v:.2f}  →  {((v - MOM_MIN) / (MOM_MAX - MOM_MIN)) * PRINT_SIZE:.1f} mm from front{chr(10)}" for v in MOM_TICKS)}

── Print tip ────────────────────────────────────────────────────────
Print the border in a neutral color (grey or white) so the 4 terrain
color bands read clearly against it.
────────────────────────────────────────────────────────────────────
""")