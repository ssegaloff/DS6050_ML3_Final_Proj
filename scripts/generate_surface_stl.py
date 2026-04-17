"""
generate_surface_stl.py

Generates 4 color-band STL files from the hyperparameter sweep results
(tune_results.csv) for 3D printing on the Bambu X1C using the AMS.

The print is a terrain surface where:
    X axis = lr0 (learning rate, low → high, left → right)
    Z axis = momentum (low → high, front → back)
    Y axis = fitness (height)

Fitness is interpolated across the grid using Inverse Distance Weighting
from your 30 actual sweep points, which are also marked as raised dots.
The best point is marked with a taller pin.

The surface is split into 4 horizontal color bands. Import all 4 STLs into
Bambu Slicer simultaneously, right-click → Assemble, then assign one filament
color per body.

Suggested AMS colors (low → high fitness):
    band_1_valley  →  dark teal / navy
    band_2_low     →  mid blue or grey
    band_3_high    →  orange
    band_4_peak    →  yellow or white

Usage:
    python generate_surface_stl.py

    The script expects tune_results.csv at the path set in DATA_CSV below.
    Outputs are written to stl_output/.

Requirements:
    pip install numpy
    (no other dependencies — STL is written as raw binary)
"""

import csv
import math
import struct
import numpy as np
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────────

DATA_CSV   = Path("runs/detect/tune_sweep_results/01_broad_musgd/tune_results.csv")   # adjust if needed
OUTPUT_DIR = Path("stl_output")

GRID_N     = 60      # grid resolution — 60 gives a smooth surface, raise to 80 for final print
PRINT_SIZE = 150.0   # mm — footprint (square), fits X1C build plate comfortably
BASE_H     = 3.0     # mm — solid base below the lowest fitness point
SURFACE_H  = 45.0    # mm — total height range mapped across fitness (low → high)
N_BANDS    = 4       # number of color bands (one STL per band)

# Marker geometry
DOT_R      = 1.2     # mm — radius of sweep-point dots on surface
DOT_H      = 1.5     # mm — height of dots above surface
PIN_R      = 1.8     # mm — radius of best-point pin
PIN_H      = 6.0     # mm — extra height of best-point pin above surface
PIN_SIDES  = 12      # polygon sides for dots/pins (12 = smooth enough for PLA at 0.15mm)


# ── Load data ──────────────────────────────────────────────────────────────────

rows = []
with open(DATA_CSV) as f:
    for row in csv.DictReader(f):
        rows.append({
            'lr0': float(row['lr0']),
            'mom': float(row['momentum']),
            'fit': float(row['fitness']),
        })

LR0_MIN = min(r['lr0'] for r in rows)
LR0_MAX = max(r['lr0'] for r in rows)
MOM_MIN = min(r['mom'] for r in rows)
MOM_MAX = max(r['mom'] for r in rows)
FIT_MIN = min(r['fit'] for r in rows)
FIT_MAX = max(r['fit'] for r in rows)

best = max(rows, key=lambda r: r['fit'])

print(f"Loaded {len(rows)} sweep points")
print(f"  lr0:      {LR0_MIN:.5f} – {LR0_MAX:.5f}")
print(f"  momentum: {MOM_MIN:.4f} – {MOM_MAX:.4f}")
print(f"  fitness:  {FIT_MIN:.5f} – {FIT_MAX:.5f}")
print(f"  best:     lr0={best['lr0']:.5f}  mom={best['mom']:.4f}  fit={best['fit']:.5f}")


# ── IDW interpolation ──────────────────────────────────────────────────────────

def norm_lr(v):  return (v - LR0_MIN) / (LR0_MAX - LR0_MIN)
def norm_mom(v): return (v - MOM_MIN) / (MOM_MAX - MOM_MIN)
def norm_fit(v): return (v - FIT_MIN) / (FIT_MAX - FIT_MIN)

def idw(lr_n, mom_n, power=3):
    """Inverse Distance Weighting interpolation in normalised [0,1]² space."""
    wsum = fsum = 0.0
    for d in rows:
        dx = norm_lr(d['lr0'])  - lr_n
        dy = norm_mom(d['mom']) - mom_n
        dist2 = dx*dx + dy*dy
        if dist2 < 1e-10:
            return norm_fit(d['fit'])
        w = 1.0 / (dist2 ** power)
        wsum += w
        fsum += w * norm_fit(d['fit'])
    return fsum / wsum


# ── Build height field ─────────────────────────────────────────────────────────

print(f"\nBuilding {GRID_N+1}×{GRID_N+1} height field...")

lr_norm  = np.linspace(0.0, 1.0, GRID_N + 1)
mom_norm = np.linspace(0.0, 1.0, GRID_N + 1)

# heights[j, i] in mm — j indexes momentum (Z), i indexes lr0 (X)
heights = np.zeros((GRID_N + 1, GRID_N + 1))
for j, mN in enumerate(mom_norm):
    for i, lN in enumerate(lr_norm):
        heights[j, i] = BASE_H + idw(lN, mN) * SURFACE_H

xs = lr_norm  * PRINT_SIZE   # X positions in mm (lr0 axis)
zs = mom_norm * PRINT_SIZE   # Z positions in mm (momentum axis)

# Band boundary heights in mm
band_edges = np.linspace(BASE_H, BASE_H + SURFACE_H, N_BANDS + 1)
print(f"Band height edges (mm): {np.round(band_edges, 1)}")


# ── STL primitives ─────────────────────────────────────────────────────────────

def tri_normal(v0, v1, v2):
    a = np.array(v1, dtype=float) - v0
    b = np.array(v2, dtype=float) - v0
    n = np.cross(a, b)
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else np.array([0., 1., 0.])

def write_stl(triangles, path):
    """Write a list of (v0, v1, v2) tuples as a binary STL file."""
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
    print(f"  Saved {len(triangles):,} triangles → {path.name}")


# ── Band mesh generation ───────────────────────────────────────────────────────

def make_band(h_lo, h_hi, heights, xs, zs):
    """
    Build a closed solid mesh for one color band between h_lo and h_hi mm.

    Geometry:
      Top face    — terrain surface clamped to [h_lo, h_hi]
      Bottom face — flat plane at h_lo
      4 perimeter walls joining top edges to bottom edges

    Winding convention: right-hand rule, normals pointing outward.
    """
    surf = np.clip(heights, h_lo, h_hi)
    tris = []
    N    = len(xs) - 1

    # Coordinate convention for Bambu Slicer (Z is up):
    #   X = lr0 position (mm)
    #   Y = momentum position (mm)
    #   Z = fitness height (mm)  ← slicer up axis
    def pt(i, j, h):
        return (xs[i], zs[j], h)

    # ── Top face (outward normal = +Z) ────────────────────────────────────────
    for j in range(N):
        for i in range(N):
            p00 = pt(i,   j,   surf[j,   i  ])
            p10 = pt(i+1, j,   surf[j,   i+1])
            p11 = pt(i+1, j+1, surf[j+1, i+1])
            p01 = pt(i,   j+1, surf[j+1, i  ])
            tris.append((p00, p10, p11))
            tris.append((p00, p11, p01))

    # ── Bottom face (outward normal = −Z) ─────────────────────────────────────
    for j in range(N):
        for i in range(N):
            p00 = pt(i,   j,   h_lo)
            p10 = pt(i+1, j,   h_lo)
            p11 = pt(i+1, j+1, h_lo)
            p01 = pt(i,   j+1, h_lo)
            tris.append((p00, p11, p10))
            tris.append((p00, p01, p11))

    # ── South wall (j=0, outward normal = −Y) ─────────────────────────────────
    for i in range(N):
        b0 = pt(i,   0, h_lo)
        b1 = pt(i+1, 0, h_lo)
        t0 = pt(i,   0, surf[0, i  ])
        t1 = pt(i+1, 0, surf[0, i+1])
        tris.append((b0, b1, t1))
        tris.append((b0, t1, t0))

    # ── North wall (j=N, outward normal = +Y) ─────────────────────────────────
    for i in range(N):
        b0 = pt(i,   N, h_lo)
        b1 = pt(i+1, N, h_lo)
        t0 = pt(i,   N, surf[N, i  ])
        t1 = pt(i+1, N, surf[N, i+1])
        tris.append((b0, t1, b1))
        tris.append((b0, t0, t1))

    # ── West wall (i=0, outward normal = −X) ──────────────────────────────────
    for j in range(N):
        b0 = pt(0, j,   h_lo)
        b1 = pt(0, j+1, h_lo)
        t0 = pt(0, j,   surf[j,   0])
        t1 = pt(0, j+1, surf[j+1, 0])
        tris.append((b0, t0, t1))
        tris.append((b0, t1, b1))

    # ── East wall (i=N, outward normal = +X) ──────────────────────────────────
    for j in range(N):
        b0 = pt(N, j,   h_lo)
        b1 = pt(N, j+1, h_lo)
        t0 = pt(N, j,   surf[j,   N])
        t1 = pt(N, j+1, surf[j+1, N])
        tris.append((b0, t1, t0))
        tris.append((b0, b1, t1))

    return tris


# ── Marker geometry (dots + best-point pin) ────────────────────────────────────

def make_cylinder(cx, cy_base, cz, radius, height, sides):
    """
    Upright closed cylinder centred at (cx, cz) with base at cy_base.
    Used for sweep-point dots and the best-point pin.
    """
    angles = [2 * math.pi * k / sides for k in range(sides)]
    top_y  = cy_base + height
    tris   = []

    # Cylinder axis is Z (up in slicer); radius expands in XY plane
    for k in range(sides):
        a0, a1 = angles[k], angles[(k + 1) % sides]
        # Side quad — (X, Y, Z) = (cx + r*cos, cz + r*sin, height)
        b0 = (cx + radius*math.cos(a0), cz + radius*math.sin(a0), cy_base)
        b1 = (cx + radius*math.cos(a1), cz + radius*math.sin(a1), cy_base)
        t0 = (cx + radius*math.cos(a0), cz + radius*math.sin(a0), top_y)
        t1 = (cx + radius*math.cos(a1), cz + radius*math.sin(a1), top_y)
        tris.append((b0, t1, b1))
        tris.append((b0, t0, t1))
        # Top cap
        tris.append(((cx, cz, top_y),   t0, t1))
        # Bottom cap
        tris.append(((cx, cz, cy_base), b1, b0))

    return tris


def make_markers(rows, heights_fn, xs_mm, zs_mm, best_lr0, best_mom):
    """
    Build marker geometry for all 30 sweep points plus the best-point pin.
    heights_fn(lr_n, mom_n) returns the interpolated surface height in mm.
    Each marker sits on top of the surface at its (lr0, momentum) location.
    """
    tris = []
    for d in rows:
        lrN  = norm_lr(d['lr0'])
        momN = norm_mom(d['mom'])
        cx   = lrN  * PRINT_SIZE
        cz   = momN * PRINT_SIZE
        cy   = BASE_H + idw(lrN, momN) * SURFACE_H   # surface height (Z in slicer)

        is_best = (d['lr0'] == best_lr0 and d['mom'] == best_mom)
        r = PIN_R if is_best else DOT_R
        h = (DOT_H + PIN_H) if is_best else DOT_H
        tris.extend(make_cylinder(cx, cy, cz, r, h, PIN_SIDES))

    return tris


# ── Generate all band STLs ─────────────────────────────────────────────────────

BAND_LABELS = ['1_valley', '2_low', '3_high', '4_peak']

print(f"\nGenerating band STLs...")
all_marker_tris = make_markers(rows, idw, xs, zs, best['lr0'], best['mom'])

for b in range(N_BANDS):
    h_lo = band_edges[b]
    h_hi = band_edges[b + 1]

    tris = make_band(h_lo, h_hi, heights, xs, zs)

    # Add markers that fall within this band's height range
    # (clip each marker's contribution to the band, or just add all to top band)
    # Simplest and cleanest: markers go into whichever band contains the surface at that point
    if b == N_BANDS - 1:
        # top band gets all markers so they print in the peak color
        tris.extend(all_marker_tris)

    write_stl(tris, OUTPUT_DIR / f"band_{BAND_LABELS[b]}.stl")

print(f"\nDone! 4 STL files written to: {OUTPUT_DIR}/")
print("""
── Bambu Slicer workflow ───────────────────────────────────────────
1. File → Import → select all 4 band_*.stl files at once
2. In the object list, right-click → Assemble into one object
3. In the Filament panel, assign one color per body:
     band_1_valley  →  dark teal / navy
     band_2_low     →  mid blue or grey
     band_3_high    →  orange
     band_4_peak    →  yellow or white
4. Print settings:
     Layer height:  0.15 mm  (crisper surface detail)
     Infill:        15%      (terrain walls are thin, don't need more)
     Supports:      none     (geometry is self-supporting)
     Brim:          3mm      (optional but helps adhesion on large flat base)
───────────────────────────────────────────────────────────────────

── Axis orientation ────────────────────────────────────────────────
  X axis (left → right):   lr0   {:.5f} → {:.5f}
  Z axis (front → back):   momentum  {:.4f} → {:.4f}
  Y axis (height):         fitness   {:.5f} → {:.5f}
  Best point:              lr0={:.5f}  mom={:.4f}  fit={:.5f}
     (marked with taller pin on surface)
───────────────────────────────────────────────────────────────────
""".format(LR0_MIN, LR0_MAX, MOM_MIN, MOM_MAX, FIT_MIN, FIT_MAX,
           best['lr0'], best['mom'], best['fit']))