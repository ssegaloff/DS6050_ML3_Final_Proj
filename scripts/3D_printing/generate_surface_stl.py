"""
generate_surface_stl.py

Generates 4 color-band STL files from the hyperparameter sweep results
(tune_results.csv) for 3D printing on the Bambu X1C using the AMS.

The print is a terrain surface where:
    X axis = lr0 (learning rate, low to high, left to right)
    Y axis = momentum (low to high, front to back)
    Z axis = fitness (height, up)  ← Bambu Slicer Z-up convention

Fitness is interpolated across the grid using Inverse Distance Weighting
from 30 actual sweep points, which are also marked as raised dots.
The best point is marked with a taller pin.

The surface is split into 4 horizontal color bands. Each band's lower
surface follows the terrain (tapers to zero in the valley) so there are
no floating horizontal planes. Import all 4 STLs into Bambu Slicer
simultaneously, right-click → Assemble, then assign one filament per body.

Suggested AMS colors (low to high fitness):
    band_1_valley  →  dark teal / navy
    band_2_low     →  mid blue or grey
    band_3_high    →  orange
    band_4_peak    →  yellow or white

Usage:
    python generate_surface_stl.py

Requirements:
    pip install numpy
"""

import csv
import math
import struct
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_CSV   = Path("runs/detect/tune_sweep_results/01_broad_musgd/tune_results.csv")
OUTPUT_DIR = Path("stl_output")

GRID_N     = 60      # grid resolution — 60 gives smooth surface
PRINT_SIZE = 100.0   # mm — square footprint
BASE_H     = 3.0     # mm — solid base below lowest fitness point
SURFACE_H  = 45.0    # mm — total height range mapped across fitness
N_BANDS    = 4       # number of color bands

DOT_R      = 1.2     # mm — radius of sweep-point dots
DOT_H      = 1.5     # mm — height of dots above surface
PIN_R      = 1.8     # mm — radius of best-point pin
PIN_H      = 6.0     # mm — extra height of best-point pin
PIN_SIDES  = 12      # polygon sides for dots/pins


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

print(f"\nBuilding {GRID_N+1}x{GRID_N+1} height field...")

lr_norm  = np.linspace(0.0, 1.0, GRID_N + 1)
mom_norm = np.linspace(0.0, 1.0, GRID_N + 1)

# heights[j, i] in mm — j indexes momentum (Y), i indexes lr0 (X)
heights = np.zeros((GRID_N + 1, GRID_N + 1))
for j, mN in enumerate(mom_norm):
    for i, lN in enumerate(lr_norm):
        heights[j, i] = BASE_H + idw(lN, mN) * SURFACE_H

xs = lr_norm  * PRINT_SIZE   # X positions in mm
ys = mom_norm * PRINT_SIZE   # Y positions in mm

band_edges = np.linspace(BASE_H, BASE_H + SURFACE_H, N_BANDS + 1)
print(f"Band height edges (mm): {np.round(band_edges, 1)}")


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
    print(f"  Saved {len(triangles):,} triangles -> {path.name}")


# ── Band mesh generation ───────────────────────────────────────────────────────

def make_band(band_idx, h_lo, h_hi, heights, xs, ys):
    """
    Build a closed solid mesh for one color band.

    Upper surface: min(terrain, h_hi)  — terrain capped at band top
    Lower surface: min(terrain, h_lo)  — follows terrain for bands 2-4
                   flat BASE_H         — for band 0 only (the floor)

    Upper bands taper to zero thickness in the valley, eliminating
    the floating horizontal plane artifact.

    Coordinate system (Bambu Slicer Z-up):
        X = lr0 position (mm)
        Y = momentum position (mm)
        Z = fitness height (mm)
    """
    upper = np.minimum(heights, h_hi)
    lower = np.full_like(heights, h_lo) if band_idx == 0 else np.minimum(heights, h_lo)

    tris = []
    N    = len(xs) - 1

    def pt(i, j, h):
        return (xs[i], ys[j], float(h))

    # ── Top face (+Z normal) ───────────────────────────────────────────────────
    for j in range(N):
        for i in range(N):
            p00 = pt(i,   j,   upper[j,   i  ])
            p10 = pt(i+1, j,   upper[j,   i+1])
            p11 = pt(i+1, j+1, upper[j+1, i+1])
            p01 = pt(i,   j+1, upper[j+1, i  ])
            tris.append((p00, p10, p11))
            tris.append((p00, p11, p01))

    # ── Bottom face (-Z normal) ────────────────────────────────────────────────
    for j in range(N):
        for i in range(N):
            p00 = pt(i,   j,   lower[j,   i  ])
            p10 = pt(i+1, j,   lower[j,   i+1])
            p11 = pt(i+1, j+1, lower[j+1, i+1])
            p01 = pt(i,   j+1, lower[j+1, i  ])
            tris.append((p00, p11, p10))
            tris.append((p00, p01, p11))

    # ── South wall (j=0, -Y normal) ───────────────────────────────────────────
    for i in range(N):
        b0 = pt(i,   0, lower[0, i  ])
        b1 = pt(i+1, 0, lower[0, i+1])
        t0 = pt(i,   0, upper[0, i  ])
        t1 = pt(i+1, 0, upper[0, i+1])
        tris.append((b0, b1, t1))
        tris.append((b0, t1, t0))

    # ── North wall (j=N, +Y normal) ───────────────────────────────────────────
    for i in range(N):
        b0 = pt(i,   N, lower[N, i  ])
        b1 = pt(i+1, N, lower[N, i+1])
        t0 = pt(i,   N, upper[N, i  ])
        t1 = pt(i+1, N, upper[N, i+1])
        tris.append((b0, t1, b1))
        tris.append((b0, t0, t1))

    # ── West wall (i=0, -X normal) ────────────────────────────────────────────
    for j in range(N):
        b0 = pt(0, j,   lower[j,   0])
        b1 = pt(0, j+1, lower[j+1, 0])
        t0 = pt(0, j,   upper[j,   0])
        t1 = pt(0, j+1, upper[j+1, 0])
        tris.append((b0, t0, t1))
        tris.append((b0, t1, b1))

    # ── East wall (i=N, +X normal) ────────────────────────────────────────────
    for j in range(N):
        b0 = pt(N, j,   lower[j,   N])
        b1 = pt(N, j+1, lower[j+1, N])
        t0 = pt(N, j,   upper[j,   N])
        t1 = pt(N, j+1, upper[j+1, N])
        tris.append((b0, t1, t0))
        tris.append((b0, b1, t1))

    return tris


# ── Marker geometry ────────────────────────────────────────────────────────────

def make_cylinder(cx, cy, base_z, radius, height, sides):
    """
    Upright cylinder at (cx, cy) in XY, base at base_z, extending up by height.
    Z is up (Bambu Slicer convention).
    """
    angles = [2 * math.pi * k / sides for k in range(sides)]
    top_z  = base_z + height
    tris   = []

    for k in range(sides):
        a0, a1 = angles[k], angles[(k + 1) % sides]
        b0 = (cx + radius*math.cos(a0), cy + radius*math.sin(a0), base_z)
        b1 = (cx + radius*math.cos(a1), cy + radius*math.sin(a1), base_z)
        t0 = (cx + radius*math.cos(a0), cy + radius*math.sin(a0), top_z)
        t1 = (cx + radius*math.cos(a1), cy + radius*math.sin(a1), top_z)
        # Side
        tris.append((b0, t1, b1))
        tris.append((b0, t0, t1))
        # Top cap
        tris.append(((cx, cy, top_z),  t0, t1))
        # Bottom cap
        tris.append(((cx, cy, base_z), b1, b0))

    return tris


def make_markers_for_band(rows, xs_scale, ys_scale, h_lo, h_hi, is_top_band):
    """Return marker triangles whose base height falls within this band."""
    tris = []
    for d in rows:
        lrN  = norm_lr(d['lr0'])
        momN = norm_mom(d['mom'])
        cx   = lrN * xs_scale
        cy   = momN * ys_scale
        cz   = BASE_H + idw(lrN, momN) * SURFACE_H

        # Assign to the band containing cz; top band uses >= to catch the peak
        in_band = (h_lo <= cz < h_hi) or (is_top_band and cz >= h_lo)
        if not in_band:
            continue

        is_best = (d['lr0'] == best['lr0'] and d['mom'] == best['mom'])
        r = PIN_R if is_best else DOT_R
        h = (DOT_H + PIN_H) if is_best else DOT_H
        tris.extend(make_cylinder(cx, cy, cz, r, h, PIN_SIDES))

    return tris


# ── Generate all band STLs ─────────────────────────────────────────────────────

BAND_LABELS = ['1_valley', '2_low', '3_high', '4_peak']

print(f"\nGenerating band STLs...")

for b in range(N_BANDS):
    h_lo = band_edges[b]
    h_hi = band_edges[b + 1]

    tris = make_band(b, h_lo, h_hi, heights, xs, ys)
    tris.extend(make_markers_for_band(
        rows, PRINT_SIZE, PRINT_SIZE, h_lo, h_hi, is_top_band=(b == N_BANDS - 1)
    ))

    write_stl(tris, OUTPUT_DIR / f"band_{BAND_LABELS[b]}.stl")

print(f"\nDone! 4 STL files written to: {OUTPUT_DIR}/")
print("""
── Bambu Slicer workflow ───────────────────────────────────────────
1. File -> Import -> select all 4 band_*.stl files at once
2. Right-click the object -> Assemble into one object
3. Assign one filament color per body:
     band_1_valley  ->  dark teal / navy
     band_2_low     ->  mid blue or grey
     band_3_high    ->  orange
     band_4_peak    ->  yellow or white
4. Print settings:
     Layer height:  0.15 mm
     Infill:        15%
     Supports:      none
     Brim:          3 mm (recommended for adhesion)
───────────────────────────────────────────────────────────────────

── Axis orientation ────────────────────────────────────────────────
  X (left to right):  lr0        {:.5f} to {:.5f}
  Y (front to back):  momentum   {:.4f} to {:.4f}
  Z (height):         fitness    {:.5f} to {:.5f}
  Best point pin:     lr0={:.5f}  mom={:.4f}  fit={:.5f}
───────────────────────────────────────────────────────────────────
""".format(LR0_MIN, LR0_MAX, MOM_MIN, MOM_MAX, FIT_MIN, FIT_MAX,
           best['lr0'], best['mom'], best['fit']))