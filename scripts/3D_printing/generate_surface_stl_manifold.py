"""
generate_surface_stl_manifold.py

Variant of generate_surface_stl.py that uses manifold3d boolean union to
properly fuse marker cylinders into their band meshes, eliminating the
floating region warning in Bambu Slicer.

Each band is built as a triangle soup, converted to a Manifold, then each
marker cylinder for that band is unioned in. The result is a single
watertight solid per band with no internal floating geometry.

Usage:
    python generate_surface_stl_manifold.py

Requirements:
    uv add manifold3d   (or pip install manifold3d)
    pip install numpy
"""

import csv
import math
import struct
import numpy as np
from pathlib import Path

try:
    from manifold3d import Manifold, Mesh
except ImportError:
    raise ImportError(
        "manifold3d is not installed. Run: uv add manifold3d"
    )


# ── Configuration ──────────────────────────────────────────────────────────────

DATA_CSV   = Path("runs/detect/tune_sweep_results/01_broad_musgd/tune_results.csv")
OUTPUT_DIR = Path("stl_output")

GRID_N     = 60      # grid resolution
PRINT_SIZE = 100.0   # mm — square footprint
BASE_H     = 3.0     # mm — solid base below lowest fitness point
SURFACE_H  = 45.0    # mm — total height range mapped across fitness
N_BANDS    = 4       # number of color bands

DOT_EMBED  = 2.0     # mm — how far cylinders embed into terrain surface for clean boolean union
DOT_R_MIN  = 1.0     # mm — radius of lowest-fitness dot
DOT_R_MAX  = 1.4     # mm — radius of highest-fitness dot (excluding best)
PIN_R      = 2.0     # mm — radius of best-point pin (wider to stand out)
PIN_H_MIN  = 1.0     # mm — pin height for lowest fitness point
PIN_H_MAX  = 8.0     # mm — pin height for highest fitness point
PIN_SIDES  = 16      # polygon sides for dots/pins (higher = smoother)


# ── Load data ──────────────────────────────────────────────────────────────────

rows = []
with open(DATA_CSV) as f:
    for row in csv.DictReader(f):
        rows.append({
            'lr0': float(row['lr0']),
            'mom': float(row['momentum']),
            'fit': float(row['fitness']),
        })

LR0_MIN = min(r['lr0'] for r in rows); LR0_MAX = max(r['lr0'] for r in rows)
MOM_MIN = min(r['mom'] for r in rows); MOM_MAX = max(r['mom'] for r in rows)
FIT_MIN = min(r['fit'] for r in rows); FIT_MAX = max(r['fit'] for r in rows)
best    = max(rows, key=lambda r: r['fit'])

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

heights = np.zeros((GRID_N + 1, GRID_N + 1))
for j, mN in enumerate(mom_norm):
    for i, lN in enumerate(lr_norm):
        heights[j, i] = BASE_H + idw(lN, mN) * SURFACE_H

xs = lr_norm  * PRINT_SIZE
ys = mom_norm * PRINT_SIZE

band_edges = np.linspace(BASE_H, BASE_H + SURFACE_H, N_BANDS + 1)
print(f"Band height edges (mm): {np.round(band_edges, 1)}")


# ── STL writer (fallback for writing final output) ─────────────────────────────

def write_stl_from_manifold(manifold_obj, path):
    """Export a Manifold to binary STL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = manifold_obj.to_mesh()
    verts = mesh.vert_properties  # (N, 3) float32
    faces = mesh.tri_verts         # (M, 3) int32

    def tri_normal(v0, v1, v2):
        a = v1 - v0; b = v2 - v0
        n = np.cross(a, b)
        ln = np.linalg.norm(n)
        return n / ln if ln > 1e-12 else np.array([0., 0., 1.])

    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(faces)))
        for tri in faces:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            n = tri_normal(v0, v1, v2)
            f.write(struct.pack('<fff', *n))
            f.write(struct.pack('<fff', *v0))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            f.write(struct.pack('<H', 0))

    print(f"  Saved {len(faces):,} triangles -> {path.name}")


# ── Triangle soup → Manifold ───────────────────────────────────────────────────

def tris_to_manifold(tris):
    """
    Convert a list of (v0, v1, v2) triangle tuples to a Manifold.
    Attempts Merge() to fix minor non-manifold issues from the terrain mesh.
    """
    verts_list = []
    faces_list = []
    for i, (v0, v1, v2) in enumerate(tris):
        verts_list.extend([v0, v1, v2])
        faces_list.append([i*3, i*3+1, i*3+2])

    vert_arr = np.array(verts_list, dtype=np.float32)
    face_arr = np.array(faces_list, dtype=np.uint32)

    mesh = Mesh(vert_properties=vert_arr, tri_verts=face_arr)
    mesh.merge()  # modifies in-place, merges duplicate vertices
    m = Manifold(mesh)

    if m.status().value != 0:
        raise ValueError(
            f"Mesh is not manifold after Merge() (status={m.status()}). "
            "Consider increasing GRID_N or checking for degenerate triangles."
        )
    return m


# ── Band mesh generation (triangle soup) ──────────────────────────────────────

def make_band_tris(band_idx, h_lo, h_hi, heights, xs, ys):
    """Same band geometry as original script — returns triangle soup."""
    upper = np.minimum(heights, h_hi)
    lower = np.full_like(heights, h_lo) if band_idx == 0 else np.minimum(heights, h_lo)

    tris = []
    N    = len(xs) - 1

    def pt(i, j, h): return (xs[i], ys[j], float(h))

    for j in range(N):
        for i in range(N):
            p00=pt(i,j,upper[j,i]); p10=pt(i+1,j,upper[j,i+1])
            p11=pt(i+1,j+1,upper[j+1,i+1]); p01=pt(i,j+1,upper[j+1,i])
            tris.append((p00,p10,p11)); tris.append((p00,p11,p01))

    for j in range(N):
        for i in range(N):
            p00=pt(i,j,lower[j,i]); p10=pt(i+1,j,lower[j,i+1])
            p11=pt(i+1,j+1,lower[j+1,i+1]); p01=pt(i,j+1,lower[j+1,i])
            tris.append((p00,p11,p10)); tris.append((p00,p01,p11))

    for i in range(N):
        b0=pt(i,0,lower[0,i]); b1=pt(i+1,0,lower[0,i+1])
        t0=pt(i,0,upper[0,i]); t1=pt(i+1,0,upper[0,i+1])
        tris.append((b0,b1,t1)); tris.append((b0,t1,t0))

    for i in range(N):
        b0=pt(i,N,lower[N,i]); b1=pt(i+1,N,lower[N,i+1])
        t0=pt(i,N,upper[N,i]); t1=pt(i+1,N,upper[N,i+1])
        tris.append((b0,t1,b1)); tris.append((b0,t0,t1))

    for j in range(N):
        b0=pt(0,j,lower[j,0]); b1=pt(0,j+1,lower[j+1,0])
        t0=pt(0,j,upper[j,0]); t1=pt(0,j+1,upper[j+1,0])
        tris.append((b0,t0,t1)); tris.append((b0,t1,b1))

    for j in range(N):
        b0=pt(N,j,lower[j,N]); b1=pt(N,j+1,lower[j+1,N])
        t0=pt(N,j,upper[j,N]); t1=pt(N,j+1,upper[j+1,N])
        tris.append((b0,t1,t0)); tris.append((b0,b1,t1))

    return tris


# ── Cylinder as Manifold ───────────────────────────────────────────────────────

def make_cylinder_manifold(cx, cy, base_z, radius, height, sides, band_floor):
    """
    Build a cylinder using manifold3d's built-in constructor, rooted at band_floor
    and clipped to the terrain footprint so edge-adjacent markers don't overhang.
    """
    # Embed DOT_EMBED mm into the terrain surface so the boolean union has
    # real intersection volume to work with, clamped to band_floor so it
    # never pokes through the band underside.
    actual_base = max(base_z - DOT_EMBED, band_floor)
    total_height = height + (base_z - actual_base)

    cyl = Manifold.cylinder(total_height, radius, radius, sides)
    cyl = cyl.translate([cx, cy, actual_base])

    # Clip to terrain footprint — removes any overhang beyond the terrain edges
    # so edge-adjacent markers don't require supports.
    clip_box = Manifold.cube([PRINT_SIZE, PRINT_SIZE, total_height + 1.0])
    clip_box = clip_box.translate([0.0, 0.0, actual_base - 0.5])
    cyl = cyl ^ clip_box  # intersection

    return cyl


# ── Generate all band STLs with boolean union ──────────────────────────────────

BAND_LABELS = ['1_valley', '2_low', '3_high', '4_peak']
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nGenerating band STLs with boolean union...")

for b in range(N_BANDS):
    h_lo = band_edges[b]
    h_hi = band_edges[b + 1]
    is_top = b == N_BANDS - 1

    print(f"\n  Band {b+1} ({BAND_LABELS[b]})...")

    # Build band mesh and convert to Manifold
    band_tris = make_band_tris(b, h_lo, h_hi, heights, xs, ys)
    print(f"    Converting band ({len(band_tris):,} tris) to Manifold...")
    band_m = tris_to_manifold(band_tris)

    # Union each marker that belongs to this band
    marker_count = 0
    for d in rows:
        lrN  = norm_lr(d['lr0'])
        momN = norm_mom(d['mom'])
        cx   = lrN * PRINT_SIZE
        cy   = momN * PRINT_SIZE
        cz   = BASE_H + idw(lrN, momN) * SURFACE_H

        in_band = cz >= h_lo if is_top else h_lo <= cz < h_hi
        if not in_band:
            continue

        is_best = (d['lr0'] == best['lr0'] and d['mom'] == best['mom'])

        # Scale pin height proportionally to fitness across all points
        fit_norm = (d['fit'] - FIT_MIN) / (FIT_MAX - FIT_MIN)
        h = PIN_H_MIN + fit_norm * (PIN_H_MAX - PIN_H_MIN)

        # Best point gets wider radius to stand out; others scale subtly with fitness
        if is_best:
            r = PIN_R
        else:
            r = DOT_R_MIN + fit_norm * (DOT_R_MAX - DOT_R_MIN)

        cyl_m = make_cylinder_manifold(cx, cy, cz, r, h, PIN_SIDES, h_lo)
        band_m = band_m + cyl_m  # boolean union
        marker_count += 1

    print(f"    Unioned {marker_count} markers")
    write_stl_from_manifold(band_m, OUTPUT_DIR / f"band_{BAND_LABELS[b]}.stl")

# ── White base slab ───────────────────────────────────────────────────────────
# Flat slab from Z=0 to Z=BASE_H covering the full terrain footprint.
# Print in white — gives a clean foundation the border and bands sit on.
print("\n  Generating white base slab...")
base_slab = Manifold.cube([PRINT_SIZE, PRINT_SIZE, BASE_H])
write_stl_from_manifold(base_slab, OUTPUT_DIR / "base_white.stl")

print(f"\nDone! 5 STL files written to: {OUTPUT_DIR}/")
print("""
── Bambu Slicer workflow ───────────────────────────────────────────
1. File -> Import -> select all 5 STL files at once (4 bands + base_white.stl)
2. Right-click the object -> Assemble into one object
3. Assign one filament color per body:
     base_white     ->  white
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
""")
print(f"""
── Axis orientation ────────────────────────────────────────────────
  X (left to right):  lr0        {LR0_MIN:.5f} to {LR0_MAX:.5f}
  Y (front to back):  momentum   {MOM_MIN:.4f} to {MOM_MAX:.4f}
  Z (height):         fitness    {FIT_MIN:.5f} to {FIT_MAX:.5f}
  Best point pin:     lr0={best['lr0']:.5f}  mom={best['mom']:.4f}  fit={best['fit']:.5f}
───────────────────────────────────────────────────────────────────
""")