'''
hyperparameter_search.py

Runs a two-stage sweep of Ultralytics YOLO .tune() experiments using MuSGD,
    tuning only the highest-impact hyperparameters to keep total run time managable.

All experiments use yolo26.pt for speed;
    The best hyperparameters transfer to yolo28l.pt for final training.

Results from each experiment are written to a timestamped master CSV as they complete,
    and a summary is printed to the console.

Experiments:
    1. core_musgd: Fast first pass. Tunes the 5 most important MuSGD paramters:
        lr0, lrf, momentum, weight_decay, mosaic
        ~10 epochs x 20 iteractions = 200 total training epochs ≈ 2–3 hours on yolo26s.pt
    2. broad_musgd: Comprehensive second pass. Extends core_musgd with drone_aware augmentation params
        (degrees, scale, flips, hsv).
        ~10 epochs x 30 iterations = 300 total training epochs ≈ 4–6 hours on yolo26s.pt
        Before runnning, paste tightened LR ranges from --recommend output into broad_musgd space dict.

Typical Workflow:
    # Step 1 - fast first pass:
    python hyperparameter_search.py --experiments core_musgd

    # Step 2 - read results and get recommended config for tune.py / train.py:
    python hyperparameter_search.py --recommend tune_sweep_results/sweep_YYYYMMDD_HHMMSS.csv

    # Step 3 - paste tightened LR ranges into broad_musgd space dict in this file, then run comprehensive second pass:
    python hyperparameter_search.py --experiments broad_musgd

Other options: 
    # Print the experiment plan without running any training:
    python hyperparameter_search.py --dry-run

    # Run both experiments in sequence (default):
    python hyperparameter_search.py
'''

import os
import csv
import yaml
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO


# Hardware-Agnostic Device Selection
if torch.cuda.is_available():
    device = torch.device("cuda") 
    # Researve 2 corese for OS
    # cap at 12 to avoid pytorch over-allocation
    NUM_WORKERS = min(12, max(1, os.cpu_count() - 2))
    IS_CUDA = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    NUM_WORKERS = 0 # multiprocessing on MPS can hang, so we set workers to 0 for safety

    IS_CUDA = False
else:
    device = torch.device("cpu")
    NUM_WORKERS = 0
    IS_CUDA = False

if IS_CUDA:
    device_arg = 0 # first GPA index (Ultralytics expects an int for CUDA devices)
elif str(device) =="mps":
    device_arg = "mps"
else:
    device_arg = "cpu"

print(f"Device: {device} | Workers: {NUM_WORKERS} | CUDA: {IS_CUDA}")


# Paths
DATA_YAML = Path('../DS6050_ML3_Final_Proj/data/raw/data.yaml')
MODEL_WEIGHTS = "yolo26s.pt"
RESULTS_DIR = Path("tune_sweep_results")
RESULTS_CSV = RESULTS_DIR / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"



# Experiments Definitions
#
# Each experiment sets:
#   name - a unique identifier for the experiment, used in directory names and CLI filtering
#   description - a human-readable description of what the experiment is testing
#   optimizer - the optimizer to use in the tume() method; 'Adam' | 'AdamW' | 'MuSGD' | 'auto'
#   epochs - how many epochs each candidate model trains (keep this relatively low, e.g., 10-15, to save compute)
#   iterations - how many mutations/generations to run in the tuning sweep (at least 30 is recommended for a good search)
#   space - a dictionary of hyperparameter search bounds {param: (min, max)}
#               For example, 'lr': (1e-5, 1e-2) would search learning rates between 0.00001 and 0.01. 
#               Values outside the Ultralytics default space are noted inline
#
# Notes:
#   - 'batch' and 'imgsz' in 'space' require Ultralytics >= 8.1; remove them if you hit an "unexpected keyword" error on older versions.
#   - Aumentation params (flipud, fliplr, degrees, etc.) are drone-image-aware:
#        sharks can appear at any orientation in aeiral footage, so wider ranges than the Ultralytics defaults are intentional
#   - The 'broad_musgd' experiment is the most thorough - run it last or schedule it overnight, as epochs = 15 x iterations = 50 = 750 total epochs!



EXPERIMENTS = [

    # 1. Core sweep — highest-impact parameters only (faster)
    #
    # These 5 parameters drive the vast majority of MuSGD performance.
    # Run this first; use --recommend on the output CSV to get tightened
    # ranges before running broad_musgd.
    #
    #   Estimated time on yolo26s: ~10 epochs × 20 iter = 200 training epochs
    #   ≈ 2–3 hours
    {
        "name":        "core_musgd",
        "description": "Fast sweep of the 5 highest-impact MuSGD params",
        "optimizer":   "MuSGD",
        "epochs":      10,
        "iterations":  20,
        "space": {
            "lr0":          (1e-4, 1e-1),  # most critical — SGD needs higher LR than Adam
            "lrf":          (0.01, 0.3),   # shape of LR decay schedule
            "momentum":     (0.70, 0.98),  # Nesterov momentum
            "weight_decay": (0.0,  5e-4),  # regularisation
            "mosaic":       (0.5,  1.0),   # biggest single augmentation lever for YOLO
        },
    },
    
    # 2. Broad sweep — all axes, tightened ranges (overnight)
    #
    # Before running this, use:
    #   python hyperparameter_search.py --recommend <core_musgd CSV>
    # and paste the tightened lr0/lrf/momentum/weight_decay ranges it prints
    # into the space dict below.
    #
    #   Estimated time on yolo26s: ~10 epochs × 30 iter = 300 training epochs
    #   ≈ 4–6 hours

    {
        "name":        "broad_musgd",
        "description": "Comprehensive sweep — LR + augmentation + MuSGD (run overnight)",
        "optimizer":   "MuSGD",
        "epochs":      10, 
        "iterations":  30,   # 10 × 30 = 300 total training epochs
        "space": {
            # LR / optimiser — tighten these from core_musgd --recommend output
            "lr0":          (1e-4,  1e-1),
            "lrf":          (0.01,  0.3),
            "momentum":     (0.70,  0.98),
            "weight_decay": (0.0,  5e-4),

            # Augmentation — drone/aerial imagery aware
            "degrees":      (0.0,   30.0), # rotation — sharks appear at any angle
            "scale":        (0.1,    0.8),
            "flipud":       (0.0,    0.5),
            "fliplr":       (0.0,    0.5),
            "mosaic":       (0.5,    1.0),
            "hsv_s":        (0.3,    0.9), # saturation — water reflections vary
            "hsv_v":        (0.2,    0.7), # brightness — sunlight angle varies
        },
    },
]



# CSV Schema

CSV_FIELDS = [
    'experiment_name',
    'description',
    'optimizer',
    'epochs',
    'iterations',
    'total_training_epochs', # epcohs x iterations
    'space_keys',
    'started_at',
    "finished_at",
    'duration_minutes',
    'status', # "success" | "failed"
    "error", 

    # Best hyperparameters recovered from Ultralytics output
    'best_fitness',
    'best_lr0',
    'best_lrf',
    'best_momentum',
    'best_weight_decay',
    'best_batch',
    'best_imgsz',
    'best_degrees',
    'best_mosaic',
    'best_flipud',
    'best_fliplr',
    'results_dir'
]


def parse_best_hyperparameters(tune_run_dir: Path) -> dict:
    """
    Read the best_hyperparaters.yaml that Ultralytics writes after .tune().
    Returns an empty dictionary if the file is missing.
    """

    best_yaml = tune_run_dir / "best_hyperparameters.yaml"
    if not best_yaml.exists():
        print(f" [warn] best_hyperparameters.yaml not found in {tune_run_dir}")
        return {}
    with open(best_yaml) as f:
        return yaml.safe_load(f) or {}
    

def parse_best_fitness(tune_run_dir: Path) -> float | None:
    """
    Read the best fitness value from tune_results.csv
    Ultralytics names the fitness column 'fitness' (or similar),
    """
    results_csv = tune_run_dir / "tune_results.csv"
    if not results_csv.exists():
        return None
    try:
        with open(results_csv, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None
        fitness_col = next(
            (col for col in rows[0] if "fitness" in col.lower()), None
        )
        if fitness_col is None:
            return None
        values = [
            float(r[fitness_col]) for r in rows if r.get(fitness_col, "").strip()
        ]
        return max(values) if values else None
    except Exception as exc:
        print(f" [warn] Could not parse tune_results.csv: {exc}")
        return None
    
def find_actual_run_dir(results_dir: Path, run_idx: int, name: str) -> Path:
    """
    Ultralytics may append a numeric suffix to avoid overwriting existing dirs.
    Find the most recently created matching directory.
    """
    prefix = f"{run_idx:02d}_{name}"
    candidates = sorted(results_dir.glob(f"{prefix}*"), reverse=True)
    for d in candidates:
        if (d / "best_hyperparameters.yaml").exists(): # checks each dir for the yaml
            return d
    return candidates[0] if candidates else results_dir / prefix
 
 
def write_csv_row(row: dict, first_row: bool = False) -> None:
    """Write (or append) one result row to the master CSV."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if first_row else "a"
    with open(RESULTS_CSV, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if first_row:
            writer.writeheader()
        writer.writerow(row)

def recommend_config(csv_path: Path) -> None:
    """
    Read a sweep results CSV and print the recommended tune.py config for a final training run on yolo26l.pt

    The recommendation is based on the highest-fitness successful experiment.
    LR/momentum search space bounds for broad_musgd are also tighened around the best values, to focus any further tuning.

    Usage:
        python hyperparameter_search.py --recommend tune_sweep_results/sweep_YYYYMMDD_HHMMSS.csv
    """
    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}")
        return
 
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "success"]
 
    if not rows:
        print("[error] No successful experiments found in the CSV.")
        return
 
    def fitness_key(r):
        try:
            return float(r["best_fitness"])
        except (ValueError, TypeError):
            return -1.0

    # Sort all rows so we can show the full ranking
    ranked = sorted(rows, key=fitness_key, reverse=True)
    best   = ranked[0]
 
    # Helper: parse a float from the CSV, return None if missing/blank
    def f(key):
        v = best.get(key, "").strip()
        try:
            return float(v)
        except (ValueError, TypeError):
            return None
 
    best_lr0   = f("best_lr0")
    best_lrf   = f("best_lrf")
    best_mom   = f("best_momentum")
    best_wd    = f("best_weight_decay")
    best_batch = f("best_batch")
    best_imgsz = f("best_imgsz")
 
    # Tighten LR/momentum bounds to +-30% around the best value found,
    # clamped to sensible min/max limits — useful if you want to run a follow-up broad sweep with a narrower search space.
    def tighten(val, pct=0.30, lo=0.0, hi=1.0):
        if val is None:
            return None, None
        margin = val * pct
        return round(max(lo, val - margin), 8), round(min(hi, val + margin), 8)
 
    lr0_lo,  lr0_hi  = tighten(best_lr0,  pct=0.30, lo=1e-6, hi=0.1)
    lrf_lo,  lrf_hi  = tighten(best_lrf,  pct=0.30, lo=0.01, hi=1.0)
    mom_lo,  mom_hi  = tighten(best_mom,  pct=0.05, lo=0.6,  hi=0.999)
    wd_lo,   wd_hi   = tighten(best_wd,   pct=0.50, lo=0.0,  hi=0.1)
 
    W = 64
    print(f"\n{'═' * W}")
    print("  RECOMMENDED CONFIG  —  paste into tune.py / train.py")
    print(f"{'═' * W}")
    print(f"  Source CSV  : {csv_path}")
    print(f"  Best exp    : {best['experiment_name']}  (fitness {best['best_fitness']})")
    print()
    print("  ── tune.py / train.py parameters ───────────────────────")
    print(f"  model     = YOLO('yolo26l.pt')   # scale back up to large")
    print(f"  optimizer = '{best['optimizer']}'")
    if best_lr0   is not None: print(f"  lr0       = {best_lr0}")
    if best_lrf   is not None: print(f"  lrf       = {best_lrf}")
    if best_mom   is not None: print(f"  momentum  = {best_mom}")
    if best_wd    is not None: print(f"  weight_decay = {best_wd}")
    if best_batch is not None: print(f"  batch     = {int(best_batch)}")
    if best_imgsz is not None: print(f"  imgsz     = {int(best_imgsz)}")
    best_deg    = f("best_degrees")
    best_mosaic = f("best_mosaic")
    best_flipud = f("best_flipud")
    best_fliplr = f("best_fliplr")
    if best_deg    is not None: print(f"  degrees   = {best_deg}")
    if best_mosaic is not None: print(f"  mosaic    = {best_mosaic}")
    if best_flipud is not None: print(f"  flipud    = {best_flipud}")
    if best_fliplr is not None: print(f"  fliplr    = {best_fliplr}")
 
    if lr0_lo is not None:
        print()
        print("  ── Tightened space for a follow-up broad sweep ──────────")
        print("  Paste into the 'broad_musgd' (or equivalent) space dict:")
        print(f"  'lr0':          ({lr0_lo}, {lr0_hi}),")
        print(f"  'lrf':          ({lrf_lo}, {lrf_hi}),")
        print(f"  'momentum':     ({mom_lo}, {mom_hi}),")
        print(f"  'weight_decay': ({wd_lo},  {wd_hi}),")
 
    print()
    print("  ── All experiments ranked by fitness ────────────────────")
    print(f"  {'Rank':<5} {'Experiment':<25} {'Optimizer':<8} {'Fitness'}")
    for i, r in enumerate(ranked, 1):
        print(f"  {i:<5} {r['experiment_name']:<25} {r['optimizer']:<8} {r['best_fitness']}")
 
    print(f"{'═' * W}\n")



# Core experiment runner

def run_experiment(exp: dict, run_idx: int) -> dict:
    """
    Execute one tune experiment and return a populated result-row dict.
    Errors are caught and logged so the sweep continues even if one run fails
    """
    name = exp["name"]
    started_at = datetime.now()

    print(f"\n{'='*64}")
    print(f"[{started_at.strftime('%H:%M:%S')}] Experiment {run_idx}: {name}")
    print(f"Description: {exp['description']}")
    print(f"Optimizer: {exp['optimizer']}")
    print(f"Schedule: {exp['epochs']} epochs x {exp['iterations']} iterations"
          f" = {exp['epochs'] * exp['iterations']} total epochs")
    print(f" Search space: {list(exp['space'].keys())}")
    print(f"{'='*64}\n")

    # Base results row (populated on success or failure)
    row = {
        "experiment_name": name,
        "description": exp["description"],
        "optimizer": exp["optimizer"],
        "epochs": exp["epochs"],
        "iterations": exp["iterations"],
        "total_training_epochs": exp["epochs"] * exp["iterations"],
        "space_keys": "|".join(exp["space"].keys()),
        "started_at": started_at.isoformat(),
        "finished_at": "",
        "duration_minutes": "",
        "status": "failed", # default to failed, set to success at end of try block
        "error": "",
        "best_fitness": "",
        "best_lr0": "",
        "best_lrf": "",
        "best_momentum": "",
        "best_weight_decay": "",
        "best_batch": "",
        "best_imgsz": "",
        "best_degrees": "",
        "best_mosaic": "",
        "best_flipud": "",
        "best_fliplr": "",
        "results_dir": "",
    }

    try:
        model = YOLO(MODEL_WEIGHTS)

        model.tune(
            data = str(DATA_YAML),
            epochs = exp["epochs"],
            iterations = exp["iterations"],
            optimizer = exp["optimizer"],
            space = exp["space"],
            device = device_arg,
            workers = NUM_WORKERS,
            # Save each run to its own sub-directory for easy comparison
            project = str(RESULTS_DIR),
            name = f"{run_idx:02d}_{name}",
            plots = False,
            save = False,
            val = True, # val-True gives a more reliable fitness signal
        )

        finished_at = datetime.now()
        duration_min = (finished_at - started_at).total_seconds() / 60

        # Locate the actual output directory (Ultralytics may have suffixed it)
        actual_dir = find_actual_run_dir(RESULTS_DIR, run_idx, name)
        best_hp = parse_best_hyperparameters(actual_dir)
        best_fit = parse_best_fitness(actual_dir)

        row.update({
            "finished_at": finished_at.isoformat(),
            "duration_minutes": f"{duration_min:.1f}",
            "status": "success",
            "best_fitness": best_fit if best_fit is not None else "",
            "best_lr0": best_hp.get("lr0", ""),
            "best_lrf": best_hp.get("lrf", ""),
            "best_momentum": best_hp.get("momentum", ""),
            "best_weight_decay": best_hp.get("weight_decay", ""),
            "best_batch": best_hp.get("batch", ""),
            "best_imgsz": best_hp.get("imgsz", ""),
            "best_degrees": best_hp.get("degrees", ""),
            "best_mosaic": best_hp.get("mosaic", ""),
            "best_flipud": best_hp.get("flipud", ""),
            "best_fliplr": best_hp.get("fliplr", ""),
            "results_dir": str(actual_dir),
        })

        print(f"\n [success] '{name}' finished in {duration_min:.1f} minutes")
        print(f" Best fitness: {best_fit}")
        print(f" Best lr0: {best_hp.get('lr0', 'N/A')}")
        print(f" Best lrf: {best_hp.get('lrf', 'N/A')}")
        print(f" Best momentum: {best_hp.get('momentum', 'N/A')}")
        print(f" Best weight_decay: {best_hp.get('weight_decay', 'N/A')}")

    except Exception as exc:
        finished_at = datetime.now()
        duration_min = (finished_at - started_at).total_seconds() / 60
        row.update({
            "finished_at": finished_at.isoformat(),
            "duration_minutes": f"{duration_min:.1f}",
            "status": "failed",
            "error": str(exc)[:500],
        })
        print(f"\n [error] '{name}' FAILED after {duration_min:.1f} minutes:")
        print(traceback.format_exc())

    return row





# Entry point

def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO hyperparameter sweep runner for shark drone detection"
    )
    parser.add_argument(
        "--experiments", nargs="*",
        metavar="NAME",
        help=(
            "Names of experiments to run (default: all). "
            "E.g. --experiments lr_adam augmentation broad_adamw"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the experiment plan and exit without running any training.",
    )
    parser.add_argument(
        "--recommend", metavar="CSV",
        help=(
            "Path to a sweep results CSV. Reads the results, prints the best config to copy into tune.py / train.py, and exits"
            "E.g. --recommend tune_sweep_results/sweep_20240101_120000.csv"
        )
    )
    args = parser.parse_args()

    # Recommend mode - no training needed
    if args.recommend:
        recommend_config(Path(args.recommend))
        return
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found at {DATA_YAML}")


    # Filter experiments
    exps_to_run = EXPERIMENTS
    if args.experiments:
        name_set    = set(args.experiments)
        exps_to_run = [e for e in EXPERIMENTS if e["name"] in name_set]
        missing     = name_set - {e["name"] for e in exps_to_run}
        if missing:
            print(f"[warn] Unknown experiment names: {missing}")
            print(f"       Available: {[e['name'] for e in EXPERIMENTS]}")
        if not exps_to_run:
            print("No experiments to run. Exiting.")
            return
        
    # Print plan
    total_epochs = sum(e["epochs"] * e["iterations"] for e in exps_to_run)
    print(f"\n{'═' * 64}")
    print("  SHARK DRONE DETECTION — HYPERPARAMETER SWEEP")
    print(f"{'═' * 64}")
    print(f"  Model       : {MODEL_WEIGHTS}")
    print(f"  Data        : {DATA_YAML}")
    print(f"  Results CSV : {RESULTS_CSV}")
    print(f"  Experiments : {len(exps_to_run)}  ({total_epochs} total training epochs)")
    print()
    for i, exp in enumerate(exps_to_run, 1):
        print(
            f"  [{i:02d}] {exp['name']:<25s}  {exp['optimizer']:<6s}  "
            f"{exp['epochs']:>2d}ep × {exp['iterations']:>2d}iter  "
            f"= {exp['epochs'] * exp['iterations']:>4d} epochs"
        )
    print(f"{'═' * 64}")
 
    if args.dry_run:
        print("\n[dry-run] No training executed.")
        return
    
    # Run sweep
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
 
    for i, exp in enumerate(exps_to_run):
        row = run_experiment(exp, run_idx=i + 1)
        all_results.append(row)
        write_csv_row(row, first_row=(i == 0))

    successes = [r for r in all_results if r["status"] == "success"]
    failures  = [r for r in all_results if r["status"] == "failed"]
 
    print(f"\n{'═' * 64}")
    print("  SWEEP COMPLETE")
    print(f"{'═' * 64}")
    print(f"  Ran      : {len(all_results)} experiments")
    print(f"  Success  : {len(successes)}")
    print(f"  Failed   : {len(failures)}")
 
    if failures:
        print(f"\n  Failed experiments:")
        for r in failures:
            print(f"    - {r['experiment_name']} — {r['error'][:80]}")

    if successes:
        # Find the experiment with the highest best_fitness
        def fitness_key(r):
            try:
                return float(r["best_fitness"])
            except (ValueError, TypeError):
                return -1.0
            
        best = max(successes, key=fitness_key)
        print(f"\n ── Best experiment ──────────────────────────────────────")
        print(f"  Name          : {best['experiment_name']}")
        print(f"  Description   : {best['description']}")
        print(f"  Fitness       : {best['best_fitness']}")
        print(f"  Optimizer     : {best['optimizer']}")
        print(f"  lr0           : {best['best_lr0']}")
        print(f"  lrf           : {best['best_lrf']}")
        print(f"  momentum      : {best['best_momentum']}")
        print(f"  weight_decay  : {best['best_weight_decay']}")
        print(f"  mosaic        : {best['best_mosaic']}")
        print(f"  degrees       : {best['best_degrees']}")
 
    print(f"\n  Full results saved to:\n    {RESULTS_CSV}")
    print(f"{'═' * 64}\n")
 
 
if __name__ == "__main__":
    main()