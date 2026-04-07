'''
hyperparameter_search.py

Runs a sweep of Ultralytics YOLO .tune() experiments with different configurations.

Each experiment varies the tune() meta-parameters (optimizer, epochs, iterations) 
    and the hyperparameters (learning rate, augmentation, batch, etc.).

Results from al experiments are written to a timestamped master CSV for comparison,
and printed to the console as each run completes.

Usage:
    # Run all experiments:
    python hyperparameter_search.py

    #Run a specific experiment by name:
    python hyperparameter_search.py --experiments lr_adam augmentation broad_adamw

    # Print experiement paln wihtout training:
    python hyperparameter_search.py -- dry-run
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
MODEL_WEIGHTS = "yolo26l.pt" # Using the same model for tuning to ensure hyperparameters are optimized for the final model
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
#   - The 'auto' optimizer lets Ultralytics choose the best optimizer for each mutant model, which can lead to better results but less control.
#   - 'batch' and 'imgsz' in 'space' require Ultralytics >= 8.1; remove them if you hit an "unexpected keyword" error on older versions.
#   - Aumentation params (flipud, fliplr, degrees, etc.) are drone-image-aware:
#        sharks can appear at any orientation in aeiral footage, so wider ranges than the Ultralytics defaults are intentional
#   - The 'broad_adamw' experiment is the most thorough - run it last or schedule it overnight, as epochs = 15 x iterations = 50 = 750 total epochs!



EXPERIMENTS = [
    # 1. Learning rate sweep, Adam

    {
        "name": "lr_adam",
        "description": "LR scehdule sweep with Adam -- good Adam baseline",
        "optimizer": "Adam",
        "epochs": 10,
        "iterations": 30,
        "space": {
            'lr0': (1e-5, 1e-2), # initial LR
            'lrf': (0.01, 0.5),   # final LR as a fraction of initial (Ultralytics default is 0.01, so this sweeps from 1% to 50% of the initial LR
            'momentum': (0.90, 0.999), # beta1 for Adam
            'weight_decay': (0.0, 1e-3) 
        }
    },

    # 2. Learning Rate sweep MuSGD
    {
        "name": "lr_musgd",
        "description": "LR schedule sweep with MuSGD",
        "optimizer": "MuSGD",
        "epochs": 10,
        "iterations": 30,
        "space": {
            'lr0': (1e-4, 1e-1), # # SGD typically needs a higher initial LR
            'lrf': (0.01, 0.3),
            'momentum': (0.70, 0.98), # Nesterov momentum
            'weight_decay': (0.0, 5e-4) 
        }
    },

    # 3. Learning rate sweep , AdamW
    {
        "name": "lr_adamw",
        "description": "LR schedule sweep with AdamW - decoupled weight decay",
        "optimizer": "AdamW",
        "epochs": 10,
        "iterations": 30,
        "space": {
            'lr0': (1e-5, 5e-3), 
            'lrf': (0.01, 0.5), 
            'momentum': (0.90, 0.999), # beta1
            'weight_decay': (1e-5, 1e-2) # AdamW benefits from more weight decay
        }
    },

    # 4. Augementation parameters
    {
        "name": "augmentation",
        "description": "Aerial-imagery augmentation sweep (drone-angle aware)",
        "optimizer": "MuSGD",
        "epochs": 10,
        "iterations": 30,
        "space": {
            # Geometric - sharks appear from directly above; rotation matters
            'degrees': (0.0, 45.0), # rotation range in degrees - wider than default 0
            'scale': (0.1, 0.9), # scale jitter
            'translate': (0.0, 0.3), # translation fraction
            'flipud': (0.0, 0.5), # vertical flip - meaningful for drone views
            'fliplr': (0.0, 0.5), # horizontal flip
            'shear': (0.0, 10.0), # shear angle in degrees
            'mosaic': (0.5, 1.0), # mosaic helps with dense/sparse scenes across 7k+ images
            'hsv_h': (0.0, 0.05), # Color, compensate for water reflection, sunlight angle
            'hsv_s': (0.3, 0.9), # hue jitter (narrow - underwater color stable)
            'hsv_v': (0.2, 0.7), # brightness jitter 
        }
    },

    # 5. Batch size and image resolution
    {
        "name": "batch_imgsize",
        "description": "Bath and image size sweep - hardware throughput vs. detail",
        "optimizer": "MuSGD",
        "epochs": 10,
        "iterations": 20, # fewer iterationc: larger imgsz = longer per trial
        "space": {
            # Note: requires Ultralytics >= 8.1; remove if you get an error
            'batch': (8, 32), # lower = more gradient noise; higher = faster
            'imgsz': (416, 1024),  # higher res helps for small/distance sharks
        }
    },
    
    # 6. Broad multi-axis sweep (recommended for final run)

    {
        "name":        "broad_adamw",
        "description": "Comprehensive sweep — LR + augmentation + AdamW (run overnight)",
        "optimizer":   "AdamW",
        "epochs":      15,   # slightly longer per trial for a more reliable signal
        "iterations":  50,   # 15 × 50 = 750 total training epochs
        "space": {
            "lr0":          (1e-5,  1e-2),
            "lrf":          (0.01,  0.3),
            "momentum":     (0.85,  0.999),
            "weight_decay": (1e-5,  1e-2),
            "degrees":      (0.0,   30.0),
            "scale":        (0.1,    0.8),
            "flipud":       (0.0,    0.5),
            "fliplr":       (0.0,    0.5),
            "mosaic":       (0.5,    1.0),
            "hsv_s":        (0.3,    0.9),
            "hsv_v":        (0.2,    0.7),
        },
    },
]



# CSV Schema

CSV_FIELDS = [
    'experiment_name',
    'desciption',
    'optimizer',
    'epochs',
    'iterations',
    'total_training_epochs', # epcohs x iterations
    'space_keys',
    'started_at',
    "finished_at",
    'duration_minutes',
    'status', # "success" | "failed"
    "error", # Best hyperparameters recovered from Ultralytics output
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
        return yaml.afe_load(f) or {}
    

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



# Core experiment runner

def run_experiment(exp: dict, run_idx: int) -> dict:
    """
    Execute one tune experiment and return a populated result-row dict.
    Errors are caught and logged so the sweep continues even if one run fails
    """
    name = exp["name"]
    started_at = datetime.now()

    print(f"\n{'=' +64}")
    print(f"[{started_at.strftime('%H:%M:%:S')}] Experiment {run_idx}: {name}")
    print(f'Description: {exp['description']}')
    print(f'Optimizer: {exp["optimizer"]}')
    print(f'Schedule: {exp['epochs']} epochs x {exp['iterations']} iterations'
          f' = {exp['epochs'] * exp['iterations']} total epochs')
    print(f' Search space: {list(exp['space'].keys())}')
    print(f"{'='*64}\n")

    # Base results row (populated on success or failure)
    row = {
        "experiment_name": name,
        "desciption": exp["description"],
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
    args = parser.parse_args()
 
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