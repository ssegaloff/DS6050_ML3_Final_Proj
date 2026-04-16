'''
dashboard.py

Generates a self-contained HTML results dashboard for the Team Toothless
shark detection project.

Attribution: Please note that the Claude LLM was utilized to help draft the 
HTML/CSS/JS structure and styling of this dashboard.

Sections:
  - Summary cards (mAP50, Recall, Precision, F1 per run)
  - Overall metrics | Per-class AP50 | Per-class Recall bar charts
  - Training curves: mAP50 (+ best-epoch dot) | box loss | cls loss
    with early-stopping vertical marker on all three
  - Confusion matrices (click to enlarge lightbox)
  - PR curve + F1 curve plots (click to enlarge, same lightbox)
  - Dataset annotation distribution
  - Run hyperparameter table

Usage:
    python dashboard.py

Per-class recall requires validate.py and baseline_validate.py to save
{class}_Recall and {class}_Precision rows — see validate_updated.py.
All sections gracefully hide if their data files are not found.
'''

import csv
import json
import base64
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Config — edit these to match your runs
# ---------------------------------------------------------------------------

BASELINE_CSV = Path("runs/detect/baseline_yolo26l/validation/test_metrics.csv")

RUN_CSVS = [
    Path("runs/detect/sharks_v5_freeze23/validation/test_metrics.csv"),
    Path("runs/detect/sharks_v5_freeze11/validation/test_metrics.csv"),
    Path("runs/detect/sharks_v5_freeze10/validation/test_metrics.csv")
    
]

OUTPUT = Path("dashboard.html")

OVERALL_KEYS   = ["mAP50", "mAP50-95", "Precision", "Recall", "F1"]
CLASS_SUFFIXES = ["boat", "human", "other", "shark"]
METADATA_KEYS  = [
    "optimizer", "freeze", "epochs", "patience", "seed",
    "lr0", "lrf", "momentum", "weight_decay",
    "batch", "imgsz",
    "mosaic", "degrees", "scale", "flipud", "fliplr", "hsv_s", "hsv_v",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metrics(csv_path: Path) -> dict:
    with open(csv_path, newline="") as f:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(f)}


def infer_run_dir(csv_path: Path) -> Path:
    """runs/detect/<n>/validation/test_metrics.csv  ->  runs/detect/<n>"""
    return csv_path.parent.parent


def load_training_curves(run_dir: Path) -> dict | None:
    """
    Read Ultralytics results.csv; return epoch-aligned arrays plus
    early-stopping and best-epoch metadata.
    Column names have leading whitespace in Ultralytics output — strip them.
    """
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return None

    epochs, map50 = [], []
    train_box, val_box = [], []
    train_cls, val_cls = [], []

    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                epochs.append(int(float(row["epoch"])))
                map50.append(float(row.get("metrics/mAP50(B)", 0)))
                train_box.append(float(row.get("train/box_loss", 0)))
                val_box.append(float(row.get("val/box_loss", 0)))
                train_cls.append(float(row.get("train/cls_loss", 0)))
                val_cls.append(float(row.get("val/cls_loss", 0)))
            except (KeyError, ValueError):
                continue

    if not epochs:
        return None

    # Early stopping: compare last trained epoch to max configured epochs
    max_epochs = None
    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        with open(args_yaml) as f:
            args = yaml.safe_load(f) or {}
        max_epochs = args.get("epochs")

    stop_epoch   = epochs[-1]
    early_stopped = max_epochs is not None and stop_epoch < max_epochs - 1

    # Best mAP50 epoch
    best_j    = max(range(len(map50)), key=lambda j: map50[j])
    best_epoch = epochs[best_j]
    best_map50 = map50[best_j]

    return {
        "epochs": epochs,
        "map50": map50,
        "train_box": train_box, "val_box": val_box,
        "train_cls": train_cls, "val_cls": val_cls,
        "stop_epoch":    stop_epoch,
        "max_epochs":    max_epochs,
        "early_stopped": early_stopped,
        "best_epoch":    best_epoch,
        "best_map50":    best_map50,
    }


def _find_file(run_dir: Path, filename: str) -> str | None:
    """
    Return base64-encoded PNG from the first of several candidate paths,
    handling the Ultralytics doubled-path quirk.
    """
    candidates = [
        run_dir / "validation" / filename,
        run_dir / filename,
        Path("runs") / "detect" / run_dir / "validation" / filename,
    ]
    for p in candidates:
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode()
    return None


def load_plots(run_dir: Path) -> dict:
    """Load all embeddable PNG plots from the validation directory."""
    return {
        "confusion_matrix_b64": _find_file(run_dir, "confusion_matrix_normalized.png"),
        "pr_curve_b64":         _find_file(run_dir, "PR_curve.png"),
        "f1_curve_b64":         _find_file(run_dir, "F1_curve.png"),
    }


def load_metadata(run_dir: Path) -> dict | None:
    """Read args.yaml written by Ultralytics at the start of training."""
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.exists():
        return None
    with open(args_yaml) as f:
        args = yaml.safe_load(f) or {}
    return {k: args.get(k) for k in METADATA_KEYS}


# ---------------------------------------------------------------------------
# HTML template  (uses __DATA__ / __RUN_COUNT__ for injection so the JS
# can be written as plain, unescaped code with no double-brace escaping)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Team Toothless - Shark Detection Results</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #ffffff;
    --bg2:     #f5f4f0;
    --text:    #1a1a18;
    --text2:   #5f5e5a;
    --text3:   #888780;
    --border:  rgba(0,0,0,0.10);
    --success: #0f6e56;
    --stop:    rgba(210,120,30,0.65);
    --radius:  8px;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg:      #1a1a18;
      --bg2:     #242422;
      --text:    #f0ede6;
      --text2:   #a8a69e;
      --text3:   #6b6a64;
      --border:  rgba(255,255,255,0.10);
      --success: #5dcaa5;
      --stop:    rgba(230,150,60,0.7);
    }
  }

  body {
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    padding: 2rem 1.5rem 4rem; max-width: 1000px; margin: 0 auto;
  }

  .eyebrow { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--text3); margin-bottom: 4px; }
  h1 { font-size: 24px; font-weight: 500; margin-bottom: 4px; }
  .subtitle { font-size: 13px; color: var(--text2); margin-bottom: 2rem; }

  .section { margin-bottom: 2.5rem; }
  .section-label    { font-size: 13px; font-weight: 500; color: var(--text2); margin: 0 0 12px; }
  .section-sublabel { font-size: 11px; color: var(--text3); margin: 0 0 8px; }

  /* summary cards */
  .run-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin-bottom: 2.5rem; }
  .run-card { background: var(--bg2); border-radius: var(--radius); padding: 14px 12px; border: 1px solid transparent; }
  .run-card.best { border-color: var(--success); }
  .run-card-title { font-size: 12px; font-weight: 500; color: var(--text2); margin-bottom: 10px; }
  .run-card-title span { font-weight: 400; color: var(--text3); }
  .metric-mini-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .metric-mini-label { font-size: 11px; color: var(--text3); margin-bottom: 2px; }
  .metric-mini-val   { font-size: 16px; font-weight: 500; color: var(--text); }
  .metric-mini-val.dim   { color: var(--text2); }
  .metric-mini-val.green { color: var(--success); }

  /* chart grids */
  .chart-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  .chart-grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.25rem; }
  @media (max-width: 640px) {
    .chart-grid-2, .chart-grid-3 { grid-template-columns: 1fr; }
  }
  .chart-wrap { position: relative; width: 100%; }

  /* metrics 3-col (auto-collapses if a column is hidden) */
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2.5rem;
  }

  /* image grids (confusion matrices, PR/F1 curves) */
  .img-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.5rem; }
  .img-label { font-size: 12px; font-weight: 500; color: var(--text2); margin-bottom: 6px; }
  .img-thumb { width: 100%; border-radius: 4px; display: block; cursor: zoom-in; transition: opacity .15s; }
  .img-thumb:hover { opacity: .85; }
  .img-download { display: block; font-size: 11px; color: var(--text3); margin-top: 5px; text-align: right; }
  .img-download:hover { color: var(--text2); }
  .chart-dl { display: block; font-size: 11px; color: var(--text3); margin-top: 5px; text-align: right; cursor: pointer; text-decoration: none; }
  .chart-dl:hover { color: var(--text2); }

  /* lightbox */
  #lightbox {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,.82); z-index: 1000;
    align-items: center; justify-content: center; cursor: zoom-out;
  }
  #lightbox.open { display: flex; }
  #lightbox img { max-width: min(90vw,900px); max-height: 90vh; border-radius: 6px; box-shadow: 0 8px 40px rgba(0,0,0,.5); }
  #lb-close { position: absolute; top: 1rem; right: 1.25rem; font-size: 28px; color: rgba(255,255,255,.7); cursor: pointer; line-height: 1; user-select: none; }
  #lb-close:hover { color: #fff; }

  /* hyperparameter table */
  .table-wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { padding: 7px 14px; text-align: left; border-bottom: 1px solid var(--border); }
  th { font-weight: 500; color: var(--text2); font-size: 12px; }
  .meta-key { font-family: monospace; font-size: 12px; color: var(--text3); }
  tr:last-child td { border-bottom: none; }
  tbody tr:hover { background: var(--bg2); }

  /* legend */
  .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; font-size: 12px; color: var(--text2); }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .legend-swatch { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
  .legend-line-solid { display: inline-block; width: 18px; height: 0; border-top: 2px solid var(--text3); vertical-align: middle; }
  .legend-line-dash  { display: inline-block; width: 18px; height: 0; border-top: 2px dashed var(--text3); vertical-align: middle; }
  .legend-line-stop  { display: inline-block; width: 18px; height: 0; border-top: 2px dashed var(--stop); vertical-align: middle; }
</style>
</head>
<body>

<p class="eyebrow">DS 6050 - Team Toothless</p>
<h1>Shark detection results</h1>
<p class="subtitle">Test split &middot; conf 0.15 &middot; __RUN_COUNT__ runs compared</p>

<!-- Summary cards -->
<div class="run-grid" id="runCards"></div>

<!-- Overall | Per-class AP50 | Per-class Recall -->
<div class="metrics-grid">
  <div>
    <p class="section-label">Overall metrics</p>
    <div class="chart-wrap" style="height:220px;"><canvas id="overallChart"></canvas></div>
    <a class="chart-dl" onclick="downloadCanvas('overallChart','overall_metrics.png')">Download</a>
    <div class="legend" id="overallLegend"></div>
  </div>
  <div>
    <p class="section-label">Per-class AP50 &mdash; fine-tuned</p>
    <div class="chart-wrap" style="height:220px;"><canvas id="perClassChart"></canvas></div>
    <a class="chart-dl" onclick="downloadCanvas('perClassChart','per_class_ap50.png')">Download</a>
    <div class="legend" id="perClassLegend"></div>
  </div>
  <div id="pcRecallCol">
    <p class="section-label">Per-class recall &mdash; fine-tuned</p>
    <div class="chart-wrap" style="height:220px;"><canvas id="pcRecallChart"></canvas></div>
    <a class="chart-dl" onclick="downloadCanvas('pcRecallChart','per_class_recall.png')">Download</a>
    <div class="legend" id="pcRecallLegend"></div>
  </div>
</div>

<!-- Training curves -->
<div class="section" id="curvesSection">
  <p class="section-label">Training curves &mdash; fine-tuned runs</p>
  <div class="chart-grid-3" style="margin-bottom:10px;">
    <div>
      <p class="section-sublabel">Validation mAP50 &mdash; &bull; best epoch</p>
      <div class="chart-wrap" style="height:190px;"><canvas id="map50Chart"></canvas></div>
      <a class="chart-dl" onclick="downloadCanvas('map50Chart','training_map50.png')">Download</a>
    </div>
    <div>
      <p class="section-sublabel">Box loss &mdash; val (solid) &middot; train (dashed)</p>
      <div class="chart-wrap" style="height:190px;"><canvas id="boxLossChart"></canvas></div>
      <a class="chart-dl" onclick="downloadCanvas('boxLossChart','training_box_loss.png')">Download</a>
    </div>
    <div>
      <p class="section-sublabel">Cls loss &mdash; val (solid) &middot; train (dashed)</p>
      <div class="chart-wrap" style="height:190px;"><canvas id="clsLossChart"></canvas></div>
      <a class="chart-dl" onclick="downloadCanvas('clsLossChart','training_cls_loss.png')">Download</a>
    </div>
  </div>
  <div class="legend" id="curveLegend"></div>
</div>

<!-- Confusion matrices -->
<div class="section" id="cmSection">
  <p class="section-label">Confusion matrices (normalized) &mdash; click to enlarge</p>
  <div class="img-grid" id="cmGrid"></div>
</div>

<!-- PR + F1 curves -->
<div class="section" id="curveplotsSection">
  <p class="section-label">PR &amp; F1 curves &mdash; fine-tuned runs &mdash; click to enlarge</p>
  <div class="img-grid" id="curveplotsGrid"></div>
</div>

<!-- Dataset distribution -->
<div class="section">
  <p class="section-label">Dataset &mdash; annotation distribution per split</p>
  <div class="chart-wrap" style="height:160px;"><canvas id="distChart"></canvas></div>
  <a class="chart-dl" onclick="downloadCanvas('distChart','dataset_distribution.png')">Download</a>
  <div class="legend" style="margin-top:10px;">
    <span class="legend-item"><span class="legend-swatch" style="background:#1D9E75;"></span>boat</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#378ADD;"></span>human</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#888780;"></span>other</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#D85A30;"></span>shark</span>
  </div>
</div>

<!-- Hyperparameter table -->
<div class="section" id="metaSection">
  <p class="section-label">Run hyperparameters</p>
  <div class="table-wrap"><table id="metaTable"></table></div>
</div>

<!-- Lightbox (shared by all image grids) -->
<div id="lightbox">
  <span id="lb-close">&times;</span>
  <img id="lb-img" src="" alt="">
</div>

<script>
// ── Canvas download helper ───────────────────────────────────────────────────
// Chart.js renders on a transparent background; we composite onto white
// before triggering the download so the PNG looks correct outside the browser.
function downloadCanvas(canvasId, filename) {
  const src  = document.getElementById(canvasId);
  const tmp  = document.createElement('canvas');
  tmp.width  = src.width;
  tmp.height = src.height;
  const ctx  = tmp.getContext('2d');
  ctx.fillStyle = isDark ? '#1a1a18' : '#ffffff';
  ctx.fillRect(0, 0, tmp.width, tmp.height);
  ctx.drawImage(src, 0, 0);
  const a    = document.createElement('a');
  a.href     = tmp.toDataURL('image/png');
  a.download = filename;
  a.click();
}

const DATA = __DATA__;

const PALETTE    = ['#B4B2A9', '#378ADD', '#1D9E75', '#D85A30', '#7F77DD', '#BA7517'];
const STOP_COLOR = 'rgba(210,120,30,0.60)';
const isDark     = matchMedia('(prefers-color-scheme: dark)').matches;
const gridColor  = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)';
const tickColor  = isDark ? '#888' : '#999';
const base       = { responsive: true, maintainAspectRatio: false };


// ── Lightbox (shared) ─────────────────────────────────────────────────────────
const lightbox = document.getElementById('lightbox');
const lbImg    = document.getElementById('lb-img');
function openLightbox(src) { lbImg.src = src; lightbox.classList.add('open'); }
lightbox.addEventListener('click', e => { if (e.target !== lbImg) lightbox.classList.remove('open'); });
document.getElementById('lb-close').addEventListener('click', () => lightbox.classList.remove('open'));
document.addEventListener('keydown', e => { if (e.key === 'Escape') lightbox.classList.remove('open'); });


// ── Image grid helper (confusion matrices + curve plots) ──────────────────────
function addImgItem(grid, label, b64, filename) {
  const src  = 'data:image/png;base64,' + b64;
  const item = document.createElement('div');
  item.innerHTML =
    '<p class="img-label">' + label + '</p>' +
    '<img class="img-thumb" src="' + src + '" alt="' + label + '">' +
    '<a class="img-download" href="' + src + '" download="' + filename + '">Download</a>';
  item.querySelector('img').addEventListener('click', () => openLightbox(src));
  grid.appendChild(item);
}


// ── Summary cards ─────────────────────────────────────────────────────────────
const bestIdx = DATA.runs.reduce((bi, r, i) =>
  r.overall[0] > DATA.runs[bi].overall[0] ? i : bi, 0);

DATA.runs.forEach((run, i) => {
  const isBest     = i === bestIdx && DATA.runs.length > 1;
  const isBaseline = run.is_baseline;
  const dimCls     = isBaseline ? ' dim' : '';
  const greenCls   = isBest ? ' green' : '';
  const subtitle   = isBaseline ? 'COCO pretrained'
                   : (run.label.match(/freeze(\d+)/)?.[0]?.replace('freeze', 'freeze=') ?? '');
  const card = document.createElement('div');
  card.className = 'run-card' + (isBest ? ' best' : '');
  card.innerHTML =
    '<p class="run-card-title">' + run.label + ' <span>' + subtitle + '</span></p>' +
    '<div class="metric-mini-grid">' +
      '<div><p class="metric-mini-label">mAP50</p>'     + '<p class="metric-mini-val' + dimCls + greenCls + '">' + run.overall[0].toFixed(3) + '</p></div>' +
      '<div><p class="metric-mini-label">Recall</p>'    + '<p class="metric-mini-val' + dimCls + greenCls + '">' + run.overall[3].toFixed(3) + '</p></div>' +
      '<div><p class="metric-mini-label">Precision</p>' + '<p class="metric-mini-val' + dimCls + '">'             + run.overall[2].toFixed(3) + '</p></div>' +
      '<div><p class="metric-mini-label">F1</p>'        + '<p class="metric-mini-val' + dimCls + '">'             + run.overall[4].toFixed(3) + '</p></div>' +
    '</div>';
  document.getElementById('runCards').appendChild(card);
});


// ── Shared bar chart options factory ─────────────────────────────────────────
function barOpts(min, max) {
  return {
    ...base,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: ctx => ' ' + ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(3) } }
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: tickColor, font: { size: 11 } } },
      y: { min: min, max: max, grid: { color: gridColor },
           ticks: { color: tickColor, font: { size: 11 }, stepSize: 0.2, callback: v => v.toFixed(1) } }
    }
  };
}

function appendLegend(el, runs, offset) {
  runs.forEach((run, i) => {
    el.innerHTML +=
      '<span class="legend-item"><span class="legend-swatch" style="background:' +
      PALETTE[(i + offset) % PALETTE.length] + ';"></span>' + run.label + '</span>';
  });
}

// ── Overall metrics bar chart ─────────────────────────────────────────────────
new Chart(document.getElementById('overallChart'), {
  type: 'bar',
  data: {
    labels: ['mAP50', 'mAP50-95', 'Prec', 'Recall', 'F1'],
    datasets: DATA.runs.map((run, i) => ({
      label: run.label, data: run.overall,
      backgroundColor: PALETTE[i % PALETTE.length], borderWidth: 0, borderRadius: 3,
    }))
  },
  options: barOpts(0, 1)
});
appendLegend(document.getElementById('overallLegend'), DATA.runs, 0);


// ── Per-class AP50 bar chart ──────────────────────────────────────────────────
const ftRuns = DATA.runs.filter(r => !r.is_baseline);
new Chart(document.getElementById('perClassChart'), {
  type: 'bar',
  data: {
    labels: ['boat', 'human', 'other', 'shark'],
    datasets: ftRuns.map((run, i) => ({
      label: run.label, data: run.per_class_ap50,
      backgroundColor: PALETTE[(i + 1) % PALETTE.length], borderWidth: 0, borderRadius: 3,
    }))
  },
  options: barOpts(0, 1)
});
appendLegend(document.getElementById('perClassLegend'), ftRuns, 1);


// ── Per-class recall bar chart ────────────────────────────────────────────────
const pcRecallCol   = document.getElementById('pcRecallCol');
const runsWithRecall = ftRuns.filter(r => r.per_class_recall);
if (runsWithRecall.length === 0) {
  pcRecallCol.style.display = 'none';
} else {
  new Chart(document.getElementById('pcRecallChart'), {
    type: 'bar',
    data: {
      labels: ['boat', 'human', 'other', 'shark'],
      datasets: runsWithRecall.map((run, i) => ({
        label: run.label, data: run.per_class_recall,
        backgroundColor: PALETTE[(i + 1) % PALETTE.length], borderWidth: 0, borderRadius: 3,
      }))
    },
    options: barOpts(0, 1)
  });
  appendLegend(document.getElementById('pcRecallLegend'), runsWithRecall, 1);
}


// ── Training curves ───────────────────────────────────────────────────────────
const curvesSection = document.getElementById('curvesSection');
const ftCurves = DATA.runs.filter(r => !r.is_baseline && r.curves);

if (ftCurves.length === 0) {
  curvesSection.style.display = 'none';
} else {
  // Hidden y-axis spanning 0-1 used by early-stop vertical lines so they
  // always span the full chart height regardless of the actual y scale.
  const yMarkAxis = { yMark: { display: false, min: 0, max: 1 } };

  function stopLine(run) {
    if (!run.curves.early_stopped) return null;
    return {
      label: '',
      data: [{ x: run.curves.stop_epoch, y: 0 }, { x: run.curves.stop_epoch, y: 1 }],
      yAxisID: 'yMark',
      borderColor: STOP_COLOR,
      borderWidth: 1.5, borderDash: [5, 3], pointRadius: 0, tension: 0, order: 99,
    };
  }

  function lineOpts(extraY, extraScales) {
    return {
      ...base,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: 'index', intersect: false,
          filter: item => item.dataset.label !== '',     // hide early-stop rows in tooltip
          callbacks: { label: ctx => ' ' + ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(4) }
        }
      },
      scales: Object.assign({
        x: { type: 'linear', grid: { color: gridColor },
             ticks: { color: tickColor, font: { size: 10 }, maxTicksLimit: 10 } },
        y: Object.assign({ grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 11 } } }, extraY)
      }, yMarkAxis, extraScales || {})
    };
  }

  // mAP50 — main line + best-epoch dot + optional early-stop line
  new Chart(document.getElementById('map50Chart'), {
    type: 'line',
    data: {
      datasets: ftCurves.flatMap((run, i) => {
        const color = PALETTE[(i + 1) % PALETTE.length];
        const best  = run.curves.best_epoch;
        const datasets = [
          {
            label: run.label,
            data: run.curves.epochs.map((e, j) => ({ x: e, y: run.curves.map50[j] })),
            borderColor: color, backgroundColor: 'transparent',
            borderWidth: 2, pointRadius: 0, tension: 0.3,
          },
          {  // Best-epoch marker
            label: run.label + ' best (ep ' + best + ', ' + run.curves.best_map50.toFixed(3) + ')',
            data: [{ x: best, y: run.curves.best_map50 }],
            borderColor: color, backgroundColor: color,
            pointRadius: 5, pointHoverRadius: 7, borderWidth: 2, showLine: false,
          }
        ];
        const sl = stopLine(run);
        if (sl) datasets.push(sl);
        return datasets;
      })
    },
    options: lineOpts({ min: 0, max: 1,
      ticks: { color: tickColor, font: { size: 11 }, callback: v => v.toFixed(2) } })
  });

  // Box loss — val solid, train dashed + early-stop line
  function lossDatasets(runs, valKey, trainKey) {
    return runs.flatMap((run, i) => {
      const color = PALETTE[(i + 1) % PALETTE.length];
      const datasets = [
        {
          label: run.label + ' val',
          data: run.curves.epochs.map((e, j) => ({ x: e, y: run.curves[valKey][j] })),
          borderColor: color, backgroundColor: 'transparent',
          borderWidth: 2, pointRadius: 0, tension: 0.3,
        },
        {
          label: run.label + ' train',
          data: run.curves.epochs.map((e, j) => ({ x: e, y: run.curves[trainKey][j] })),
          borderColor: color, backgroundColor: 'transparent',
          borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0, tension: 0.3,
        }
      ];
      const sl = stopLine(run);
      if (sl) datasets.push(sl);
      return datasets;
    });
  }

  new Chart(document.getElementById('boxLossChart'), {
    type: 'line',
    data: { datasets: lossDatasets(ftCurves, 'val_box', 'train_box') },
    options: lineOpts({})
  });

  new Chart(document.getElementById('clsLossChart'), {
    type: 'line',
    data: { datasets: lossDatasets(ftCurves, 'val_cls', 'train_cls') },
    options: lineOpts({})
  });

  // Curves legend
  const curveLegend = document.getElementById('curveLegend');
  ftCurves.forEach((run, i) => {
    curveLegend.innerHTML +=
      '<span class="legend-item"><span class="legend-swatch" style="background:' +
      PALETTE[(i + 1) % PALETTE.length] + ';"></span>' + run.label + '</span>';
  });
  curveLegend.innerHTML +=
    '<span class="legend-item" style="margin-left:8px;"><span class="legend-line-solid"></span>&nbsp;val</span>' +
    '<span class="legend-item"><span class="legend-line-dash"></span>&nbsp;train</span>';
  if (ftCurves.some(r => r.curves.early_stopped)) {
    curveLegend.innerHTML +=
      '<span class="legend-item"><span class="legend-line-stop"></span>&nbsp;early stop</span>';
  }
}


// ── Confusion matrices ────────────────────────────────────────────────────────
const cmSection = document.getElementById('cmSection');
const cmGrid    = document.getElementById('cmGrid');
DATA.runs.forEach(run => {
  if (!run.confusion_matrix_b64) return;
  const label = run.label + (run.is_baseline ? ' (baseline)' : '');
  addImgItem(cmGrid, label, run.confusion_matrix_b64, run.label + '_confusion_matrix.png');
});
if (!cmGrid.children.length) cmSection.style.display = 'none';


// ── PR + F1 curves ────────────────────────────────────────────────────────────
const curveplotsSection = document.getElementById('curveplotsSection');
const curveplotsGrid    = document.getElementById('curveplotsGrid');
DATA.runs.filter(r => !r.is_baseline).forEach(run => {
  if (run.pr_curve_b64)
    addImgItem(curveplotsGrid, run.label + ' PR curve', run.pr_curve_b64, run.label + '_PR_curve.png');
  if (run.f1_curve_b64)
    addImgItem(curveplotsGrid, run.label + ' F1 curve', run.f1_curve_b64, run.label + '_F1_curve.png');
});
if (!curveplotsGrid.children.length) curveplotsSection.style.display = 'none';


// ── Dataset distribution ──────────────────────────────────────────────────────
new Chart(document.getElementById('distChart'), {
  type: 'bar',
  data: {
    labels: ['train', 'val', 'test'],
    datasets: [
      { label: 'boat',  data: [482, 90, 38],     backgroundColor: '#1D9E75', borderWidth: 0 },
      { label: 'human', data: [9594, 2373, 1143], backgroundColor: '#378ADD', borderWidth: 0 },
      { label: 'other', data: [717, 158, 39],     backgroundColor: '#888780', borderWidth: 0 },
      { label: 'shark', data: [5328, 1314, 618],  backgroundColor: '#D85A30', borderWidth: 0 }
    ]
  },
  options: {
    ...base,
    plugins: { legend: { display: false }, tooltip: { mode: 'index' } },
    scales: {
      x: { stacked: true, grid: { display: false }, ticks: { color: tickColor, font: { size: 12 } } },
      y: { stacked: true, grid: { color: gridColor },
           ticks: { color: tickColor, font: { size: 11 }, callback: v => v >= 1000 ? (v/1000).toFixed(0)+'k' : v } }
    }
  }
});


// ── Hyperparameter table ──────────────────────────────────────────────────────
const metaSection = document.getElementById('metaSection');
const metaRuns    = DATA.runs.filter(r => r.metadata);
if (metaRuns.length === 0) {
  metaSection.style.display = 'none';
} else {
  const fmt  = v => (v === null || v === undefined) ? '&mdash;'
                  : (typeof v === 'number' && !Number.isInteger(v)) ? v.toPrecision(4)
                  : String(v);
  const keys = Object.keys(metaRuns[0].metadata);
  // A row is "different" if not all formatted values are identical across runs.
  const rowDiffers = k => {
    const vals = metaRuns.map(r => fmt(r.metadata[k]));
    return vals.some(v => v !== vals[0]);
  };
  document.getElementById('metaTable').innerHTML =
    '<thead><tr><th>Parameter</th>' + metaRuns.map(r => '<th>' + r.label + '</th>').join('') + '</tr></thead>' +
    '<tbody>' + keys.map(k => {
      const differs = rowDiffers(k);
      const style   = differs ? ' style="font-weight:600;"' : '';
      return '<tr><td class="meta-key"' + style + '>' + k + '</td>' +
        metaRuns.map(r => '<td' + style + '>' + fmt(r.metadata[k]) + '</td>').join('') + '</tr>';
    }).join('') + '</tbody>';
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Build runs data
# ---------------------------------------------------------------------------

def per_class_recall(m: dict) -> list | None:
    """Return per-class recall list if any class has data in the CSV, else None."""
    vals = [m.get(f"{c}_Recall") for c in CLASS_SUFFIXES]
    return vals if any(v is not None for v in vals) else None


runs = []

# Baseline
if BASELINE_CSV.exists():
    m       = load_metrics(BASELINE_CSV)
    run_dir = infer_run_dir(BASELINE_CSV)
    plots   = load_plots(run_dir)
    runs.append({
        "label":              "baseline",
        "is_baseline":        True,
        "overall":            [m.get(k, 0.0) for k in OVERALL_KEYS],
        "per_class_ap50":     [m.get(f"{c}_AP50", 0.0) for c in CLASS_SUFFIXES],
        "per_class_recall":   per_class_recall(m),
        "curves":             None,
        "metadata":           None,
        **plots,
    })
else:
    print(f"[warn] baseline CSV not found: {BASELINE_CSV}")

# Fine-tuned runs
for p in RUN_CSVS:
    if not p.exists():
        print(f"[warn] run CSV not found: {p}")
        continue
    m       = load_metrics(p)
    run_dir = infer_run_dir(p)
    parts   = p.parts
    label   = parts[parts.index("detect") + 1] if "detect" in parts else p.stem
    plots   = load_plots(run_dir)
    runs.append({
        "label":            label,
        "is_baseline":      False,
        "overall":          [m.get(k, 0.0) for k in OVERALL_KEYS],
        "per_class_ap50":   [m.get(f"{c}_AP50", 0.0) for c in CLASS_SUFFIXES],
        "per_class_recall": per_class_recall(m),
        "curves":           load_training_curves(run_dir),
        "metadata":         load_metadata(run_dir),
        **plots,
    })

if not runs:
    raise ValueError("No valid CSVs found - check BASELINE_CSV and RUN_CSVS paths above.")


# ---------------------------------------------------------------------------
# Render and save
# ---------------------------------------------------------------------------

html = HTML_TEMPLATE.replace("__DATA__", json.dumps({"runs": runs}, indent=2))
html = html.replace("__RUN_COUNT__", str(len(runs)))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(html, encoding="utf-8")
print(f"Dashboard saved to: {OUTPUT}")
print(f"  Runs: {[r['label'] for r in runs]}")
for r in runs:
    c = r.get("curves") or {}
    flags = []
    if c.get("early_stopped"): flags.append(f"early stop @ ep {c['stop_epoch']}")
    if r.get("confusion_matrix_b64"): flags.append("confusion matrix")
    if r.get("pr_curve_b64"):  flags.append("PR curve")
    if r.get("f1_curve_b64"):  flags.append("F1 curve")
    if r.get("per_class_recall"): flags.append("per-class recall")
    if flags:
        print(f"    {r['label']}: {', '.join(flags)}")