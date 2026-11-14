# P7 corrected rerun preparation

## Cross-method deliverables

After the formal result files are present, regenerate P7's auditable summary and figures with:

```powershell
python methods/p7/build_deliverables.py
```

This writes `results/summary.csv`, `results/p7_figure_data.json`, and four PNG/PDF figure pairs under `figures/`. Corrected full-run metric cells remain blank until those exported files are synchronized into this checkout; the script never infers or fabricates missing deltas.

For the compact paper-style figures and the tidy values table used to draw them, run:

```powershell
python methods/p7/build_paper_figures.py
```

This writes `results/p7_plot_values.csv` and the four `fig_p7_*` PNG/PDF pairs. These use the same DejaVu Sans typography, light grid, colour-blind-safe palette, 300-DPI PNG export, and vector PDF export as the team's M1 figure style.

This directory isolates P7's corrected-dataset rerun from the completed P4/P5/P6 experiments.
Nothing here reuses or rewrites the historical RF approval records.

## Safety rules

- Only the dataset version changes from `original` to `corrected`.
- Seeds, model hyperparameters, threshold rules, and evaluation settings remain unchanged.
- All generated artifacts go under `results/p7_corrected/`.
- Do not run `methods.m1.run_rf trial`; that is the completed P4 RF resource trial.
- Run preflight and smoke checks before any full training.

## Commands

From the repository root with `venv_p7` activated:

```powershell
python methods/p7/run.py preflight
python methods/p7/run.py m1-smoke
python methods/p7/run.py m2-smoke
python methods/p7/run.py m3-preflight
python methods/p7/run.py m3-smoke
```

## Teammate full rerun handoff

First verify the M1 formal plan. This command never opens a dataset and never trains:

```powershell
python methods/p7/run.py m1-full-plan
```

Before approval it must report `blocked_pending_approval`. After the content-addressed
`execution.m1-corrected.v1.json` record has been explicitly approved and committed, the
same command reports `approved_ready`. Only then may the teammate start M1 with:

```powershell
python methods/p7/run_m1_full.py run --confirm-start
```

The M1 runner fits only G05 and RF_FIXED for seeds 42--46. It freezes all validation
predictions and thresholds before constructing the Test loader. It never reuses or edits
the historical P4 RF approval. Every run is written to a new directory below
`results/p7_corrected/m1/formal/`; existing runs are never overwritten.

M2 and M3 use their existing formal entry points and isolated P7 outputs:

```powershell
python methods/m2/train.py --config methods/p7/m2_corrected.yaml
python methods/m2/evaluate.py --config methods/p7/m2_corrected.yaml

python methods/m3/run.py preflight --config methods/p7/m3_corrected.json
python methods/m3/run.py search --config methods/p7/m3_corrected.json --output results/p7_corrected/m3
python methods/m3/run.py evaluate --config methods/p7/m3_corrected.json --output results/p7_corrected/m3
```

The corrected parquet files and verified preprocessing artifacts must already be present
on the teammate's computer; Git is only used to transfer code, configurations and approval
records, not the large local dataset or generated result directories.

## Prepared configurations

- `m1_smoke.corrected.json`: corrected train/validation development smoke check.
- `m1_corrected.json`: frozen P7 M1 rerun specification; not an RF approval record.
- `m2_corrected.yaml`: P5 M2 parameters with corrected data and isolated outputs.
- `m3_corrected.json`: P6 M3 parameters with corrected data and isolated outputs.

