# P6 M3 delivery

The selected autoencoder is A06. Eight configurations and five seeds produced 40 fits. Original stratified main test evaluation contains five selected-model seeds at two validation-benign quantiles, plus one fit each of Isolation Forest and One-Class SVM.

- `summary.json`: small public summary, including five-seed means and sample standard deviations, per-seed metrics, auxiliary results, timing and provenance. Curve arrays are omitted, so this file is not a P3 single-record schema document.
- `search.json`: all 40 training histories, validation-only ranking, configuration and checkpoint hashes.
- `../../submission/P6_handoff.zip`: Git LFS archive containing full P3 records, models, scores with real row IDs, threshold logs, source and manuscript.
- `../../submission/P6_handoff.sha256.txt`: archive SHA256. The ZIP also contains a per-file manifest.

## Obtain full records

From the repository root, use Git LFS to download only the P6 archive, verify its SHA256 against the text file, and extract to a separate review directory:

```powershell
git lfs pull --include="submission/P6_handoff.zip" --exclude=""
Get-FileHash submission/P6_handoff.zip -Algorithm SHA256
Get-Content submission/P6_handoff.sha256.txt
Expand-Archive -LiteralPath submission/P6_handoff.zip -DestinationPath .tmp/p6-review
```

All `results/m3_runs/...` references in the result index resolve relative to the extracted directory. Full records remain exactly as evaluated; no metrics were edited or recomputed during publication. Raw P2 datasets are not included. P7 must obtain the matching datasets separately to replay models.

## Scope

q95 and q99 are fixed validation-benign thresholds, not test-selected ROC thresholds. Auxiliary models use one common seed-42 sample of 10,000 training negatives. Cross-method paired bootstrap, corrected-version reruns and P7 independent acceptance remain downstream work. P5 validation checkpoint replay passed its checked tolerances; this does not establish full retraining acceptance.
