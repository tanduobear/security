"""Read-only P7 corrected-rerun validation. No dataset rows or models are loaded."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "results/p7_corrected"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def without(value, *paths):
    value = json.loads(json.dumps(value))
    for path in paths:
        parent = value
        for key in path[:-1]:
            parent = parent[key]
        parent.pop(path[-1], None)
    return value


def parquet_status(path):
    path = Path(path)
    magic = b""
    if path.is_file():
        with path.open("rb") as stream:
            magic = stream.read(4)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "exists": path.is_file(),
        "real_parquet": path.is_file() and magic == b"PAR1",
        "bytes": path.stat().st_size if path.is_file() else None,
    }


def main():
    checks = []

    def check(name, passed, details=None):
        checks.append({"name": name, "passed": bool(passed), "details": details})

    m1_base = load_json(ROOT / "methods/m1/config_smoke.json")
    m1 = load_json(HERE / "m1_smoke.corrected.json")
    check("M1 smoke changes only dataset_directory",
          without(m1_base, ("dataset_directory",)) == without(m1, ("dataset_directory",)) and
          m1["dataset_directory"] == "data/split/corrected/stratified")

    m2_base = load_yaml(ROOT / "methods/m2/config.yaml")
    m2 = load_yaml(HERE / "m2_corrected.yaml")
    check("M2 model settings unchanged",
          without(m2_base, ("experiment_name",), ("data", "version"), ("output",)) ==
          without(m2, ("experiment_name",), ("data", "version"), ("output",),
                  ("expected_scale_pos_weight",)) and m2["data"]["version"] == "corrected")
    check("M2 output isolated",
          all(str(value).replace("\\", "/").startswith("results/p7_corrected/")
              for value in m2["output"].values()))

    m3_base = load_json(ROOT / "methods/m3/config.json")
    m3 = load_json(HERE / "m3_corrected.json")
    check("M3 changes only dataset version",
          without(m3_base, ("dataset", "version")) == without(m3, ("dataset", "version")) and
          m3["dataset"]["version"] == "corrected")

    spec = load_json(HERE / "m1_corrected.json")
    check("M1 full rerun remains blocked",
          spec["status"] == "prepared_not_executed" and
          spec["training_authorized"] is False and
          spec["historical_rf_approval_reused"] is False)
    gbdt = load_json(ROOT / "methods/m1/execution.gbdt-search.v1.json")
    rf = load_json(ROOT / "methods/m1/execution.rf-only.proposed.json")
    original_g05 = next(row["params"] for row in gbdt["candidates"] if row["id"] == "G05")
    check("M1 G05 parameters copied exactly",
          spec["models"]["G05"]["params"] == original_g05 and spec["seeds"] == gbdt["seeds"])
    check("M1 RF parameters copied exactly",
          spec["models"]["RF_FIXED"]["params"] == rf["parameters"] and spec["seeds"] == rf["seeds"])

    data = {}
    for split in ("train", "val", "test"):
        status = parquet_status(ROOT / "data/split/corrected/stratified" / f"{split}.parquet")
        data[split] = status
        check(f"corrected {split} is real parquet", status["real_parquet"], status)

    preprocessor = load_json(ROOT / "data/split/corrected/stratified/preprocessor.json")
    actual_weight = float(preprocessor["scale_pos_weight"])
    expected_weight = float(m2["expected_scale_pos_weight"])
    check("M2 corrected class weight pinned", abs(actual_weight - expected_weight) <= 1e-12,
          {"actual": actual_weight, "expected": expected_weight})

    run_m3 = (ROOT / "methods/m3/run.py").read_text(encoding="utf-8")
    check("M3 calibration uses benign train/validation",
          "read_data(cfg, 'train', benign_only=True" in run_m3 and
          "read_data(cfg, 'val', benign_only=True" in run_m3 and
          "未读取测试集" in run_m3)

    required = ["numpy", "pandas", "pyarrow", "duckdb", "sklearn", "jsonschema",
                "matplotlib", "yaml", "torch"]
    modules = {name: importlib.util.find_spec(name) is not None for name in required}
    check("Python is 3.11 or newer", sys.version_info >= (3, 11), platform.python_version())
    check("runtime modules installed", all(modules.values()), modules)

    handoff = ROOT / "submission/P6_handoff.zip"
    receipt = ROOT / "submission/P6_handoff.sha256.txt"
    if handoff.is_file() and receipt.is_file():
        expected = receipt.read_text(encoding="utf-8").split()[0].lower()
        digest = hashlib.sha256()
        with handoff.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        check("P6 handoff ZIP hash", actual == expected, {"expected": expected, "actual": actual})
    else:
        check("P6 handoff ZIP hash", False, "handoff or receipt missing")

    config_ok = all(item["passed"] for item in checks if item["name"] != "runtime modules installed")
    environment_ok = next(item["passed"] for item in checks if item["name"] == "runtime modules installed")
    report = {
        "scope": "read_only_preflight_no_training_no_dataset_rows_loaded",
        "python": sys.executable,
        "config_and_data_ready": config_ok,
        "environment_ready": environment_ok,
        "ready_for_smoke": config_ok and environment_ok,
        "output_root": OUTPUT_ROOT.relative_to(ROOT).as_posix(),
        "checks": checks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_smoke"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

