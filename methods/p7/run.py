"""P7 corrected rerun wrapper; formal commands remain explicitly gated."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


COMMANDS = {
    "preflight": [sys.executable, "methods/p7/preflight.py"],
    "m1-smoke": [sys.executable, "-m", "methods.m1.run_smoke",
                 "--config", "methods/p7/m1_smoke.corrected.json",
                 "--output-root", "results/p7_corrected/m1"],
    "m2-smoke": [sys.executable, "methods/p7/m2_smoke.py",
                 "--config", "methods/p7/m2_corrected.yaml",
                 "--output", "results/p7_corrected/m2_smoke"],
    "m3-preflight": [sys.executable, "methods/m3/run.py", "preflight",
                     "--config", "methods/p7/m3_corrected.json"],
    "m3-smoke": [sys.executable, "methods/m3/run.py", "smoke",
                 "--config", "methods/p7/m3_corrected.json",
                 "--output", "results/p7_corrected/m3_smoke"],
    "m1-full-plan": [sys.executable, "methods/p7/run_m1_full.py", "plan"],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args(argv)
    raise SystemExit(subprocess.run(COMMANDS[args.command], cwd=ROOT).returncode)


if __name__ == "__main__":
    main()

