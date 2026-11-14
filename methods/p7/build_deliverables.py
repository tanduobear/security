"""Collect original and accepted corrected formal results; missing results stop export."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_results import main

if __name__ == '__main__':
    main()
