"""Render P7 figures from the collected, audited metric values."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_collected import main

if __name__ == '__main__':
    main()
