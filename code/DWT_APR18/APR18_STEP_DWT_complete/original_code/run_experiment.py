"""Run all three stages in order without editing any source data."""
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
for stage in ['01_audit_sources.py', '02_extract_and_compare.py', '03_plot_results.py']:
    print(f'Running {stage}')
    runpy.run_path(str(root / stage), run_name='__main__')
