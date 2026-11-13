import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUICK = ROOT / "results" / "quick"


def _run(dst):
    subprocess.run([sys.executable, str(ROOT / "code" / "main.py"), "--quick"], check=True,
                   cwd=ROOT)
    shutil.copytree(QUICK, dst)


def test_quick_runs_are_identical(tmp_path):
    _run(tmp_path / "a")
    _run(tmp_path / "b")
    # runtime*.csv hold wall-clock timings (E7) and are not expected to repeat bit by bit
    csvs = sorted(p.name for p in (tmp_path / "a").glob("*.csv") if not p.name.startswith("runtime"))
    assert csvs
    match, mismatch, errors = filecmp.cmpfiles(tmp_path / "a", tmp_path / "b", csvs, shallow=False)
    assert not mismatch and not errors
