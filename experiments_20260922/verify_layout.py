#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

checks = []

def add(name, ok, detail):
    checks.append((name, bool(ok), str(detail)))

add("NFlow/Python/model2.py", (ROOT / "Python" / "model2.py").exists(), ROOT / "Python" / "model2.py")
add("NFlow/Python/bnn_metric.py", (ROOT / "Python" / "bnn_metric.py").exists(), ROOT / "Python" / "bnn_metric.py")
for pkg in ["numpy", "pandas", "scipy", "sklearn", "torch"]:
    add(f"Python package: {pkg}", importlib.util.find_spec(pkg) is not None, "installed" if importlib.util.find_spec(pkg) else "missing")
add("Rscript", shutil.which("Rscript") is not None, shutil.which("Rscript") or "not on PATH")
add("git", shutil.which("git") is not None, shutil.which("git") or "not on PATH")
repo = HERE / "external_methods" / "SS_Group_Shrinkage_New"
add("SS-GL/SS-GHS official repo", repo.exists(), repo)

width = max(len(x[0]) for x in checks)
for name, ok, detail in checks:
    print(f"{'OK' if ok else 'MISSING':7s}  {name:<{width}s}  {detail}")

required = [checks[0], checks[1]]
if not all(x[1] for x in required):
    raise SystemExit("\nPlace this folder directly under the NFlow repository root before running DSS-LVR.")
