#!/usr/bin/env python3
"""Validate the Python environment used to build and run cheby-tools."""

from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


expected_env = os.environ.get("CHEBY_PYTHON_ENV")
if not expected_env:
    fail("CHEBY_PYTHON_ENV is not set; source env.sh first")

expected_prefix = Path(expected_env).resolve()
actual_prefix = Path(sys.prefix).resolve()
if actual_prefix != expected_prefix:
    fail(f"Python prefix is {actual_prefix}, expected {expected_prefix}")

if sys.version_info < (3, 11):
    fail(f"Python 3.11 or newer is required, got {sys.version.split()[0]}")

# Installed cheby-tools packages are checked only after cmake --install.
# Do not let the source tree shadow packages installed in the environment.
source_root = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or os.getcwd()).resolve() != source_root
]

import numpy
import cheby_tools

print(f"[OK] Python      : {sys.version.split()[0]} ({sys.executable})")
print(f"[OK] NumPy       : {numpy.__version__}")
print(f"[OK] cheby-tools : {Path(cheby_tools.__file__).resolve()}")

if importlib.util.find_spec("cheby_tools._tecio") is None:
    print("[INFO] TecIO backend: not installed")
else:
    print("[OK] TecIO backend: cheby_tools._tecio")
