#!/usr/bin/env python3
"""Validate the Python environment used to build and run cheby-tools."""

from __future__ import annotations

import os
import sys
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

if sys.version_info[:2] != (3, 12):
    fail(f"Python 3.12 is required, got {sys.version.split()[0]}")

import numpy
import h5py
from mpi4py import MPI

if not h5py.get_config().mpi:
    fail("h5py was built without MPI support")

if h5py.version.hdf5_version_tuple[:2] != (1, 14):
    fail(f"HDF5 1.14.x is required, got {h5py.version.hdf5_version}")

print(f"[OK] Python : {sys.version.split()[0]} ({sys.executable})")
print(f"[OK] NumPy  : {numpy.__version__}")
print(f"[OK] h5py   : {h5py.__version__}, HDF5 {h5py.version.hdf5_version}, MPI enabled")
print(f"[OK] mpi4py : {MPI.Get_version()}, vendor={MPI.get_vendor()[0]}")

# Installed cheby-tools packages are checked only after cmake --install.
# Do not let the source tree shadow packages installed in the environment.
source_root = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or os.getcwd()).resolve() != source_root
]

for module_name in ("discr", "stats", "spec_forge", "tecio_wrapper"):
    try:
        module = __import__(module_name)
    except ModuleNotFoundError:
        print(f"[INFO] {module_name}: not installed yet")
    else:
        print(f"[OK] {module_name}: {Path(module.__file__).resolve()}")
