"""Statistics helpers with optional HDF5 support."""

from .Sebilleau_loader import (
    load_dns_database_sebilleau,
    wall_profile,
    wall_profiles,
)

__all__ = [
    "H5DB",
    "dotify_vars",
    "load_dns_database_sebilleau",
    "wall_profile",
    "wall_profiles",
]

_HDF5_EXPORTS = {"H5DB", "dotify_vars"}


def __getattr__(name):
    if name not in _HDF5_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from . import statistics_loader
    except ModuleNotFoundError as exc:
        if exc.name != "h5py":
            raise
        raise ImportError(
            f"stats.{name} requires h5py; install it with "
            "`python -m pip install 'cheby-tools[io]'`."
        ) from exc

    value = getattr(statistics_loader, name)
    globals()[name] = value
    return value
