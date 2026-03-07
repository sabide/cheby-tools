"""Unified imports for cheby-tools-adastra."""

try:
    from . import discr as discr
    from . import stats as stats
    from . import tecio_wrapper as tecio
except ImportError:
    # Backward-compatible layout where modules are installed at top-level.
    import discr as discr
    import stats as stats
    import tecio_wrapper as tecio

__all__ = ["discr", "stats", "tecio"]
