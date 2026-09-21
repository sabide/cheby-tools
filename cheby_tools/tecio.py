"""Optional TecIO output for :class:`cheby_tools.Field` objects."""

from pathlib import Path

import numpy as np

from .field import Field

try:
    from . import _tecio as _backend
except ImportError as exc:
    _backend = None
    _backend_import_error = exc
else:
    _backend_import_error = None


def _same_grid(left, right):
    if (
        left.dim != right.dim
        or left.n != right.n
        or left.bases != right.bases
        or left.xmin != right.xmin
        or left.xmax != right.xmax
    ):
        return False
    return all(
        np.array_equal(left_nodes, right_nodes)
        for left_nodes, right_nodes in zip(left.nodes, right.nodes)
    )


def _normalize_fields(fields):
    if isinstance(fields, Field):
        return [fields]
    try:
        fields = list(fields)
    except TypeError as exc:
        raise TypeError(
            "fields must be a Field or an iterable of Field objects."
        ) from exc
    if not fields:
        raise ValueError("fields must contain at least one Field.")
    for index, field in enumerate(fields):
        if not isinstance(field, Field):
            raise TypeError(f"fields[{index}] must be a Field.")
    return fields


def write_plt(path, fields):
    """Write one or more compatible real fields to classic Tecplot ``.plt``."""
    path = Path(path)
    if path.suffix != ".plt":
        raise ValueError(f"TecIO path must end in '.plt', got {path!s}.")

    fields = _normalize_fields(fields)
    grid = fields[0].discretization
    coordinate_names = ["x", "y", "z"][: grid.dim]
    names = [field.name for field in fields]
    for field in fields:
        if "\0" in field.name or "\n" in field.name:
            raise ValueError(
                f"TecIO field name {field.name!r} must not contain NUL or "
                "newline characters."
            )
        try:
            encoded_name = field.name.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(
                f"TecIO field name {field.name!r} is not valid UTF-8."
            ) from exc
        if len(encoded_name) > 128:
            raise ValueError(
                f"TecIO field name {field.name!r} exceeds 128 UTF-8 bytes "
                f"({len(encoded_name)} bytes)."
            )
    if len(set(names)) != len(names):
        raise ValueError("Field names must be unique.")
    collisions = set(names).intersection(coordinate_names)
    if collisions:
        raise ValueError(
            f"Field names conflict with coordinates: {sorted(collisions)}."
        )

    for field in fields:
        if not _same_grid(grid, field.discretization):
            raise ValueError(f"Field {field.name!r} uses an incompatible grid.")
        if np.iscomplexobj(field.values):
            raise ValueError(
                f"Field {field.name!r} is complex and cannot be written."
            )

    if _backend is None:
        raise ImportError(
            "TecIO backend is not installed. Reinstall with: "
            "python -m pip install -e <cheby-tools-path>"
        ) from _backend_import_error

    axes = tuple(range(grid.dim - 1, -1, -1))
    source_arrays = [*grid.meshgrid(), *(field.values for field in fields)]
    arrays = [
        np.ascontiguousarray(np.transpose(array, axes=axes), dtype=np.float64)
        for array in source_arrays
    ]
    _backend.write_plt(str(path), coordinate_names + names, arrays)


__all__ = ["write_plt"]
