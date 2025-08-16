# statistic_loader.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, Optional
import numpy as np
import h5py
import warnings

class H5DB:
    """
    HDF5 loader of the statistics files (grid, 1st-order, 2nd-order).
    - db.get('u')                              -> ndarray
    - db.get_many('u','v','w')                 -> tuple(ndarray, ndarray, ndarray)
    - db.load_dict(['u','vT','u11_d11'])       -> {name: ndarray, ...}
    - db['/2nd-order/TT'] (absolute path)      -> ndarray
    """

    DEFAULT_PATHS: Dict[str, str] = {
        "y": "/grid/x2",
        "z": "/grid/x3",
        "u": "/1st-order/u1",
        "v": "/1st-order/u2",
        "w": "/1st-order/u3",
        "T": "/1st-order/T",
        "u.u": "/2nd-order/u11",
        "v.v": "/2nd-order/u22",
        "w.w": "/2nd-order/u33",
        "u.v": "/2nd-order/u12",
        "u.w": "/2nd-order/u13",
        "v.w": "/2nd-order/u23",
        "T.T": "/2nd-order/TT",
        "u.T": "/2nd-order/Tu1",
        "v.T": "/2nd-order/Tu2",
        "w.T": "/2nd-order/Tu3",
    }

    def __init__(self, path: str | Path,
                 extra_paths: Optional[Dict[str, str]] = None,
                 open_mode: str = "r"):

        self.path = str(path)
        self.file = h5py.File(self.path, open_mode)
        self.paths: Dict[str, str] = dict(self.DEFAULT_PATHS)
        if extra_paths:
            self.paths.update(extra_paths)

        self._cache: Dict[str, np.ndarray] = {}

        self.t_sample = self._get_attr_typed("t_sample", float)
        self.n_sample = self._get_attr_typed("n_sample", int)
        print(self.t_sample, self.n_sample)

        self._auto_register_derivatives()

    def close(self) -> None:
        if self.file:
            self.file.close()

    def __enter__(self) -> "H5DB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get(self, name: str) -> np.ndarray:
        print(name)
        # read if data are in the cache
        if name in self._cache:
            return self._cache[name]


        # Si produit croisé → ⟨ab⟩ - ⟨a⟩⟨b⟩
        if "." in name:
            a, b = name.split(".")
            ab_path = self._resolve(f"{a}.{b}")
            a_path = self._resolve(a)
            b_path = self._resolve(b)
            arr_ab = np.transpose(self.file[ab_path][:])
            arr_a = np.transpose(self.file[a_path][:])
            arr_b = np.transpose(self.file[b_path][:])
            result = arr_ab - arr_a * arr_b
            self._cache[name] = result
            return result

        path = self._resolve(name)
        if path not in self.file:
            raise KeyError(f"Missing dataset: {path} (for '{name}')")

        arr = np.transpose(self.file[path][...])
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        self._cache[name] = arr
        return arr

    def get_many(self, *names: str) -> Tuple[np.ndarray, ...]:
        return tuple(self.get(n) for n in names)

    def load_dict(self, names: Iterable[str]) -> Dict[str, np.ndarray]:
        return {n: self.get(n) for n in names}

    def __getitem__(self, name_or_path: str) -> np.ndarray:
        if name_or_path.startswith("/"):
            return self.get(name_or_path)
        return self.get(name_or_path)

    def available(self) -> Dict[str, str]:
        out = {}
        for k, p in self.paths.items():
            if p in self.file:
                out[k] = p
        return out

    def summary(self) -> str:
        lines = [f"H5DB: {self.path}"]
        if self.t_sample is not None or self.n_sample is not None:
            lines.append(f"  attrs: t_sample={self.t_sample}, n_sample={self.n_sample}")
        for k, p in sorted(self.available().items()):
            shape = self.file[p].shape
            lines.append(f"  {k:<8} -> {p}  shape={shape}")
        txt = "\n".join(lines)
        return txt

    def _resolve(self, name: str) -> str:
        if name.startswith("/"):
            return name
        if name in self.paths:
            return self.paths[name]
        raise KeyError(f"Unknown name '{name}'. Known: {', '.join(sorted(self.paths))} or HDF5 absolute path.")

    def _get_attr_typed(self, key: str, ty):
        try:
            val = self.file.attrs[key]
            if hasattr(val, "shape"):
                try:
                    val = val[()]
                except Exception:
                    pass
            return ty(val)
        except Exception:
            return None

    def _auto_register_derivatives(self) -> None:
        grp = "/2nd-order"
        if grp not in self.file:
            return
        g = self.file[grp]
        axis = ["x", "y", "z"]
        comp = ["u", "v", "w"]

        for i in (1, 2, 3):
            for d in (1, 2, 3):
                name = f"u{i}_d{d}"
                path = f"{grp}/{name}"
                label = f"{comp[i-1]}{axis[d-1]}"
                if path in g:
                    self.paths[label] = path

        for p in (1, 2, 3):
            for q in (1, 2, 3):
                for r in (1, 2, 3):
                    for s in (1, 2, 3):
                        name = f"u{p}{q}_d{r}{s}"
                        path = f"{grp}/{name}"
                        label = f"{comp[p-1]}{axis[r-1]}.{comp[q-1]}{axis[s-1]}"
                        if path in g:
                            self.paths[label] = path

        #for label in self.paths:
        #    print(label, " -> ", self.paths[label])
