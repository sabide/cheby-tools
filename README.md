# cheby-tools

Outils Python de post-traitement spectral sur grilles de Chebyshev et de
Fourier. Le cœur Python est installable indépendamment de TecIO et d'un
compilateur C++.

L'API spectrale prise en charge est :

```python
from spec_forge import SpectralDiscretization, SpectralInterpolate
```

Le paquet historique `discr` reste disponible comme façade de compatibilité,
mais les nouveaux scripts doivent utiliser `spec_forge`.

## Installation du cœur Python

Prérequis : Python 3.11 ou une version ultérieure, avec `pip`. Les versions
3.11 et 3.12 sont qualifiées par les tests actuels.

### Linux et macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

NumPy est la seule dépendance obligatoire. Les lecteurs HDF5 de `stats`
nécessitent l'extra `io` :

```bash
python -m pip install '.[io]'
```

L'extra `dev` fournit les outils de construction et de validation des
archives :

```bash
python -m pip install '.[dev]'
```

## Vérification rapide

Depuis la racine des sources :

```bash
python -m pip install '.[dev]'
python examples/spectral_quickstart.py
python -m unittest discover -s tests -v
```

L'exemple construit deux grilles périodiques, dérive un champ analytique et
l'interpole sur une grille plus fine. Il termine avec un code non nul si
l'erreur dépasse la tolérance numérique.

## Construction d'une wheel et d'une sdist

```bash
python -m pip install '.[dev]'
python -m build
python -m twine check dist/*
```

Les deux archives sont créées dans `dist/` :

```text
cheby_tools-0.1.0-py3-none-any.whl
cheby_tools-0.1.0.tar.gz
```

La wheel contient uniquement `spec_forge`, `discr` et `stats`. TecIO, Boost
et les autres sources tierces ne font pas partie de cette distribution
Python.

## Installation sur ADASTRA

Créer l'environnement dans un emplacement de travail persistant choisi par
l'utilisateur ; ne pas coder en dur le chemin d'un autre compte :

```bash
python3 -m venv /chemin/vers/venvs/cheby-tools
source /chemin/vers/venvs/cheby-tools/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python examples/spectral_quickstart.py
```

Pour une installation hors ligne, préparer un répertoire de wheels sur une
machine ayant accès à l'index Python :

```bash
python -m pip wheel --wheel-dir wheelhouse .
```

Pour préparer également h5py et ses dépendances :

```bash
python -m pip wheel --wheel-dir wheelhouse '.[io]'
```

Après transfert de `wheelhouse/` sur ADASTRA :

```bash
python -m pip install --no-index --find-links wheelhouse cheby-tools
# Avec les lecteurs HDF5 :
python -m pip install --no-index --find-links wheelhouse 'cheby-tools[io]'
```

La wheel de `cheby-tools` est pure Python, mais NumPy et h5py contiennent des
composants natifs. Les wheels déposées dans `wheelhouse/` doivent donc être
compatibles avec la version de Python et la plate-forme cibles. Pour h5py
parallèle/MPI, conserver un environnement HPC séparé et utiliser la pile
logicielle qualifiée du site.

## TecIO et installation CMake optionnelle

Le wrapper TecIO n'est pas construit par `pip`. Il conserve son installation
CMake séparée :

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix build/install
export PYTHONPATH="$PWD/build/install/lib/python${PYTHONPATH:+:$PYTHONPATH}"
python -c "import tecio_wrapper; print(tecio_wrapper.__file__)"
```

Pour désactiver TecIO et utiliser seulement l'installation CMake historique
des modules Python :

```bash
cmake -S . -B build \
  -DCHEBY_INSTALL_TECIO_WRAPPER=OFF \
  -DCHEBY_INSTALL_POSTPROCESSING_TOOLS=ON
cmake --install build --prefix build/install
```

Pour ne construire que TecIO :

```bash
cmake -S . -B build \
  -DCHEBY_INSTALL_POSTPROCESSING_TOOLS=OFF
cmake --build build -j
cmake --install build --prefix build/install
```

Pour utiliser une bibliothèque TecIO déjà construite :

```bash
cmake -S . -B build \
  -DCHEBY_USE_BUNDLED_TECIO=OFF \
  -DCHEBY_TECIO_INCLUDE_DIR=/chemin/vers/teciosrc \
  -DCHEBY_TECIO_LIBRARY=/chemin/vers/libtecio.a
cmake --build build -j
```

Les en-têtes Boost 1.88 vendus dans `external/boost` sont sélectionnés par
défaut. Une autre installation peut être imposée avec :

```bash
cmake -S . -B build \
  -DCHEBY_BOOST_INCLUDE_DIR=/chemin/vers/boost-root
```

Ce chemin doit contenir `boost/version.hpp`.

Le fichier `cfg_adastra.sh` conserve les options CMake du profil HPC existant.
Le cœur spectral installé avec `pip` ne dépend pas de ce profil.
