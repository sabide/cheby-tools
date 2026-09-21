# cheby-tools

`cheby-tools` fournit un petit cœur Python pour manipuler des champs nodaux
sur des grilles tensorielles de Fourier et de Chebyshev en une, deux ou trois
dimensions. Son API publique principale est volontairement limitée à :

```python
from cheby_tools import Field, SpectralDiscretization
```

NumPy est la seule dépendance obligatoire. L’écriture Tecplot est une option
native séparée.

## Installation Python

Python 3.11 ou plus récent est requis.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Pour développer et construire les archives :

```bash
python -m pip install '.[dev]'
python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*
```

La wheel contient uniquement le paquet Python `cheby_tools`. L’extension
TecIO et ses dépendances C++ ne sont pas embarquées dans cette wheel.

## Grilles et convention de stockage

Une discrétisation associe à chaque axe ses bornes, son nombre de nœuds et sa
base. La base `fourier` représente un axe périodique sur l’intervalle
semi-ouvert correspondant ; la base `chebyshev` inclut les deux extrémités.

```python
import numpy as np
from cheby_tools import Field, SpectralDiscretization

grid = SpectralDiscretization(
    xmin=[0.0, -1.0],
    xmax=[2.0 * np.pi, 1.0],
    n=[64, 33],
    bases=["fourier", "chebyshev"],
)
x, y = grid.meshgrid()
temperature = Field(np.cos(x) + y, grid, "temperature")
```

Pour une grille de dimensions `(nx, ny, nz)`, les valeurs d’un `Field` ont
exactement la forme `(nx, ny, nz)`. L’axe NumPy 0 correspond donc au premier
axe physique, l’axe 1 au deuxième, etc. `grid.meshgrid()` applique la même
convention avec `indexing="ij"`.

## Dérivation et interpolation

Les opérations créent un nouveau `Field` et ne modifient pas les valeurs du
champ d’origine :

```python
dtemperature_dx = temperature.derivative(axis=0)
d2temperature_dy2 = temperature.derivative(axis=1, order=2)

fine = SpectralDiscretization(
    [0.0, -1.0],
    [2.0 * np.pi, 1.0],
    [128, 65],
    ["fourier", "chebyshev"],
)
temperature_fine = temperature.interpolate(fine)
```

Un exemple analytique exécutable vérifie les deux opérations :

```bash
python examples/field_quickstart.py
```

## Sortie Tecplot `.plt`

L’adaptateur optionnel écrit un ou plusieurs champs réels définis sur la même
grille dans un fichier Tecplot binaire classique `.plt`. Il ne produit jamais
de fichier `.szplt`.

```python
import numpy as np
from cheby_tools import Field, SpectralDiscretization
from cheby_tools.tecio import write_plt

grid = SpectralDiscretization(
    [0.0, -1.0], [2.0 * np.pi, 1.0], [64, 33],
    ["fourier", "chebyshev"],
)
x, y = grid.meshgrid()
u = Field(np.sin(x) * (1.0 - y**2), grid, "u")
temperature = Field(np.cos(2.0 * x) + y, grid, "temperature")
write_plt("fields.plt", [u, temperature])
```

Le même code se trouve dans `examples/write_plt.py`.

### Construction de l’extension TecIO

Initialiser d’abord les sous-modules, puis installer le cœur Python et
l’extension dans le même environnement :

```bash
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .

export CHEBY_PYTHON_ENV="$PWD/.venv"
export CHEBY_BOOST_INCLUDE_DIR="$PWD/external/boost"
./run_cmake.sh
python examples/write_plt.py
```

`run_cmake.sh` utilise `build/` par défaut. Un autre répertoire peut être
choisi avec `CHEBY_BUILD_DIR`. Pour lier une installation TecIO existante au
lieu du sous-module fourni, configurer CMake avec
`CHEBY_USE_BUNDLED_TECIO=OFF`, `CHEBY_TECIO_INCLUDE_DIR` et
`CHEBY_TECIO_LIBRARY`.

## Installation sur ADASTRA

Le chemin de l’environnement Python appartient à l’utilisateur et doit être
fourni explicitement. Par exemple :

```bash
module purge
module load python/3.12.1
export CHEBY_PYTHON_ENV="$WORK/venvs/cheby-tools"
python -m venv "$CHEBY_PYTHON_ENV"

source env.sh
python -m pip install .
python examples/field_quickstart.py
```

Pour ajouter TecIO, initialiser les sous-modules puis lancer le script de
construction dans le même environnement :

```bash
git submodule update --init --recursive
source env.sh
./run_cmake.sh
python examples/write_plt.py
```

`env.sh` charge la pile compilateur/CMake prévue pour ADASTRA, active
`CHEBY_PYTHON_ENV` et vérifie Python ainsi que les en-têtes Boost. Aucun
chemin de compte utilisateur n’est codé en dur.
