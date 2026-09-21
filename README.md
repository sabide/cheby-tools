# cheby-tools

`cheby-tools` fournit un petit cœur Python pour manipuler des champs nodaux
sur des grilles tensorielles de Fourier et de Chebyshev en une, deux ou trois
dimensions. Son API publique principale est volontairement limitée à :

```python
from cheby_tools import Field, SpectralDiscretization
```

NumPy est la dépendance Python d’exécution. Le backend natif TecIO est compilé
et installé par défaut.

## Installation

Les plateformes prises en charge sont Linux, dont ADASTRA, et macOS. Il faut
Python 3.11 ou plus récent, CMake 3.18 ou plus récent, un compilateur C++ et un
accès à l’index ou au cache `pip` contenant `scikit-build-core`.

Le clonage récursif initialise le sous-module pybind11. TecIO et les en-têtes
Boost sont déjà fournis dans le dépôt : CMake ne télécharge aucune dépendance.

```bash
cd <parent-directory>
git clone --recurse-submodules -b update \
  git@github.com:sabide/cheby-tools.git cheby-tools-update
python3 -m venv post-processing/.venv
source post-processing/.venv/bin/activate
python -m pip install -e ./cheby-tools-update
```

La dernière commande installe `cheby-tools` et `_tecio` dans le venv actif du
projet de post-traitement. Les modifications Python sont visibles
immédiatement. Après une modification C++ ou CMake, relancer la même commande
recompile `_tecio`.

Pour installer uniquement le cœur Python :

```bash
python -m pip install -e ./cheby-tools-update \
  -Ccmake.define.CHEBY_INSTALL_TECIO=OFF
```

`Field` et `SpectralDiscretization` restent alors disponibles. Un appel à
`write_plt` lève une `ImportError` qui indique comment réinstaller le backend.

Pour développer le paquet et construire ses archives depuis le dépôt :

```bash
cd cheby-tools-update
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*
```

La wheel produite est native à la plateforme et contient le paquet Python
`cheby_tools` ainsi que son extension privée `_tecio`.

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

L’installation principale compile déjà l’extension TecIO. Après activation du
venv, l’exemple peut être lancé depuis le dépôt :

```bash
python examples/write_plt.py
```

## Installation sur ADASTRA

Charger d’abord la pile de compilation, puis créer le venv du projet de
post-traitement et lancer la même installation :

```bash
module purge
module load cpe/24.07
module load PrgEnv-gnu/8.5.0
module load cmake/4.0.3
module load python/3.12.1
python -m venv post-processing/.venv
source post-processing/.venv/bin/activate
python -m pip install -e ./cheby-tools-update
```

Le dépôt doit avoir été cloné avec `--recurse-submodules`. Pour corriger un
clonage existant incomplet :

```bash
git submodule update --init --recursive
```

### Workflow CMake direct avancé

Le script historique reste disponible pour déboguer le build natif ou choisir
un préfixe CMake manuellement. Il n’est pas nécessaire pour l’installation
normale avec `pip`.

```bash
cd cheby-tools-update
export CHEBY_PYTHON_ENV=<absolute-path>/post-processing/.venv
source env.sh
./run_cmake.sh
```

`run_cmake.sh` utilise `build/` par défaut. `CHEBY_BUILD_DIR` permet de choisir
un autre répertoire. Pour lier une installation TecIO externe, configurer
`CHEBY_USE_BUNDLED_TECIO=OFF`, `CHEBY_TECIO_INCLUDE_DIR` et
`CHEBY_TECIO_LIBRARY`. Aucun chemin de compte utilisateur n’est codé en dur.
