#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <venv-path>" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

requested_venv=$1
if [[ -e "$requested_venv" && ! -x "$requested_venv/bin/python" ]]; then
    echo "Refusing to overwrite $requested_venv: not a valid Python virtual environment" >&2
    exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

if ! command -v module >/dev/null 2>&1; then
    for module_init in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        if [[ -r "$module_init" ]]; then
            # shellcheck source=/dev/null
            source "$module_init"
            break
        fi
    done
fi

if ! command -v module >/dev/null 2>&1; then
    echo "The environment-modules command is required on ADASTRA" >&2
    exit 1
fi

module purge
module load cpe/24.07
module load PrgEnv-gnu/8.5.0
module load cmake/4.0.3
module load python/3.12.1

if ! c_compiler="$(command -v cc)"; then
    echo "ADASTRA C compiler wrapper 'cc' was not found" >&2
    exit 1
fi
if ! cxx_compiler="$(command -v CC)"; then
    echo "ADASTRA C++ compiler wrapper 'CC' was not found" >&2
    exit 1
fi
export CC="$c_compiler"
export CXX="$cxx_compiler"

if [[ ! -f "$project_root/external/pybind11/CMakeLists.txt" ]]; then
    git -C "$project_root" submodule update --init --recursive
fi

mkdir -p "$(dirname "$requested_venv")"
venv_parent="$(cd "$(dirname "$requested_venv")" && pwd -P)"
venv_path="$venv_parent/$(basename "$requested_venv")"

if [[ ! -x "$venv_path/bin/python" ]]; then
    python -m venv "$venv_path"
fi

"$venv_path/bin/python" -m pip install -e "$project_root"
"$venv_path/bin/python" -c \
    'from cheby_tools import Field, SpectralDiscretization, _tecio'

echo "Installation ADASTRA terminée."
echo "Activez le venv avec :"
echo "source $venv_path/bin/activate"
