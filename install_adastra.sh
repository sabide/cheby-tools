#!/usr/bin/env bash

usage() {
    echo "Usage: $0 <venv-path>" >&2
}

resolve_executable() {
    local executable_name=$1
    local executable_path
    local executable_parent

    executable_path="$(type -P -- "$executable_name")" || return 1
    if [[ "$executable_path" != /* ]]; then
        executable_parent="$(cd "$(dirname "$executable_path")" && pwd -P)"
        executable_path="$executable_parent/$(basename "$executable_path")"
    fi
    [[ -x "$executable_path" ]] || return 1
    printf '%s\n' "$executable_path"
}

is_valid_venv() {
    local venv_path=$1
    local expected_prefix
    local reported_prefix

    [[ -d "$venv_path" ]] || return 1
    [[ -f "$venv_path/pyvenv.cfg" ]] || return 1
    [[ -x "$venv_path/bin/python" ]] || return 1
    [[ -f "$venv_path/bin/activate" ]] || return 1

    expected_prefix="$(cd "$venv_path" && pwd -P)"
    reported_prefix="$("$venv_path/bin/python" - "$expected_prefix" <<'PY'
from pathlib import Path
import sys

expected = Path(sys.argv[1]).resolve()
prefix = Path(sys.prefix).resolve()
if sys.prefix == sys.base_prefix or prefix != expected:
    raise SystemExit(1)
print(prefix)
PY
)" || return 1
    [[ "$reported_prefix" == "$expected_prefix" ]]
}

main() {
    set -euo pipefail

    if [[ $# -ne 1 ]]; then
        usage
        exit 2
    fi

    local requested_venv=$1
    local project_root
    local venv_parent
    local venv_path
    local c_compiler
    local cxx_compiler

    project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    mkdir -p "$(dirname "$requested_venv")"
    venv_parent="$(cd "$(dirname "$requested_venv")" && pwd -P)"
    venv_path="$venv_parent/$(basename "$requested_venv")"

    if [[ -e "$venv_path" ]] && ! is_valid_venv "$venv_path"; then
        echo "Refusing to overwrite $venv_path: not a valid Python virtual environment" >&2
        exit 1
    fi

    if ! command -v module >/dev/null 2>&1; then
        local module_init
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

    if ! c_compiler="$(resolve_executable cc)"; then
        echo "ADASTRA C compiler wrapper 'cc' was not found" >&2
        exit 1
    fi
    if ! cxx_compiler="$(resolve_executable CC)"; then
        echo "ADASTRA C++ compiler wrapper 'CC' was not found" >&2
        exit 1
    fi
    export CC="$c_compiler"
    export CXX="$cxx_compiler"

    if [[ ! -f "$project_root/external/pybind11/CMakeLists.txt" ]]; then
        git -C "$project_root" submodule update --init --recursive
    fi

    if [[ ! -e "$venv_path" ]]; then
        python -m venv "$venv_path"
    fi

    "$venv_path/bin/python" -m pip install -e "$project_root"
    "$venv_path/bin/python" -c \
        'from cheby_tools import Field, SpectralDiscretization, _tecio'

    echo "Installation ADASTRA terminée."
    echo "Activez le venv avec :"
    printf 'source %q\n' "$venv_path/bin/activate"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
