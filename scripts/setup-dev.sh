#!/usr/bin/env bash
set -euo pipefail

# ── Resolve repo root relative to this script ──────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="$REPO_ROOT/.venv"
LOCK_FILE="$REPO_ROOT/requirements-lock.txt"

# ── Colour helpers (disabled when stdout is not a tty) ─────────────────
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; RESET='\033[0m'
else
    GREEN=''; YELLOW=''; RED=''; RESET=''
fi

info()  { printf "${GREEN}[setup-dev]${RESET} %s\n" "$*"; }
warn()  { printf "${YELLOW}[setup-dev]${RESET} %s\n" "$*"; }
die()   { printf "${RED}[setup-dev]${RESET} %s\n" "$*" >&2; exit 1; }

# ── Find a suitable Python interpreter ─────────────────────────────────
find_python() {
    local candidates=("python3.12" "python3.11" "python3.10" "python3")
    for cmd in "${candidates[@]}"; do
        if command -v "$cmd" &>/dev/null; then
            echo "$cmd"
            return 0
        fi
    done
    return 1
}

validate_python() {
    local py="$1"
    command -v "$py" &>/dev/null || die "Python binary not found: $py"

    local ver
    ver="$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" \
        || die "Failed to query version from $py"

    local major minor
    major="${ver%%.*}"
    minor="${ver#*.}"

    if (( major < 3 )) || { (( major == 3 )) && (( minor < 10 )); }; then
        die "Python >= 3.10 is required (found $ver via $py)"
    fi

    info "Using $py (Python $ver)"
}

# ── Parse optional argument ────────────────────────────────────────────
if [[ $# -gt 1 ]]; then
    die "Usage: $0 [python-binary]"
fi

if [[ $# -eq 1 ]]; then
    PYTHON="$1"
else
    PYTHON="$(find_python)" || die "No suitable Python (>= 3.10) found on PATH"
fi

validate_python "$PYTHON"

# ── Remove existing venv ───────────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    warn "Removing existing virtualenv at $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

# ── Create venv ────────────────────────────────────────────────────────
info "Creating virtualenv at $VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"

# ── Activate (for the rest of this script) ─────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── Upgrade core packaging tools ──────────────────────────────────────
info "Upgrading pip, setuptools, wheel"
pip install --quiet --upgrade pip setuptools wheel

# ── Install project dependencies ──────────────────────────────────────
if [[ -f "$LOCK_FILE" ]]; then
    info "Installing from $LOCK_FILE"
    pip install --quiet -r "$LOCK_FILE"
    info "Installing xftsim in editable mode (no-deps)"
    pip install --quiet --no-deps -e "$REPO_ROOT"
else
    info "No lock file found; installing via pip install -e '.[all]'"
    pip install --quiet -e "$REPO_ROOT[all]"
    info "Generating $LOCK_FILE"
    pip freeze --exclude-editable > "$LOCK_FILE"
    info "Lock file written to $LOCK_FILE"
fi

# ── Done ───────────────────────────────────────────────────────────────
printf '\n'
info "Setup complete!"
printf '\n'
info "Activate the environment with:"
printf '\n'
printf '    source %s/bin/activate\n' "$VENV_DIR"
printf '\n'
