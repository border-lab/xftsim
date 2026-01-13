#!/bin/bash
# Build xftsim documentation
#
# Usage:
#   ./devtools/build_docs.sh          # Build docs
#   ./devtools/build_docs.sh clean    # Clean and rebuild
#   ./devtools/build_docs.sh serve    # Build and serve locally
#
# Note: Requires xftsim-test environment for API autodoc to work.
# The script automatically uses xftsim-test if available.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$REPO_ROOT/docs"

# Use xftsim-test environment's sphinx-build if available (required for API docs)
XFTSIM_TEST_SPHINX="/home/rsb/micromamba/envs/xftsim-test/bin/sphinx-build"
if [[ -x "$XFTSIM_TEST_SPHINX" ]]; then
    SPHINX_BUILD="$XFTSIM_TEST_SPHINX"
    echo "Using xftsim-test environment for docs build"
else
    SPHINX_BUILD="sphinx-build"
    echo "Warning: xftsim-test environment not found, API docs may be incomplete"
fi

cd "$DOCS_DIR"

build_docs() {
    "$SPHINX_BUILD" -M html . _build
}

case "${1:-build}" in
    clean)
        echo "Cleaning build directory..."
        rm -rf _build
        echo "Building documentation..."
        build_docs
        ;;
    serve)
        echo "Building documentation..."
        build_docs
        echo ""
        echo "Starting local server at http://localhost:8000"
        echo "Press Ctrl+C to stop"
        cd _build/html && python3 -m http.server 8000
        ;;
    build|*)
        echo "Building documentation..."
        build_docs
        ;;
esac

echo ""
echo "Documentation built successfully!"
echo "Output: $DOCS_DIR/_build/html/index.html"
