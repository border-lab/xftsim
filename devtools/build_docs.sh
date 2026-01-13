#!/bin/bash
# Build xftsim documentation
#
# Usage:
#   ./devtools/build_docs.sh          # Build docs
#   ./devtools/build_docs.sh clean    # Clean and rebuild
#   ./devtools/build_docs.sh serve    # Build and serve locally

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$REPO_ROOT/docs"

cd "$DOCS_DIR"

case "${1:-build}" in
    clean)
        echo "Cleaning build directory..."
        rm -rf _build
        echo "Building documentation..."
        make html
        ;;
    serve)
        echo "Building documentation..."
        make html
        echo ""
        echo "Starting local server at http://localhost:8000"
        echo "Press Ctrl+C to stop"
        cd _build/html && python3 -m http.server 8000
        ;;
    build|*)
        echo "Building documentation..."
        make html
        ;;
esac

echo ""
echo "Documentation built successfully!"
echo "Output: $DOCS_DIR/_build/html/index.html"
