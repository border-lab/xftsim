#!/bin/bash
# Install git hooks for xftsim development
#
# Usage: ./scripts/install_hooks.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "Installing git hooks..."

# Create pre-commit hook
cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash
# Pre-commit hook to automatically bump dev version
#
# This hook increments the dev version number (e.g., 0.2.0.dev1 -> 0.2.0.dev2)
# on every commit to track changes during development.
#
# To skip version bump for a commit, use: git commit --no-verify

set -e

SCRIPT_DIR="$(git rev-parse --show-toplevel)/scripts"
VERSION_FILE="$(git rev-parse --show-toplevel)/xftsim/__init__.py"

# Check if bump_version.py exists
if [ ! -f "$SCRIPT_DIR/bump_version.py" ]; then
    echo "Warning: bump_version.py not found, skipping version bump"
    exit 0
fi

# Check if __init__.py is being modified in this commit
# If so, don't auto-bump (user might be doing a manual version change)
if git diff --cached --name-only | grep -q "xftsim/__init__.py"; then
    echo "Note: __init__.py already staged, skipping auto version bump"
    exit 0
fi

# Bump the dev version
echo "Auto-bumping dev version..."
python3 "$SCRIPT_DIR/bump_version.py" dev

# Stage the version file change
git add "$VERSION_FILE"

echo "Dev version bumped and staged."
EOF

chmod +x "$HOOKS_DIR/pre-commit"

echo "Git hooks installed successfully!"
echo ""
echo "The pre-commit hook will automatically bump the dev version on each commit."
echo "To skip version bump for a specific commit, use: git commit --no-verify"
