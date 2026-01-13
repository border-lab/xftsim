#!/usr/bin/env python3
"""
Version management script for xftsim.

Usage:
    python scripts/bump_version.py dev      # Increment dev version (0.2.0 -> 0.2.0.dev1)
    python scripts/bump_version.py patch    # Bump patch version (0.2.0.dev3 -> 0.2.1)
    python scripts/bump_version.py minor    # Bump minor version (0.2.1 -> 0.3.0)
    python scripts/bump_version.py major    # Bump major version (0.3.0 -> 1.0.0)
    python scripts/bump_version.py show     # Show current version
"""
import re
import sys
from pathlib import Path

VERSION_FILE = Path(__file__).parent.parent / "xftsim" / "__init__.py"
VERSION_PATTERN = r'__version__\s*=\s*["\']([^"\']+)["\']'


def get_current_version() -> str:
    """Read current version from __init__.py."""
    content = VERSION_FILE.read_text()
    match = re.search(VERSION_PATTERN, content)
    if not match:
        raise ValueError(f"Could not find __version__ in {VERSION_FILE}")
    return match.group(1)


def set_version(new_version: str) -> None:
    """Write new version to __init__.py."""
    content = VERSION_FILE.read_text()
    new_content = re.sub(
        VERSION_PATTERN,
        f'__version__="{new_version}"',
        content
    )
    VERSION_FILE.write_text(new_content)
    print(f"Version updated: {get_current_version()} -> {new_version}")


def parse_version(version: str) -> tuple:
    """
    Parse version string into components.

    Returns: (major, minor, patch, dev_num or None)
    Examples:
        "0.2.0" -> (0, 2, 0, None)
        "0.2.0.dev3" -> (0, 2, 0, 3)
    """
    # Match patterns like "0.2.0" or "0.2.0.dev3"
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)(?:\.dev(\d+))?$', version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")

    major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
    dev_num = int(match.group(4)) if match.group(4) else None

    return major, minor, patch, dev_num


def bump_dev(version: str) -> str:
    """Increment dev version: 0.2.0 -> 0.2.0.dev1, 0.2.0.dev1 -> 0.2.0.dev2"""
    major, minor, patch, dev_num = parse_version(version)

    if dev_num is None:
        new_dev = 1
    else:
        new_dev = dev_num + 1

    return f"{major}.{minor}.{patch}.dev{new_dev}"


def bump_patch(version: str) -> str:
    """Bump patch version: 0.2.0.dev3 -> 0.2.1, 0.2.1 -> 0.2.2"""
    major, minor, patch, _ = parse_version(version)
    return f"{major}.{minor}.{patch + 1}"


def bump_minor(version: str) -> str:
    """Bump minor version: 0.2.1 -> 0.3.0"""
    major, minor, _, _ = parse_version(version)
    return f"{major}.{minor + 1}.0"


def bump_major(version: str) -> str:
    """Bump major version: 0.3.0 -> 1.0.0"""
    major, _, _, _ = parse_version(version)
    return f"{major + 1}.0.0"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()
    current = get_current_version()

    if command == "show":
        print(current)
        return

    if command == "dev":
        new_version = bump_dev(current)
    elif command == "patch":
        new_version = bump_patch(current)
    elif command == "minor":
        new_version = bump_minor(current)
    elif command == "major":
        new_version = bump_major(current)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)

    set_version(new_version)


if __name__ == "__main__":
    main()
