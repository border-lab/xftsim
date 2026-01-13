# Development Workflow Changelog

Changes to development tooling, CI/CD, testing infrastructure, and documentation systems.

## [Unreleased]

### Added
- Test suite with pytest (`tests/` directory)
- Demo testing framework (`tests/test_demos.py`) covering UGRM and BGRM demos
- pytest configuration (`pytest.ini`)
- `claude.md` with project documentation, roadmap, and AI assistant instructions
- Version management system (`devtools/bump_version.py`)
  - Pre-commit hook auto-increments dev version on each commit
  - Manual script for release version bumps (patch/minor/major)
- `devtools/install_hooks.sh` for setting up git hooks
- Split changelogs: `CHANGELOG.md` (software) and `devtools/CHANGELOG.dev.md` (workflow)

### Changed
- Updated `.gitignore` to exclude test virtual environments
- Reorganized development tools into `devtools/` directory
- Symlinked `claude.md` from `devtools/` to project root
