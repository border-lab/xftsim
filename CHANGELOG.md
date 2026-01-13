# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Test suite with pytest (`tests/` directory)
- Demo testing framework (`tests/test_demos.py`) covering UGRM and BGRM demos
- pytest configuration (`pytest.ini`)
- `claude.md` with project documentation, roadmap, and AI assistant instructions
- `CHANGELOG.md` for tracking changes

### Changed
- Updated `.gitignore` to exclude test virtual environments

## [0.2.0] - 2024

### Added
- `__version__` attribute in `xftsim/__init__.py`
- Run/install timing features
- Improved README for peer review

## [0.1.0] - Initial Release

### Added
- Core simulation framework
- Forward-time genetic simulation
- Multiple mating regime implementations (Random, Linear Assortative, K-Assortative, Batched)
- Phenogenetic architecture system
- Statistical estimators (GWAS, Haseman-Elston, heritability)
- I/O support for PLINK, VCF, Zarr formats
- CEU hg19 recombination map
- Comprehensive documentation with Jupyter notebook tutorials
