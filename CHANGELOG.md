# Changelog

All notable changes to xftsim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For development workflow changes (testing, CI/CD, tooling), see [devtools/CHANGELOG.dev.md](devtools/CHANGELOG.dev.md).

## [Unreleased]

## [0.3.0] - 2026-01-13

### Fixed
- API documentation now builds correctly on ReadTheDocs and locally

### Changed
- Documentation build script now uses xftsim-test environment for proper API autodoc

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
