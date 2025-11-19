# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive visualization generation script (`generate_figures.py`)
- Five high-quality figures for documentation:
  - Missingness pattern heatmap
  - RMSE/MAE comparison charts
  - Distribution comparison plots
  - Scatter plots for imputed vs original values
  - Summary metrics table
- `matplotlib` and `seaborn` as core dependencies
- `figures/` directory with README documentation
- Enhanced CONTRIBUTING.md with detailed guidelines
- CHANGELOG.md for tracking version history
- Comprehensive README with:
  - Professional badges
  - Detailed table of contents
  - Categorized imputation methods
  - Multiple usage examples
  - Installation instructions
  - Development setup guide

### Changed
- Updated README from basic overview to comprehensive documentation
- Improved project structure with better organization

### Fixed
- None

## [0.1.0] - 2024-11-XX

### Added
- 16+ imputation methods:
  - **Statistical:** Mean, Median
  - **Time Series:** LOCF, NOCB
  - **Distance-Based:** KNN, Hot Deck
  - **Regression-Based:** Regression, Stochastic Regression, PMM, MICE
  - **Tree-Based:** MissForest
  - **Matrix Completion:** SoftImpute, Bayesian PCA
  - **Deep Learning:** Autoencoder, GAIN
  - **Advanced Statistical:** Gaussian Process
- Base imputer class with unified API
- Evaluation metrics: RMSE and MAE
- Comprehensive test suite with pytest
- CI/CD pipeline with GitHub Actions
- Code quality checks with flake8
- Type checking with mypy
- Demo Jupyter notebook with examples
- Example dataset in `data/` directory
- Code of Conduct
- Contributing guidelines
- MIT License

### Dependencies
- Python 3.10+
- pandas ^2.0
- numpy ^1.24
- scikit-learn ^1.4
- fancyimpute 0.7.0
- ppca ^0.0.4
- matplotlib ^3.7
- seaborn ^0.12

### Development Dependencies
- pytest ^7.0
- flake8 ^6.0
- mypy ^1.16.1
- coverage ^7.9.2
- nbconvert ^7.0
- jupyter ^1.0

## Version History

### Versioning Strategy

This project uses [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backwards compatible manner
- **PATCH** version for backwards compatible bug fixes

### Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md with release date
3. Create git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
4. Push tag: `git push origin v0.1.0`
5. Create GitHub release with notes from CHANGELOG

### Upcoming Features

Features planned for future releases:
- Additional imputation methods (EM algorithm, etc.)
- Performance benchmarks and comparisons
- Interactive visualization dashboard
- Plugin system for custom imputers
- Documentation website with Sphinx/MkDocs
- PyPI package publication
- Improved error handling and logging
- Parallel processing support for large datasets

---

[Unreleased]: https://github.com/DiogoRibeiro7/imputation-showcase/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/imputation-showcase/releases/tag/v0.1.0
