# Changelog

This file records notable changes to `ccrit-pystats` using the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

### Changed

- Documented how to install the latest GitHub release with pip.

## [0.1.3] - 2026-11-20

### Changed

- Applied British English to project text, including the **Standardised Residuals** plot label.

## [0.1.2] - 2026-11-20

### Changed

- Matched the four diagnostic panel titles and axis labels to ggResidpanel.

## [0.1.1] - 2026-11-20

### Fixed

- Prevented `dgplots()` from displaying the same figure twice in notebooks that use Matplotlib's inline backend.

## [0.1.0] - 2026-11-20

### Added

- Made `ccrit-pystats` installable with pip.
- Added `dgplots()` for fitted statsmodels OLS models, with residual, Q-Q, location-scale and Cook's D plots.
