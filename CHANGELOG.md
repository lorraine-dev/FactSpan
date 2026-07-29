# Changelog

All notable changes to the FactSpan dataset and tooling are documented here.
This project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.1.0] - 2025-04-14
### Added
- New claims pulled from the Data Commons ClaimReview feed, extending
  coverage through 2025 (last claim date 2025-04-11): +105 rows in
  `FactSpan.csv`, +344 rows in `FactSpan_annotated.csv`.
### Fixed
- Two invalid rows removed in the lead-up to this release.

> **Zenodo archival note:** the GitHub Release for this version was
> initially published before the [`zenodo-release.yml`](./.github/workflows/zenodo-release.yml)
> workflow existed, so Zenodo's default (whole-repo) integration archived
> it as an unrelated concept DOI, [10.5281/zenodo.21670252](https://doi.org/10.5281/zenodo.21670252).
> That record is not part of this dataset's citable history. This version
> was subsequently backfilled via a manual `workflow_dispatch` run of
> `zenodo-release.yml` and is properly archived as
> [10.5281/zenodo.21671356](https://doi.org/10.5281/zenodo.21671356), a new
> version of concept [10.5281/zenodo.15084387](https://doi.org/10.5281/zenodo.15084387).

## [1.0.0] - 2025-03-25
### Added
- Initial public release of the FactSpan dataset (`FactSpan.csv` and
  `FactSpan_annotated.csv`), migrated from a private repository.
  Archived to Zenodo as DOI 10.5281/zenodo.15084388.
### Changed
- Removed the `media_associated` column from the annotated dataset.
### Fixed
- Removed one claim with an incorrect date.

> **Note on retroactive versions:** v1.0.0 and v1.1.0 were assigned
> retroactively in July 2026 when semantic versioning and Zenodo release
> automation were introduced for this repository. v1.0.0 corresponds to
> commit `0a6f5dd`, confirmed via byte-for-byte checksum match against the
> files hosted on the existing Zenodo record. v1.1.0 corresponds to commit
> `e1dc9fe`, the last commit that modified the dataset files before this
> versioning scheme was formalized.
