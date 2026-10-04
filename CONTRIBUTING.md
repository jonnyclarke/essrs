# Contributing
Thank you!
Contributions/enhancements/fixes are always welcome.

This file is included to provide a basic workflow to assist in the merge process.
Many of these checks are included as github actions so you will be unable to merge until all issues are resolved.

It is recommended to follow the ordering prescribed below as adding unit-tests almost invariably creates linting fixes and formatting issues which can then be automatically dealt with.

## Unit Testing

Please ensure all additional code is fully tested. This includes both unit,
```bash
cargo coverage-unit
```
and integration,
```bash
cargo coverage-integration
```
tests which replicate the testing procedures used within the CI/CD pipeline.

The above commands will create reports of the coverage within the repository which can be opened using
```bash
open tarpaulin-report.html
```
which provides a browser interface to check which parts of the code have been tested.

## Linting (linting: catch bugs / anti-patterns / non-idiomatic rust)
We use the standard rust linting process 'clippy'.
We additionally ensure that the check reports on all warnings as well.
```bash
cargo clippy-ci
```

## Code formatting
We again use the standard rust fmt program to enforce code style.,
```bash
cargo fmt-all
```
and also have a helper function to run the Continuous-Integration pipeline as an additional check.
```bash
cargo fmt-ci
```
Additionally, although this is not included in pre-merge checks, it is helpful to use the following to tidy up imports automatically.
```bash
cargo +nightly fmt-all
```

## Documentation
Please check the auto-documentation process works and everything is rendered correctly.
This can be done using:
```bash
cargo doc-math --open
```

## Git rebase
In order to keep the commit history as clean as possible the policy is to rebase and squash commits before merging into `main`.
The rebase can be done with:
```bash
git rebase main
```
which will likely then require you to deal with any merge conflicts.
The squash is used to clean up git commit history by merging commits. This allows any formatting changes to be merged with commit where logic was actually changed. It should not be used to compress all changes into a single commit in the case where each individual commit is its own meaningful logic change. The rebase can be done interactively using:
```bash
git rebase -i HEAD~<n_commits_to_review>
```
where `<n_commits_to_review>` is an integer value controlling how many commits back in the history are to be included in the process. The (i+1)^th commit is always squashed down onto its predecessor, the i^th commit.

## Building Python Extension
In order to build the python module that is implemented within this repo please use:
```bash
ur run maturin develop
```
to build in developer mode and:
```bash
uv run maturin develop --release
```
to build the optimised release package.