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