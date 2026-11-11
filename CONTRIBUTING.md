# Contributing

Use an issue to describe a reproducible problem or a proposed method change. Include a minimal synthetic example when possible. Keep datasets, credentials, checkpoints, generated predictions and account-specific submission records outside commits.

## Development environment

Use Python 3.12 and install the repository with `python -m pip install -e '.[dev]'`. Install the `model` extra to work on training or PIE integration and the `evaluation` extra to work on local scoring.

Before opening a pull request, run:

```bash
python -m pytest
ruff check .
ruff format --check .
python -m build
```

Keep source code, comments, docstrings and documentation in English. Prefer a small, reviewable change with a clear problem statement and validation evidence. Avoid machine-specific paths and hidden configuration. Configuration examples must work after replacing documented asset locations.

## Method changes

Preserve explicit gene and target axes. Do not silently change target seeds, control normalization, array precision, significance masks, or the definition of a reported metric. Count-generation changes must preserve the documented row/column constraints or clearly introduce a new recipe.

Use measured held-out responses only for the declared evaluation or adaptation stage. Record any panel reuse or routing change prompted by validation results. Report eligible target counts alongside aggregate metrics and distinguish local diagnostics from official scores.

By contributing, you agree that your changes may be distributed under the repository's CC BY-NC-SA 4.0 license. Retain attribution for third-party work.
