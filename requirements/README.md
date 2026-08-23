# Lockfiles

Exact, hash-verified pins for this repository's **own** CI and release pipeline, required by
[Packaging and Release Standards §4](../docs/standards/packaging-and-release-standards.md) and
[Security Standards §11](../docs/standards/security-standards.md).

| File | Contents | Used by |
|---|---|---|
| `ci.lock` | Runtime dependencies plus the `dev` extra: the whole test, lint, type and boundary toolchain | Every blocking CI job |
| `release.in` / `release.lock` | The build and publish chain (`build`, `hatchling`, `twine`) | `release.yml`, and CI's `build` job |

## What these are not

They do **not** define what a consumer installs. `pip install sweatmeter` resolves the compatible
ranges in `pyproject.toml`; a library that shipped pinned runtime dependencies would be
un-coinstallable with the rest of the suite. These files exist so that a green build stays green:
without them every CI run re-resolves, and a new `ruff` or `mypy` release can change the result
with no commit to explain it — and `pip-audit` would be auditing today's resolution rather than
what the build actually used.

## Regenerating

Run after any change to `pyproject.toml`'s dependencies or `dev` extra, and commit the result:

```bash
pip install pip-tools
pip-compile --strip-extras --extra dev --generate-hashes \
    --output-file requirements/ci.lock pyproject.toml
pip-compile --strip-extras --generate-hashes \
    --output-file requirements/release.lock requirements/release.in
```

`uv pip compile` is the sanctioned alternative (Security Standards §11).

## Interpreter

Resolved on Python 3.13. Every pin's `requires-python` admits 3.12, and no pin is
CPython-ABI-specific, so the same lock installs on both supported versions; the 3.14 early-warning
job deliberately resolves from ranges instead, because pinning a version that has no 3.14 wheels
would defeat the purpose of an early warning.
