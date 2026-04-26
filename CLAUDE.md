# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Pronoms is a Python library of normalization methods for quantitative proteomics data. The output is used in scientific publications and clinical-patient analyses, so **numerical accuracy is the primary requirement** — never trade correctness for convenience.

## Workflow rules (from the project owner)

- **TDD**: write a failing test first, then implement until it passes. Aim for 100% coverage with `pytest`, but every test must meaningfully exercise behavior — no fluff added solely to bump coverage numbers.
- **After every change** run the full test suite, the linter, and the type checker. Fix all errors and warnings (including deprecation warnings — don't silence them, resolve them).
- **No breaking changes** if avoidable. If a public-facing change is unavoidable, deprecate first (with a `DeprecationWarning`) rather than removing.
- **Drop-in compatibility**: the normalizer classes are meant to be interchangeable. Keep the `__init__` → `normalize(X)` → `plot_comparison(before, after)` shape consistent across new normalizers (DirectLFQ is the only legitimate exception today because it returns four arrays).
- **Vectorize** with NumPy/SciPy/pandas instead of Python loops where possible.
- **Always work inside `.venv`** at the project root. Never `pip install` into the host Python. Never modify anything outside the project directory without explicit permission.
- **Keep docs in sync** with code: when a change affects user-visible behavior, update `README.md`, the relevant file under `docs/`, and this `CLAUDE.md` in the same change.
- **Memory ↔ CLAUDE.md sync**: do not encode anything into Claude memory without also writing it into `CLAUDE.md` — the owner works across multiple machines.

## Common commands

All commands assume an activated `.venv` at the project root. The dev extras
(`pip install -e .[dev]`) pull in `pytest`, `pytest-cov`, `ruff`, `mypy`,
`pandas-stubs`, `types-seaborn`, and `jupyter`.

```bash
# Set up the virtual environment (one-time)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run the full test suite (warnings become errors -- see pytest.ini)
pytest

# Run a single test file / single test
pytest tests/test_median_normalizer.py
pytest tests/test_median_normalizer.py::test_normalize_matches_closed_form

# Verbose output, stop on first failure
pytest -vv -x

# Coverage
pytest --cov=src/pronoms --cov-report=term-missing

# Lint + format (ruff, configured in pyproject.toml under [tool.ruff])
ruff check src tests
ruff format src tests

# Static type check (mypy, configured under [tool.mypy] -- only checks src/)
mypy

# Build documentation locally
pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html
```

`pytest.ini` already sets `pythonpath = src`, `testpaths = tests`, and
`filterwarnings = error`. The third entry promotes any deprecation or runtime
warning to a test failure -- fix the underlying cause rather than suppressing.

The pre-flight gate before a commit is **all three of**: `pytest`, `ruff
check`, `mypy`. Treat any one going red as a blocker.

## R / VSN normalizer

`VSNNormalizer` calls Bioconductor's `vsn` package through `rpy2`. The R dependency is **lazy**: importing `pronoms` or any other normalizer does not require R or rpy2 to be functional. Only constructing or calling `VSNNormalizer` triggers `setup_r_environment(["vsn"])`.

To exercise VSN locally:

```bash
# Linux: install R + dev headers via the OS package manager (Debian/Ubuntu shown)
sudo apt install r-base r-base-dev libtirpc-dev   # (asks for sudo — confirm before running)

# Then in R:
Rscript -e 'if (!require("BiocManager", quietly=TRUE)) install.packages("BiocManager")'
Rscript -e 'BiocManager::install("vsn")'
```

CI installs R 4.3 + the `vsn` Bioconductor package and runs the full pytest suite against it (`.github/workflows/ci.yml`). VSN tests must continue to pass on CI.

## Architecture

### Layout (src layout, single top-level package `pronoms`)

```
src/pronoms/
  normalizers/   # one class per normalization method
  utils/         # validators, transformations, plotting, R interface
```

### The normalizer contract

Every normalizer follows the same lifecycle, and new ones should too:

1. `__init__(...)`: store hyperparameters, validate them eagerly (raise on bad values). Initialize all "fitted state" attributes to `None`.
2. `normalize(X: np.ndarray) -> np.ndarray`: the core method.
   - First call `validate_input_data(X)` from `utils.validators` (2D, non-empty, ndarray dtype).
   - Then `check_nan_inf(X)` and raise `ValueError` if there is anything non-finite.
   - Compute the transformation, populate the fitted-state attributes, and return an array of the **same shape** as `X` (DirectLFQ is the documented exception).
   - Inputs are `(n_samples, n_features)` — rows are samples, columns are proteins/features. This is the sklearn convention; do not flip it.
3. `plot_comparison(before, after, ...)`: delegate to `utils.plotting.create_hexbin_comparison`. Pass normalizer-specific labels and any axis hints (log scaling, autoscaling, identity vs. y=0 reference line) through that single helper rather than reinventing plotting code in each class.

`normalizers/__init__.py` re-exports every class and uses `__getattr__` to lazy-load each module on first access — this is what keeps `import pronoms.normalizers` cheap even when rpy2 / directlfq are heavy. Preserve that pattern when adding a new normalizer: add it to both `__all__` and `_lazy_imports`.

### Validation, plotting, and R helpers (`utils/`)

- `validators.validate_input_data` / `validators.check_nan_inf` are the canonical input gates — call them, do not duplicate the checks.
- `plotting.create_hexbin_comparison` is the single source of truth for "before vs. after" hexbin density plots. `plot_comparison_hexbin` is an alias kept for `VSNNormalizer`'s import; don't remove it without a deprecation cycle.
- `r_interface` is structured so that **`import pronoms.utils.r_interface` never imports `rpy2`**. The `_import_rpy2()` / `check_r_availability()` / `setup_r_environment()` chain defers all rpy2 work until an R-backed normalizer is actually used. `run_r_script()` writes the script to a tempfile, injects `input_data` into R's global env, runs `source()`, and pulls back any of `normalized_data` / `stats` / `parameters` / `diagnostics` that the script defines. Mirror that contract when adding new R-backed normalizers.

### Tests

Layout under `tests/`:

- `conftest.py` -- forces matplotlib to the headless `Agg` backend and closes
  every figure after each test (autouse fixture).
- `test_<normalizer>.py` -- one file per normalizer in `src/pronoms/normalizers`.
- `test_validators.py`, `test_transformations.py`, `test_plotting.py` --
  cover the helpers in `src/pronoms/utils`.
- `test_r_interface.py` -- pure-Python branches of the rpy2 wrapper (the live
  rpy2/R-bound tests skip themselves automatically when rpy2 cannot init R).
- `test_lazy_imports.py` -- `pronoms.normalizers.__getattr__` / `__dir__`.

Conventions for a new test file:

1. Cover the **closed-form numerical contract** (parametrize over inputs and
   compare to a hand-computed expected array with `assert_allclose`).
2. Cover the **invariant the docstring promises** (e.g. "post-normalization
   row medians equal the mean of medians" for `MedianNormalizer`) -- this
   catches semantic regressions that match-by-formula tests can miss.
3. Cover **input validation** (NaN, Inf, wrong dim, zero features) by
   parametrizing `pytest.raises` with `match=`.
4. Cover **state population** (the `scaling_factors` / `*_indices` /
   `vsn_params` attributes a normalizer sets after `normalize()`).
5. **Real-call plotting tests** preferred over mock-the-helper tests: they
   exercise the actual matplotlib plumbing too. Mock only when constructing
   real inputs would be more work than the test is worth (e.g. R-backed VSN).
6. **Do not write fluff tests** to chase coverage. Defensive guards that are
   structurally unreachable (e.g. shape sanity checks downstream of a
   validator that already enforced shape) are acceptable to leave uncovered.

For R/directlfq normalizers, include both a mocked unit test (driver
behavior) **and** a small real-end-to-end smoke test (catches regressions in
the actual third-party library).

### Documentation

- Per-normalizer Sphinx pages live under `docs/`, one `.rst` per class, all listed in `docs/index.rst`.
- They follow a fixed structure: Overview / Key Features / Algorithm Details / Parameters (`autoclass`) / Usage Example / When to Use / Considerations / See Also (and Citation where applicable). New normalizers should ship with a doc file in this same shape and be added to the toctree in `docs/index.rst`.
- `README.md` has a one-line summary per normalizer in the "Available Normalizers" list — keep it in sync when adding/removing.

## Release / publish

- `.github/workflows/ci.yml` runs on every push to `main` and on each published release.
- `.github/workflows/publish.yml` builds and publishes to PyPI on release. It derives the version from the GitHub tag (`vX.Y.Z` → `X.Y.Z`), rewrites `pyproject.toml`'s `version =` line, builds sdist + wheel, and publishes via PyPI trusted publishing (OIDC). Don't bump `pyproject.toml`'s version by hand for releases — tag the release and let the workflow update it.
