# Makefile — Developer workflow commands for the Churn Pipeline project.
#
# CHANGED: Previously there was no Makefile. Any developer cloning the repo
# had to guess at setup steps. A Makefile communicates professionalism and
# makes CI/local workflows identical.
#
# Usage:
#   make install      Install the package and all dependencies
#   make test         Run the full test suite with coverage report
#   make lint         Check code style (flake8 + isort)
#   make format       Auto-format code with black and isort
#   make clean        Remove build artefacts and __pycache__
#   make download     Download the raw Telco Churn dataset locally

.PHONY: install test lint format clean download

PYTHON := python
PIP    := $(PYTHON) -m pip
SRC    := src/churn_pipeline
TESTS  := tests

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,ml]"

# ── Tests ────────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest $(TESTS) \
		--cov=$(SRC) \
		--cov-report=term-missing \
		--cov-report=xml:coverage.xml \
		-v

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	$(PYTHON) -m flake8 $(SRC) $(TESTS) --max-line-length=100 --ignore=E501,W503
	$(PYTHON) -m isort $(SRC) $(TESTS) --profile black --check-only --diff 

format:
	$(PYTHON) -m black $(SRC) $(TESTS) --line-length=100
	$(PYTHON) -m isort $(SRC) $(TESTS) --profile black

# ── Data ─────────────────────────────────────────────────────────────────────

download:
	$(PYTHON) -c "from churn_pipeline.ingest import download_raw_data; download_raw_data()"

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f coverage.xml .coverage
	@echo "Clean complete."
