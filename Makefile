VENV    := .venv
PYTHON  := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip

.PHONY: setup data run cliff plot all clean

# ── 1. Create venv, install dependencies, download dataset ───────────────────
setup: $(VENV)/bin/activate
	@echo ""
	@echo "=== Download LoCoMo dataset ==="
	@if [ ! -d "locomo" ]; then \
		git clone https://github.com/snap-research/locomo.git; \
	else \
		echo "locomo/ already exists, skipping clone."; \
	fi
	@echo ""
	@echo "=== Copy .env ==="
	@if [ ! -f ".env" ]; then \
		cp .env.example .env; \
		echo "Created .env — please fill in your API key before continuing."; \
	else \
		echo ".env already exists."; \
	fi
	@echo ""
	@echo "✓ Setup done. Next: edit .env, then run: make data"

$(VENV)/bin/activate:
	@echo "=== Creating virtual environment in $(VENV)/ ==="
	python3 -m venv $(VENV)
	@echo "=== Installing dependencies ==="
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Virtual environment ready."

# ── 2–5. Pipeline steps ──────────────────────────────────────────────────────
data:
	$(PYTHON) data/load_locomo.py

run:
	$(PYTHON) eval/run_all.py

cliff:
	$(PYTHON) eval/cliff_analysis.py

plot:
	$(PYTHON) analysis/plot_results.py

all: data run cliff plot

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	rm -rf results/*.json results/*.png __pycache__ */__pycache__ */*/__pycache__

clean-all: clean
	rm -rf $(VENV)
