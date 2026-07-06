.PHONY: install install-dev install-advanced lint test app api v1 v2 v3 v0 report docker clean

install:            ## Install core dependencies
	pip install -r requirements.txt

install-advanced:   ## Install v2/v3 advanced-analytics dependencies
	pip install -r requirements-v2.txt

install-dev:        ## Install dev tooling (ruff, pytest)
	pip install ruff pytest

lint:               ## Lint source with ruff
	ruff check src

test:               ## Run smoke tests
	pytest

app:                ## Run the dashboard (default v1)
	python scripts/run_app.py --version v1

api:                ## Run the FastAPI prediction service
	uvicorn api.main:app --reload

v1:                 ## Build v1 (real-data) pipeline
	python scripts/run_v1_pipeline.py

v2:                 ## Build v2 (synthetic advanced analytics) pipeline
	python scripts/run_v2_pipeline.py

v3:                 ## Build v3 (risk estimator features) pipeline
	python scripts/run_v3_pipeline.py

v0:                 ## Build v0 static visual assets
	MHP_VERSION=v0 python src/v0_visuals.py

report:             ## Build the LaTeX report PDF
	cd report_latex && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

docker:             ## Build and run the dashboard in Docker
	docker build -t mental-health-viz . && docker run -p 8501:8501 mental-health-viz

clean:              ## Remove Python/build caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .pytest_cache
