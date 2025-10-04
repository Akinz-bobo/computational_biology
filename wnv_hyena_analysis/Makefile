# WNV HyenaDNA Analysis Pipeline Makefile
# Publication-ready bioinformatics analysis

.PHONY: help install setup clean lint test run-analysis run-full docs

# Default target
help:
	@echo "WNV HyenaDNA Classification Pipeline"
	@echo "===================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install all dependencies"
	@echo "  setup        - Set up development environment"
	@echo "  clean        - Clean temporary files and cache"
	@echo "  lint         - Run code quality checks"
	@echo "  test         - Run test suite"
	@echo "  run-quick    - Run quick analysis (sample data)"
	@echo "  run-full     - Run full analysis (all 2068 sequences)"
	@echo "  docs         - Generate documentation"
	@echo "  reproducible - Run full reproducible analysis"
	@echo ""

# Environment setup
install:
	python3 -m pip install -r requirements.txt
	python3 -m pip install -e .


setup: install
	@echo "Setting up development environment..."
	mkdir -p cache logs
	python3 -c "import nltk; nltk.download('punkt')" || echo "NLTK optional"

# Cleanup
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf cache/* logs/* .pytest_cache/
	rm -rf build/ dist/

# Code quality
lint:
	@echo "Running code quality checks..."
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

# Testing
test:
	@echo "Running test suite..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Analysis pipelines
run-quick:
	@echo "Running quick analysis (sample of sequences)..."
	python3 scripts/run_analysis.py --config config/config.yaml --quick --n-samples 100

run-full:
	@echo "Running full analysis (all 2068 sequences)..."
	python3 scripts/run_analysis.py --config config/config.yaml --full

run-traditional:
	@echo "Running traditional bioinformatics analysis only..."
	python3 scripts/run_analysis.py --config config/config.yaml --traditional-only

run-hyenadna:
	@echo "Running HyenaDNA deep learning analysis only..."
	python3 scripts/run_analysis.py --config config/config.yaml --hyenadna-only

# Reproducible analysis
reproducible:
	@echo "Running fully reproducible analysis..."
	python3 scripts/run_analysis.py --config config/config.yaml --reproducible --full
	python3 scripts/generate_report.py --output results/reports/

# Documentation
docs:
	@echo "Generating documentation..."
	cd docs && make html

# Data processing
process-data:
	@echo "Processing raw FASTA data..."
	python3 src/analysis/data_processor.py --input data/raw/west_nile_genomes.fasta --output data/processed/

# Model training
train-models:
	@echo "Training all models..."
	python3 src/models/train.py --config config/config.yaml

# Visualization
generate-plots:
	@echo "Generating all publication figures..."
	python3 src/visualization/plot_generator.py --config config/config.yaml

# Pipeline status
status:
	@echo "Pipeline Status:"
	@echo "==============="
	@ls -la data/raw/ | grep -E "\.(fasta|fa)$$" | wc -l | xargs echo "Raw FASTA files:"
	@ls -la data/processed/ | grep -E "\.(csv|h5|pkl)$$" | wc -l | xargs echo "Processed data files:"
	@ls -la results/models/ | grep -E "\.(pkl|joblib|pt)$$" | wc -l | xargs echo "Trained models:"
	@ls -la results/figures/ | grep -E "\.(png|pdf|svg)$$" | wc -l | xargs echo "Generated figures:"