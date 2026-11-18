.PHONY: install test lint models data calibrate gate-record gate dashboard

install:
	pip install -e ".[dev,local]"

test:
	pytest

lint:
	ruff check src tests experiments scripts && ruff format --check src tests experiments scripts

models:          ## download the GGUF judges/generators (~9 GB) from Docker Hub's ai/ namespace
	./scripts/pull_models.sh

data:            ## RAGTruth raw files + the golden RAG-QA set
	python scripts/build_golden_rag_qa.py

calibrate:       ## experiment 1: judges vs human labels (CPU, a few hours; resumable via cassettes)
	python experiments/judge_calibration.py --judges $$(cat results/judge_calibration/JUDGES)
	python experiments/analyze_calibration.py

gate-record:     ## experiment 2: record baseline + candidate runs with local models
	./experiments/regression_gate.sh record

gate:            ## replay the gate exactly as CI does (seconds)
	./experiments/regression_gate.sh replay

dashboard:
	evallab report --out results/dashboard.html
