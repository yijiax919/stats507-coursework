.PHONY: data experiments

CONDA_ENV_PATH ?= d:/xxxl/.conda
PYTHON := conda run -p $(CONDA_ENV_PATH) python

data:
	$(PYTHON) scripts/download_dataset.py

experiments:
	$(PYTHON) scripts/run_experiments.py
