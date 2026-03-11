.PHONY: install run test train lint clean

# ── Setup ──────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

# ── Run app ────────────────────────────────────────────────────────────────
run:
	streamlit run src/ui/app.py

# ── Test ───────────────────────────────────────────────────────────────────
test:
	cd src && python -m pytest ../tests/ -v

test-cov:
	cd src && python -m pytest ../tests/ -v --cov=. --cov-report=term-missing --cov-report=html

# ── Train CUAD model ───────────────────────────────────────────────────────
train:
	python src/classification/train_cuad.py \
		--output_dir models/cuad_finetuned \
		--epochs 3 \
		--batch_size 8 \
		--fp16

train-fast:
	python src/classification/train_cuad.py \
		--output_dir models/cuad_finetuned_fast \
		--epochs 1 \
		--batch_size 16 \
		--fp16

# ── Lint ───────────────────────────────────────────────────────────────────
lint:
	python -m flake8 src/ --max-line-length=100 --ignore=E501,W503
	python -m mypy src/ --ignore-missing-imports

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
