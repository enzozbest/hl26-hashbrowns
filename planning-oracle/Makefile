.PHONY: install train evaluate serve test all lint \
       docker-build docker-run clean

PYTHON   := uv run python
PYTEST   := uv run pytest
UVICORN  := uv run uvicorn

# ── Defaults ────────────────────────────────────────────────────────────────

EPOCHS       ?= 50
BATCH_SIZE   ?= 64
LR           ?= 3e-5
CHECKPOINT   ?= checkpoints
PORT         ?= 8000
COUNCILS     ?=

# ── Core targets ────────────────────────────────────────────────────────────

install:
	uv sync

train:
ifndef COUNCILS
	$(error COUNCILS is required — e.g. make train COUNCILS="1 2 3")
endif
	$(PYTHON) -m training.train \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--lr $(LR) \
		--checkpoint-dir $(CHECKPOINT) \
		--councils $(COUNCILS)

evaluate:
	$(PYTHON) -m training.evaluate

serve:
	$(UVICORN) inference.api:app --host 0.0.0.0 --port $(PORT) --reload

test:
	$(PYTEST) tests/ -v

lint:
	uv run ruff check .

# ── Compound targets ───────────────────────────────────────────────────────

all: train evaluate

# ── Docker ──────────────────────────────────────────────────────────────────

docker-build:
	docker build -t planning-oracle .

docker-run:
	docker run --rm -p $(PORT):8000 --env-file .env planning-oracle

# ── Housekeeping ────────────────────────────────────────────────────────────

clean:
	rm -rf $(CHECKPOINT) outputs __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
