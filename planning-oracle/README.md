# Planning Oracle

An ML pipeline that predicts UK planning application approval probability using a multi-branch neural network trained on historical council decision data.

## Problem Statement

Planning applications in England go through local planning authorities (councils), each with different approval tendencies, processing speeds, and policy priorities. Developers and applicants often have little visibility into how likely their application is to succeed before submitting.

Planning Oracle fetches historical decision data from the Planning API, engineers features from application text, metadata, and council statistics, then trains a neural network to estimate approval probability with calibrated confidence intervals.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         Inference API                        │
│                     (FastAPI / uvicorn)                       │
│                                                              │
│  POST /predict ─► Parser ─► Ranker ─► Model ─► Calibrator   │
│                                                              │
└──────────────────────┬───────────────────────────────────────┘
                       │ loads artefacts from
┌──────────────────────▼───────────────────────────────────────┐
│                     Training Pipeline                        │
│                                                              │
│  Fetch ─► Dataset ─► Train ─► Calibrate ─► Evaluate          │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌────────────┐ │
│  │ API      │  │ Text     │  │ Application│  │ Council    │ │
│  │ Client   │  │ Embedder │  │ Features   │  │ Features   │ │
│  └──────────┘  └──────────┘  └────────────┘  └────────────┘ │
└──────────────────────────────────────────────────────────────┘

Model Architecture (ApprovalModel):
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  Text Branch   │  │   App Branch   │  │ Council Branch │
│  384→256→128   │  │   N→128→128    │  │   N→128→128    │
│  (LayerNorm,   │  │  (BatchNorm,   │  │  (BatchNorm,   │
│   Dropout)     │  │   Dropout)     │  │   Dropout)     │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                   │                   │
        └───────────┬───────┘───────────────────┘
                    ▼
            ┌──────────────┐
            │ Fusion Head  │
            │ 384→256→64→1 │
            │ (LayerNorm,  │
            │  Dropout)    │
            └──────┬───────┘
                   ▼
             P(approval)
```

## Project Structure

```
planning-oracle/
├── config/          # Pydantic-settings configuration
├── data/            # API client and schema definitions
├── features/        # Feature extraction (text, application, council)
├── model/           # Neural network, calibration, council ranker, attribution
├── training/        # Dataset, training loop, evaluation
├── inference/       # NLU parser, pipeline orchestrator, FastAPI endpoint
├── scripts/         # End-to-end pipeline runner
└── tests/           # Pytest suite
```

## Quick Start

### 1. Install

```bash
# Using UV
uv sync

# Or with Make
make install
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your API credentials:

```env
PLANNING_API_BASE_URL=https://api.example.com/v1
PLANNING_API_AUTH_TOKEN=your-token-here
```

### 3. Run the Full Pipeline

```bash
# End-to-end: train → calibrate → evaluate
make all

# Or use the pipeline script with more control
uv run python -m scripts.run_pipeline --councils council-01 council-02 --epochs 30 --serve
```

### 4. Start the Server

```bash
make serve
# or
uv run uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

### Predict Approval Probability

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"proposal_text": "Construction of 12 new residential dwellings with associated parking and landscaping in Westminster"}'
```

```json
{
  "result": {
    "parsed_proposal": {
      "raw_text": "Construction of 12 new residential dwellings ...",
      "num_houses": 12,
      "project_type": "medium_scale",
      "region": "london",
      "specific_council": "Westminster"
    },
    "approval_probability": 0.73,
    "confidence_interval": [0.64, 0.81],
    "top_councils": [
      {"council_id": "westminster", "council_name": "Westminster City Council", "score": 0.82}
    ],
    "feature_attributions": {}
  }
}
```

## Makefile Targets

| Target           | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `make install`   | Install dependencies via UV                          |
| `make train`     | Train the approval model                             |
| `make evaluate`  | Evaluate model and generate plots                    |
| `make serve`     | Start the FastAPI server (port 8000)                 |
| `make test`      | Run the test suite                                   |
| `make all`       | Full pipeline: train → evaluate                      |
| `make docker-build` | Build the Docker image                            |
| `make docker-run`   | Run the container (reads `.env`)                  |
| `make clean`     | Remove checkpoints, outputs, and caches              |

Override training hyperparameters:

```bash
make train EPOCHS=100 BATCH_SIZE=128 LR=0.0005
```

## Model Architecture

**ApprovalModel** is a multi-branch feed-forward network:

- **Text Branch**: Processes sentence-transformer embeddings (384-dim) through two linear layers with LayerNorm and dropout.
- **Application Branch**: Processes numerical/categorical application features (log-transformed counts, unit mix ratios, one-hot encoded types) through BatchNorm and two layers.
- **Council Branch**: Processes council statistics (approval rates, decision speed, activity level) through BatchNorm and two layers.
- **Fusion Head**: Concatenates the three 128-dim branch outputs and passes through a final MLP to produce a single logit.

**Training details**:
- **Loss**: Focal loss (`gamma=2.0, alpha=0.7`) to handle class imbalance
- **Optimiser**: AdamW with weight decay `1e-4`
- **Scheduler**: CosineAnnealingWarmRestarts (`T_0=10, T_mult=2`)
- **Early stopping**: Patience of 10 epochs on validation AUROC
- **Calibration**: Post-hoc temperature scaling via LBFGS on validation set

## Features

### Application Features

| Feature              | Description                                        |
| -------------------- | -------------------------------------------------- |
| `log_num_houses`     | Log-transformed number of proposed houses          |
| `log_floor_area`     | Log-transformed gross floor area (sqm)             |
| `unit_*_ratio`       | Proportion of 1-bed, 2-bed, 3-bed, 4+-bed units   |
| `affordable_ratio`   | Proportion of affordable housing units             |
| `app_type_*`         | One-hot encoded application type                   |
| `project_type_*`     | One-hot encoded project type                       |

### Council Features

| Feature                | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `approval_rate`        | Historical approval rate (0-1)                   |
| `avg_decision_days`    | Average days to decision (by project type)       |
| `activity_level_*`     | One-hot encoded activity level (high/medium/low) |
| `total_applications`   | Total applications in the stats period           |
| `homes_approved`       | Net new homes approved                           |

### Text Features

384-dimensional embeddings from `all-MiniLM-L6-v2` sentence-transformer, with memmap caching for fast repeated access.

## Evaluation Metrics

After training, `make evaluate` produces four plots in `outputs/` and prints:

| Metric               | Description                                           |
| --------------------- | ----------------------------------------------------- |
| AUROC                 | Area under ROC curve (discrimination)                 |
| Average Precision     | Area under precision-recall curve                     |
| F1 Score              | Harmonic mean of precision and recall (optimal threshold) |
| ECE (raw)             | Expected calibration error before temperature scaling |
| ECE (calibrated)      | Expected calibration error after temperature scaling  |
| Accuracy              | Classification accuracy at optimal threshold          |

**Generated plots**: reliability diagram, ROC curve, precision-recall curve, score distributions.

## Configuration Reference

All settings are loaded from environment variables or a `.env` file.

| Variable                      | Default                    | Description                          |
| ----------------------------- | -------------------------- | ------------------------------------ |
| `PLANNING_API_BASE_URL`       | `https://api.example.com/v1` | Planning API root URL              |
| `PLANNING_API_AUTH_TOKEN`     | (empty)                    | Bearer token for API auth            |
| `LEARNING_RATE`               | `0.001`                    | AdamW learning rate                  |
| `BATCH_SIZE`                  | `64`                       | Training mini-batch size             |
| `EPOCHS`                      | `50`                       | Maximum training epochs              |
| `EMBEDDING_DIM`               | `384`                      | Text embedding dimensionality        |
| `MAX_CONCURRENT_REQUESTS`     | `10`                       | API rate limit (concurrent requests) |
| `TEXT_ENCODER_MODEL`          | `all-MiniLM-L6-v2`        | HuggingFace sentence-transformer ID |
| `CHECKPOINT_DIR`              | `checkpoints`              | Directory for model checkpoints      |
| `OUTPUT_DIR`                  | `outputs`                  | Directory for evaluation artefacts   |

## Docker

```bash
# Build
docker build -t planning-oracle .

# Run (reads .env for config)
docker run --rm -p 8000:8000 --env-file .env planning-oracle

# Or via Make
make docker-build
make docker-run
```

## Testing

```bash
make test
# or
uv run pytest tests/ -v
```

## Known Limitations

- **SHAP attribution is a skeleton**: `model/attribution.py` is not yet implemented.
- **Single-model architecture**: No ensemble or model selection — the pipeline trains one `ApprovalModel`.
- **UK-only**: Region detection and council matching are hardcoded for English local planning authorities.
- **No GPU multi-device support**: Training runs on a single device (CPU or one GPU).

## Future Work

- Add SHAP-based feature attribution for explainability
- Experiment with transformer-based text encoders (e.g. `all-mpnet-base-v2`)
- Add segment-level evaluation dashboards
- CI/CD pipeline with automated retraining on new data
