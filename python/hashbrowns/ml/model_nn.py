import asyncio

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from hashbrowns.config import settings
from hashbrowns.ibex.client import IbexClient
from hashbrowns.ml.encoder import TARGET_NAMES, encode_batch

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


async def query_db():
    async with IbexClient(settings) as client:
        return await client.applications(
            date_from="2025-01-01", date_to="2025-12-31", council_ids=[366]
        )


responses = asyncio.run(query_db())
print(f"Fetched {len(responses)} planning applications")

X, y_raw = encode_batch(responses)
print(f"Feature matrix: {X.shape}, Target vector: {y_raw.shape}")

le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = len(le.classes_)
class_names = [TARGET_NAMES[i] for i in le.classes_]
print(f"Classes ({n_classes}): {class_names}")

# Stratified split to preserve class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardise features — important for NN convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Compute class weights to handle imbalance
weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(weights, dtype=torch.float32)
print(f"Class weights: {dict(zip(class_names, weights.round(3)))}")

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

n_features = X_train.shape[1]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PlanningNet(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


model = PlanningNet(n_features, n_classes)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode="min", patience=10, factor=0.5
)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

EPOCHS = 300
BATCH_SIZE = 32
PATIENCE = 25  # early stopping patience

dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

best_val_loss = float("inf")
best_state = None
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in loader:
        optimiser.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item() * len(y_batch)
    epoch_loss /= len(y_train_t)

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_test_t)
        val_loss = criterion(val_logits, y_test_t).item()
        preds = val_logits.argmax(dim=1)
        acc = (preds == y_test_t).float().mean().item()

    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict().copy()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if (epoch + 1) % 25 == 0:
        lr = optimiser.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS}  "
            f"loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={acc:.3f}  lr={lr:.1e}"
        )

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping at epoch {epoch + 1} (no improvement for {PATIENCE} epochs)")
        break

# Restore best model
if best_state is not None:
    model.load_state_dict(best_state)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

model.eval()
with torch.no_grad():
    y_pred = model(X_test_t).argmax(dim=1).numpy()

acc = sklearn.metrics.accuracy_score(y_test, y_pred)
print(f"\nFinal accuracy: {acc:.3f}")
print(
    sklearn.metrics.classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )
)