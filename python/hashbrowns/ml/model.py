import asyncio

import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from hashbrowns.config import settings
from hashbrowns.ibex.client import IbexClient
from hashbrowns.ml.encoder import TARGET_NAMES, encode_batch


async def query_db():
    async with IbexClient(settings) as client:
        return await client.search([547136, 184084], 500, 27700)


responses = asyncio.run(query_db())
print(f"Fetched {len(responses)} planning applications")

X, y_raw = encode_batch(responses)
print(f"Feature matrix: {X.shape}, Target vector: {y_raw.shape}")

# Remap sparse target labels to contiguous 0..n_classes-1 for XGBoost
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = [TARGET_NAMES[i] for i in le.classes_]
print(f"Classes in dataset: {class_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0
)

clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = sklearn.metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}")
