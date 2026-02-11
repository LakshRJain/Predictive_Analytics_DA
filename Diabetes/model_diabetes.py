import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, brier_score_loss
)
from xgboost.callback import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
import joblib

# =====================================================
# 1. Load and preprocess data
# =====================================================
df = pd.read_csv('patient_180days_readings.csv')

# Time-series features (per day)
ts_features = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk'
]

# Static features (per patient)
static_features = ['Sex', 'Age', 'Education', 'Income']

# Encode categorical: make Sex numeric
df['Sex'] = (df['Sex'].astype(str).str.lower() == 'male').astype(int)

# Group by patient_id
patients, labels, static = [], [], []
for pid, group in df.groupby('patient_id'):
    group = group.sort_values('day')
    patients.append(group[ts_features].values)           # shape (180, 17)
    static.append(group[static_features].iloc[0].values) # take static once
    labels.append(group['deplict'].iloc[0])              # same label for all rows

X_seq = np.stack(patients)   # (N, 180, 17)
X_static = np.stack(static)  # (N, 4)
y = np.array(labels)

# Train/val/test split
X_seq_train, X_seq_test, X_static_train, X_static_test, y_train, y_test = train_test_split(
    X_seq, X_static, y, test_size=0.2, random_state=42, stratify=y
)
X_seq_train, X_seq_val, X_static_train, X_static_val, y_train, y_val = train_test_split(
    X_seq_train, X_static_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# =====================================================
# 2. Transformer Model
# =====================================================
class PatientDataset(Dataset):
    def __init__(self, X_seq, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X_seq[idx], self.y[idx]

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, 1)  # <--- added final classifier head
    def forward(self, x):
        x = self.input_proj(x)       # (B, seq, d_model)
        x = self.transformer(x)      # (B, seq, d_model)
        x = x.transpose(1, 2)        # (B, d_model, seq)
        x = self.pool(x).squeeze(-1) # (B, d_model)
        x = self.fc_out(x)           # (B, 1)
        return x.squeeze(-1)         # (B,)

# =====================================================
# 3. Train Transformer
# =====================================================
def train_transformer(X_seq_train, y_train, X_seq_val, y_val, input_dim, epochs=15, batch_size=64):
    model = TransformerEncoder(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(PatientDataset(X_seq_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PatientDataset(X_seq_val, y_val), batch_size=batch_size)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                out = model(Xb)
                val_loss += criterion(out, yb).item() * len(yb)
        
        train_loss /= len(y_train)
        val_loss /= len(y_val)
        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'transformer_encoder.pt')

    # Load best
    model.load_state_dict(torch.load('transformer_encoder.pt'))
    return model

# Train transformer for 15 epochs
transformer = train_transformer(
    X_seq_train, y_train, X_seq_val, y_val, input_dim=len(ts_features), epochs=15
)

# =====================================================
# 4. Get sequence embeddings
# =====================================================
def get_embeddings(model, X_seq):
    model.eval()
    embs = []
    with torch.no_grad():
        X_seq_torch = torch.tensor(X_seq, dtype=torch.float32)
        for i in range(0, len(X_seq), 256):
            batch = X_seq_torch[i:i+256]
            x = model.input_proj(batch)
            x = model.transformer(x)
            x = x.transpose(1,2)
            x = model.pool(x).squeeze(-1)
            embs.append(x.numpy())
    return np.vstack(embs)

emb_train = get_embeddings(transformer, X_seq_train)
emb_val = get_embeddings(transformer, X_seq_val)
emb_test = get_embeddings(transformer, X_seq_test)

# =====================================================
# 5. XGBoost on [embeddings + static]
# =====================================================
Xgb_train = np.hstack([emb_train, X_static_train])
Xgb_val = np.hstack([emb_val, X_static_val])
Xgb_test = np.hstack([emb_test, X_static_test])

# Simple XGBoost for binary classification
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",  # Changed to logloss for binary classification
    random_state=42
)

# Use the correct training data (Xgb_train and y_train)
xgb_model.fit(Xgb_train, y_train)

# =====================================================
# 6. Evaluation
# =====================================================
y_pred_proba = xgb_model.predict_proba(Xgb_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

auroc = roc_auc_score(y_test, y_pred_proba)
auprc = average_precision_score(y_test, y_pred_proba)
print(f"AUROC: {auroc:.3f}  AUPRC: {auprc:.3f}")

# Calibration plot
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('Predicted probability')
plt.ylabel('True probability')
plt.title('Calibration Curve')
plt.show()

# =====================================================
# 7. SHAP explainability
# =====================================================
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(Xgb_test)
shap.summary_plot(
    shap_values, Xgb_test,
    feature_names=[f"emb_{i}" for i in range(emb_test.shape[1])] + static_features
)

# =====================================================
# 8. Save models
# =====================================================
joblib.dump(xgb_model, 'xgb_hybrid_model.joblib')
torch.save(transformer.state_dict(), 'transformer_encoder.pt')
np.save('scaler_static_mean.npy', X_static_train.mean(axis=0))
np.save('scaler_static_std.npy', X_static_train.std(axis=0))

print('✅ Models saved!')