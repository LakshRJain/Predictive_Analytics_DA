import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# =====================================================
# 1. Load and preprocess new dataset
# =====================================================
print("Loading and preprocessing new dataset...")
df = pd.read_csv('heart_patients_full.csv')

# ---- Update these lists if your dataset columns have changed ----
ts_features = [
    'systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 
    'spo2', 'cholesterol_total', 'hdl', 'ldl', 'triglycerides', 
    'glucose', 'hba1c', 'creatinine', 'medication_adherence'
]

static_features = ['age', 'sex', 'ethnicity', 'bmi', 'smoking_status', 'alcohol_use', 'physical_activity_level']

label_col = 'deplict'   # <-- Change here if your label column name is different

# Encode categorical features
categorical_cols = ['sex', 'ethnicity', 'smoking_status', 'alcohol_use', 'physical_activity_level']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Fill missing values (per patient for time-series)
for col in ts_features + static_features:
    if col in categorical_cols:
        df[col] = df.groupby('patient_id')[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 0)
        )
    else:
        df[col] = df.groupby('patient_id')[col].transform(lambda x: x.fillna(x.mean()))

# Build patient sequences
patients, labels, static = [], [], []
for pid, group in df.groupby('patient_id'):
    group = group.sort_values('days_since_start')

    # Ensure each patient has full sequence (pad/trim if needed)
    seq = group[ts_features].values
    if len(seq) < 180:  # pad short sequences
        pad_len = 180 - len(seq)
        seq = np.vstack([seq, np.zeros((pad_len, len(ts_features)))])
    elif len(seq) > 180:  # trim long sequences
        seq = seq[:180]

    patients.append(seq)
    static.append(group[static_features].iloc[0].values)
    labels.append(group[label_col].iloc[0])

X_seq = np.stack(patients)   # (N_patients, 180, num_ts_features)
X_static = np.stack(static)  # (N_patients, num_static_features)
y = np.array(labels)

print(f"✅ New dataset processed: {X_seq.shape[0]} patients, {X_seq.shape[1]} days, {X_seq.shape[2]} features")
# =====================================================
# Train/val/test split
# =====================================================
X_seq_train, X_seq_test, X_static_train, X_static_test, y_train, y_test = train_test_split(
    X_seq, X_static, y, test_size=0.2, random_state=42, stratify=y
)
X_seq_train, X_seq_val, X_static_train, X_static_val, y_train, y_val = train_test_split(
    X_seq_train, X_static_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Scale static features
static_scaler = StandardScaler()
X_static_train_scaled = static_scaler.fit_transform(X_static_train)
X_static_val_scaled = static_scaler.transform(X_static_val)
X_static_test_scaled = static_scaler.transform(X_static_test)

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
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.fc_out(x)
        return x.squeeze(-1)

# =====================================================
# 3. Train Transformer
# =====================================================
def train_transformer(X_seq_train, y_train, X_seq_val, y_val, input_dim, epochs=10, batch_size=32):
    model = TransformerEncoder(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_loader = DataLoader(PatientDataset(X_seq_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PatientDataset(X_seq_val, y_val), batch_size=batch_size)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        # Training
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
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'transformer_encoder_heart.pt')

    # Load best model
    model.load_state_dict(torch.load('transformer_encoder_heart.pt'))
    return model, train_losses, val_losses

# Train transformer
print("Training transformer model...")
transformer, train_losses, val_losses = train_transformer(
    X_seq_train, y_train, X_seq_val, y_val, 
    input_dim=len(ts_features), epochs=10
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
            x = x.transpose(1, 2)
            x = model.pool(x).squeeze(-1)
            embs.append(x.numpy())
    return np.vstack(embs)

print("Extracting embeddings...")
emb_train = get_embeddings(transformer, X_seq_train)
emb_val = get_embeddings(transformer, X_seq_val)
emb_test = get_embeddings(transformer, X_seq_test)

# =====================================================
# 5. XGBoost on [embeddings + static] - SIMPLIFIED
# =====================================================
Xgb_train = np.hstack([emb_train, X_static_train_scaled])
Xgb_val = np.hstack([emb_val, X_static_val_scaled])
Xgb_test = np.hstack([emb_test, X_static_test_scaled])

print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

# Simple fit without early stopping
xgb_model.fit(Xgb_train, y_train)

# =====================================================
# 6. Evaluation
# =====================================================
print("Evaluating model...")
y_pred_proba = xgb_model.predict_proba(Xgb_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# AUROC and AUPRC
auroc = roc_auc_score(y_test, y_pred_proba)
auprc = average_precision_score(y_test, y_pred_proba)
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Risk', 'Risk'], 
            yticklabels=['No Risk', 'Risk'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_heart.png')
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auroc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve_heart.png')
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AP = {auprc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('pr_curve_heart.png')
plt.close()

# Calibration Curve
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
plt.legend()
plt.savefig('calibration_curve_heart.png')
plt.close()

# Brier Score
brier_score = brier_score_loss(y_test, y_pred_proba)
print(f"Brier Score: {brier_score:.4f}")

# =====================================================
# 7. SHAP Explainability
# =====================================================
print("Generating SHAP explanations...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(Xgb_test)

# Feature names for SHAP
feature_names = [f'emb_{i}' for i in range(emb_test.shape[1])] + static_features

# Summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, Xgb_test, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('shap_summary_heart.png')
plt.close()

# =====================================================
# 8. Save Models and Artifacts
# =====================================================
print("Saving models...")
torch.save(transformer.state_dict(), 'transformer_encoder_heart.pt')
joblib.dump(xgb_model, 'xgb_heart_model.joblib')
joblib.dump(static_scaler, 'static_scaler_heart.joblib')
joblib.dump(label_encoders, 'label_encoders_heart.joblib')
np.save('static_features_heart.npy', np.array(static_features))
np.save('ts_features_heart.npy', np.array(ts_features))

print("✅ All models and artifacts saved successfully!")
print("✅ Training completed!")

# Print final metrics
print(f"\n=== FINAL RESULTS ===")
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(f"Brier Score: {brier_score:.4f}")
print(f"Confusion Matrix:\n{cm}")