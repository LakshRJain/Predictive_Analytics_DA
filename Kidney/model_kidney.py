from sklearn.calibration import calibration_curve
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Load data
print("Loading and preprocessing data...")
df = pd.read_csv('kidney_disease_timeseries.csv')
df['date'] = pd.to_datetime(df['date'])

numerical_features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
                      'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
categorical_features = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 
                        'appet', 'pe', 'ane']

# Handle missing values
for col in numerical_features:
    df[col] = df.groupby('patient_id')[col].transform(lambda x: x.fillna(x.mean()))
for col in categorical_features:
    df[col] = df.groupby('patient_id')[col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'unknown'))

# Label encoding for categorical
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Create patient-level dataset
def create_patient_dataset(df, sequence_length=180):
    patient_data, target_data = [], []
    for patient_id in df['patient_id'].unique():
        patient_df = df[df['patient_id'] == patient_id].sort_values('date')
        if len(patient_df) < sequence_length:
            continue
        patient_seq = patient_df.tail(sequence_length)
        numerical_data = patient_seq[numerical_features].values
        categorical_data = patient_seq[categorical_features].values
        target = patient_seq['will_deteriorate'].iloc[0]
        patient_data.append((numerical_data, categorical_data))
        target_data.append(target)
    return patient_data, np.array(target_data)

patient_data, targets = create_patient_dataset(df)

# Train-validation-test split
X_train, X_test, y_train, y_test = train_test_split(
    patient_data, targets, test_size=0.2, random_state=42, stratify=targets)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Standardize numerical features
numerical_scaler = StandardScaler()
train_numerical = np.vstack([x[0] for x in X_train])
numerical_scaler.fit(train_numerical)

def scale_data(patient_data, scaler):
    scaled_data = []
    for numerical, categorical in patient_data:
        scaled_numerical = scaler.transform(numerical)
        scaled_data.append((scaled_numerical, categorical))
    return scaled_data

X_train_scaled = scale_data(X_train, numerical_scaler)
X_val_scaled = scale_data(X_val, numerical_scaler)
X_test_scaled = scale_data(X_test, numerical_scaler)

# PyTorch Dataset
class KidneyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        numerical, categorical = self.data[idx]
        return (
            torch.FloatTensor(numerical),
            torch.LongTensor(categorical),
            torch.FloatTensor([self.targets[idx]])
        )

batch_size = 32
train_loader = DataLoader(KidneyDataset(X_train_scaled, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(KidneyDataset(X_val_scaled, y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(KidneyDataset(X_test_scaled, y_test), batch_size=batch_size, shuffle=False)

# Transformer Model
class PatientTransformer(nn.Module):
    def __init__(self, numerical_dim, categorical_dims, embed_dim=64, num_heads=8, num_layers=2, hidden_dim=128):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in categorical_dims])
        self.numerical_projection = nn.Linear(numerical_dim, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(180, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=hidden_dim, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, numerical, categorical):
        numerical_proj = self.numerical_projection(numerical)
        embedded_categorical = torch.stack([emb(categorical[:,:,i]) for i, emb in enumerate(self.embeddings)], dim=0).sum(dim=0)
        combined = numerical_proj + embedded_categorical + self.pos_encoder.unsqueeze(0)
        encoded = self.transformer_encoder(combined)
        pooled = encoded.mean(dim=1)
        x = self.dropout(self.relu(self.fc1(pooled)))
        x = self.sigmoid(self.fc2(x))
        return x

categorical_dims = [len(label_encoders[col].classes_) for col in categorical_features]
model = PatientTransformer(len(numerical_features), categorical_dims)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training loop
def train_transformer(model, train_loader, val_loader, epochs=30):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for numerical, categorical, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(numerical, categorical)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for numerical, categorical, targets in val_loader:
                outputs = model(numerical, categorical)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        if (epoch+1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    return train_losses, val_losses

print("Training transformer model...")
train_losses, val_losses = train_transformer(model, train_loader, val_loader, epochs=30)

# Feature extraction for XGBoost
def extract_features(model, data_loader):
    model.eval()
    features, targets = [], []
    with torch.no_grad():
        for numerical, categorical, target in data_loader:
            numerical_proj = model.numerical_projection(numerical)
            embedded_categorical = torch.stack([emb(categorical[:,:,i]) for i, emb in enumerate(model.embeddings)], dim=-1).sum(dim=-1)
            combined = numerical_proj + embedded_categorical + model.pos_encoder.unsqueeze(0)
            encoded = model.transformer_encoder(combined)
            pooled = encoded.mean(dim=1)
            hidden = model.relu(model.fc1(pooled))
            features.append(hidden.numpy())
            targets.append(target.numpy())
    return np.vstack(features), np.vstack(targets).flatten()

print("Extracting features...")
X_train_features, y_train_xgb = extract_features(model, train_loader)
X_val_features, y_val_xgb = extract_features(model, val_loader)
X_test_features, y_test_xgb = extract_features(model, test_loader)

import xgboost as xgb
print(f"XGBoost version: {xgb.__version__}")

# Train XGBoost
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    eval_metric=['logloss', 'auc', 'aucpr']
)

# Simple fit without early stopping for now
xgb_model.fit(X_train_features, y_train_xgb)

print("XGBoost model trained successfully!")
# Hybrid prediction
def hybrid_predict(model, xgb_model, numerical, categorical):
    model.eval()
    with torch.no_grad():
        numerical_proj = model.numerical_projection(numerical)
        embedded_categorical = torch.stack([emb(categorical[:,:,i]) for i, emb in enumerate(model.embeddings)], dim=-1).sum(dim=-1)
        combined = numerical_proj + embedded_categorical + model.pos_encoder.unsqueeze(0)
        encoded = model.transformer_encoder(combined)
        pooled = encoded.mean(dim=1)
        hidden = model.relu(model.fc1(pooled))
        return xgb_model.predict_proba(hidden.numpy())[:,1]

# Evaluate hybrid
def evaluate_hybrid(model, xgb_model, data_loader):
    all_preds, all_targets = [], []
    with torch.no_grad():
        for numerical, categorical, targets in data_loader:
            preds = hybrid_predict(model, xgb_model, numerical, categorical)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy().flatten())
    return np.array(all_preds), np.array(all_targets)

print("Evaluating hybrid model...")
test_preds, test_targets = evaluate_hybrid(model, xgb_model, test_loader)

auroc = roc_auc_score(test_targets, test_preds)
auprc = average_precision_score(test_targets, test_preds)
print(f"Test AUROC: {auroc:.4f}, Test AUPRC: {auprc:.4f}")

# Confusion matrix
test_preds_binary = (test_preds>=0.5).astype(int)
cm = confusion_matrix(test_targets, test_preds_binary)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix'); plt.ylabel('True'); plt.xlabel('Predicted')
plt.savefig("confusion_matrix.png")
plt.close()


# ---------------------------
# 1. Calibration Curve
# ---------------------------
# test_preds: predicted probabilities from your hybrid model
# test_targets: true labels

prob_true, prob_pred = calibration_curve(test_targets, test_preds, n_bins=10, strategy='uniform')

plt.figure(figsize=(8,6))
plt.plot(prob_pred, prob_true, marker='o', label='Hybrid Model')
plt.plot([0,1],[0,1], linestyle='--', color='gray', label='Perfect Calibration')
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Probability')
plt.title('Calibration Curve')
plt.legend()
plt.show()


# ---------------------------
# 2. SHAP for XGBoost
# ---------------------------
# Use the features extracted from the transformer as input
# X_train_features, X_test_features: already extracted hidden features

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_features)

# Summary plot
shap.summary_plot(shap_values, X_test_features, feature_names=[f'hidden_{i}' for i in range(X_test_features.shape[1])])

# Optional: bar plot for mean absolute SHAP values
shap.summary_plot(shap_values, X_test_features, feature_names=[f'hidden_{i}' for i in range(X_test_features.shape[1])], plot_type='bar')


# Save models
torch.save(model.state_dict(), 'transformer_model.pth')
joblib.dump(xgb_model, 'xgboost_model.pkl')
joblib.dump(numerical_scaler, 'numerical_scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
