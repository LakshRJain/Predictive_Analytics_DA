# Distributed IoT-Fog-Cloud Architecture for Chronic Patient Monitoring

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)

This repository implements a three-tier **IoT-Fog-Cloud** framework designed for real-time monitoring and 180-day longitudinal risk assessment of chronic diseases. By fusing **Transformer-based temporal embeddings** with **XGBoost static classification**, the model achieves high-fidelity deterioration forecasting.



## 🏗 System Architecture

The framework is distributed across three hierarchical layers to balance latency and computational depth:

1.  **Edge Layer**: Lightweight thresholding on IoT sensors for instantaneous detection of life-threatening anomalies ($SpO_2$ drops, tachycardia).
2.  **Fog Layer**: Localized processing hub for data cleaning, noise reduction, and normalization of streaming physiological vitals.
3.  **Cloud Layer**: Hosts the **Hybrid Temporal-Static Fusion (HTSF)** engine for deep longitudinal analysis.



---

## 🩺 Disease Models & Parameters

We provide specialized predictive pipelines for three major chronic conditions, utilizing foundational parameters from the **MIMIC-IV** clinical database.

| Disease | Primary Temporal Vitals | Static Markers |
| :--- | :--- | :--- |
| **Heart Disease** | Heart Rate, $SpO_2$, Blood Pressure | Age, Sex, Smoking Status |
| **Chronic Kidney Disease**| Serum Creatinine, Hemoglobin, Albumin | BMI, History of Hypertension |
| **Diabetes/Obesity** | Glucose Levels, Physical Activity | Income, Education, BMI |

---

## 🧠 The HTSF Model
The **Hybrid Temporal-Static Fusion** model follows a two-stage inference process:
* **Stage 1**: A **Transformer Encoder** processes 180-day time-series windows to extract latent temporal features.
* **Stage 2**: These embeddings are concatenated with static demographics and fed into an **XGBoost** classifier to produce a calibrated risk score.

### Explainability with SHAP
To ensure clinical trust, we utilize **SHAP (SHapley Additive exPlanations)**. This allows clinicians to see exactly which temporal trends (e.g., a 10-day decline in $SpO_2$) contributed to a high-risk alert.



---

## 📊 Performance Summary
* **AUROC**: $>0.85$ across all cohorts.
* **Calibration**: High alignment with the 45-degree ideal diagonal, ensuring predicted risk matches real-world incidence.
* **Latency**: Reduced by ~40% through Edge-assisted pre-screening.
