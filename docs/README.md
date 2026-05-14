# 🏭 Carbon Capture Facility Screening Model

## Project Overview
This project builds a Machine Learning pipeline to classify industrial facilities in Alberta, Canada as candidates for **Carbon Capture and Storage (CCS)** or **Carbon Utilization (CU)** technology. The model helps regulators and investors quickly screen facilities using publicly available data.

## Business Problem
Testing every industrial facility for gas composition is expensive and time-consuming. This model provides a **first-pass screening tool** to identify which facilities deserve detailed analysis.

## Key Features
- ✅ Data cleaning pipeline for messy, bilingual government data
- ✅ Feature engineering to prevent data leakage
- ✅ Decision Tree model optimized for interpretability
- ✅ Handling of class imbalance (87% CCS vs 13% CU)
- ✅ **Deployed on Google Cloud Platform (Vertex AI)**

## Tech Stack
- Python, Pandas, NumPy
- Scikit-Learn (Decision Tree Classifier)
- Google Cloud Platform (Vertex AI, Cloud Storage)
- Matplotlib, Seaborn (Visualization)

## Model Performance
| Metric | Score |
|:---|:---|
| Accuracy | 76% |
| Precision (CCS) | 94% |
| Recall (CCS) | 77% |
| F1-Score (Weighted) | 0.79 |

## Project Structure