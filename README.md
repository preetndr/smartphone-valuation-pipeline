<div align="center">

# 📱 Smartphone Valuation Pipeline

<br>

<a href="https://smartphone-price-predictor.streamlit.app/">
  <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-red?style=for-the-badge&logo=streamlit" />
</a>
<a href="https://www.kaggle.com/code/preetndr/mobile-price-prediction-linear-regression">
  <img src="https://img.shields.io/badge/Kaggle-Notebook-blue?style=for-the-badge&logo=kaggle" />
</a>
<a href="https://github.com/preetndr/smartphone-valuation-pipeline">
  <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github" />
</a>

<br><br>

<img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python" />
<img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=flat-square&logo=scikitlearn" />
<img src="https://img.shields.io/badge/Streamlit-Deployed-red?style=flat-square&logo=streamlit" />
<img src="https://img.shields.io/badge/Status-Production%20Ready-success?style=flat-square" />

</div>

### Intelligent Smartphone Price Prediction Using Machine Learning

A production-ready machine learning pipeline that predicts smartphone market value from hardware specifications using advanced preprocessing, feature engineering, statistical validation, and regression modeling.

<br>

## 🔗 Live Application

👉 [**Launch Streamlit App**](https://smartphone-price-predictor.streamlit.app/)

## 📓 Kaggle Notebook

👉 [**View Full Project Notebook**](https://www.kaggle.com/code/preetndr/mobile-price-prediction-linear-regression)

## 📊 Dataset Source

👉 [**Mobile Price Prediction Dataset**](https://www.kaggle.com/datasets/mohannapd/mobile-price-prediction)

<br>

---

# ✨ Project Overview

This project presents a complete end-to-end machine learning workflow for predicting smartphone market prices based on hardware specifications and device characteristics.

The system combines:

* Advanced preprocessing pipelines
* Custom feature engineering
* Statistical analysis
* Multicollinearity validation
* Regression modeling
* Cross-validation
* Production deployment with Streamlit

The final model is deployed as an interactive web application that allows users to estimate smartphone valuation in real time.

---

# 🖥️ Application Preview

<div align="center">

<img src="screenshot.jpeg" width="950" alt="Smartphone Valuation Streamlit Application">

<br>
<br>

<i>Interactive ML-powered smartphone valuation interface deployed with Streamlit.</i>

</div>

The deployed Streamlit application provides an interactive premium interface for estimating smartphone valuation in real time using hardware specifications and the trained machine learning pipeline.

---

# 🎯 Problem Statement

Smartphone pricing is influenced by multiple interacting hardware specifications such as RAM, processor frequency, display quality, battery capacity, and camera configuration.

The objective of this project is to build a machine learning pipeline capable of learning these relationships and accurately estimating smartphone market value using structured device specifications.

---

# 🧠 Machine Learning Workflow

## 1. Data Collection

The dataset was sourced from Kaggle and contains smartphone specifications including:

* Weight
* Display resolution
* Pixel density (PPI)
* CPU cores
* CPU frequency
* Internal memory
* RAM
* Camera specifications
* Battery capacity
* Device thickness

Dataset Size:

* **161 smartphone records**
* **14 structured features**

---

## 2. Data Validation & Quality Checks

The dataset underwent multiple validation steps before model training:

### ✔ Missing Value Analysis

* Verified missing-value consistency across all features
* Implemented median-based imputation inside the pipeline

### ✔ Duplicate Validation

* Confirmed zero duplicate observations

### ✔ Statistical Profiling

* Distribution analysis
* Outlier inspection
* Feature skewness analysis

### ✔ Multicollinearity Detection

* Variance Inflation Factor (VIF) analysis performed
* Highly correlated feature relationships were examined before final training

---

<br>

# ⚙️ Feature Engineering

Custom engineered features were introduced to improve predictive capability:

| Engineered Feature  | Description                                           |
| ------------------- | ----------------------------------------------------- |
| `performance_score` | Combines CPU frequency, CPU cores, and RAM            |
| `display_quality`   | Derived from PPI and display resolution               |
| `ram_mem`           | Captures interaction between RAM and internal storage |

These engineered variables significantly improved the model’s ability to capture real-world hardware pricing relationships.

---

<br>

# 🔒 Zero Data Leakage Architecture

This project was intentionally designed to eliminate data leakage and preserve evaluation integrity.

### Why the pipeline is leak-proof:

* The dataset was split into training and testing subsets **before** any preprocessing or transformations were applied.

* All preprocessing operations were wrapped inside a unified `sklearn Pipeline`, ensuring scalers, imputers, and transformations learned strictly from training data — even during cross-validation.

* The `Sale` column was explicitly removed during preprocessing to prevent target leakage from future post-pricing information.

This guarantees that evaluation metrics reflect true generalization performance.

---

# 🧪 Preprocessing Pipeline

The final production pipeline includes:

```python
Pipeline([
    ('custom_transform', Custom_Transform()),
    ('transform', transform),
    ('scaler', StandardScaler()),
    ('model', model)
])
```

### Pipeline Components

* Custom preprocessing transformer
* Median imputation
* Feature engineering
* Box-Cox transformations
* Standard scaling
* Regression model training

---

<br>

# 📈 Exploratory Data Analysis

The notebook includes detailed EDA covering:

* Distribution plots
* Outlier analysis
* Correlation heatmaps
* Pairwise feature relationships
* Skewness analysis
* Statistical summaries

### Key Findings

* Premium hardware specifications showed stronger positive correlation with smartphone pricing.
* Storage capacity and RAM exhibited strong influence on valuation.
* Several features displayed skewed distributions, motivating the use of Box-Cox transformations.
* Engineered features improved feature-target relationships significantly.

---

<br>

# 🤖 Models Evaluated

Multiple regression algorithms were trained and compared:

| Model             | Test R²    |
| ----------------- | ---------- |
| Linear Regression | **0.9345** |
| Lasso Regression  | 0.9316     |
| Ridge Regression  | 0.9271     |
| ElasticNet        | 0.9258     |

### Final Selected Model

✅ **Linear Regression**

The Linear Regression pipeline achieved the strongest overall balance between:

* Predictive accuracy
* Generalization capability
* Stability during cross-validation
* Simplicity and interpretability

---

<br>

# 📊 Model Performance

## Final Evaluation Metrics

| Metric    | Value      |
| --------- | ---------- |
| Test R²   | **0.9345** |
| Test RMSE | **191.39** |
| Test MAE  | **163.24** |

### Cross-Validation Performance

| Metric                   | Value      |
| ------------------------ | ---------- |
| Cross-Validation Test R² | **0.9352** |
| Cross-Validation RMSE    | **176.36** |

The close alignment between training, testing, and cross-validation performance indicates strong generalization and minimal overfitting.

---

<br>

# 🚀 Streamlit Deployment

The trained pipeline was serialized using `pickle` and deployed through a fully interactive Streamlit application.

### Application Features

* Real-time smartphone valuation
* Interactive hardware sliders
* Custom UI/UX styling
* Smooth animations and transitions
* Responsive interface
* Production-ready inference pipeline

### Live Demo

🔗 [https://smartphone-price-predictor.streamlit.app/](https://smartphone-price-predictor.streamlit.app/)

---

<br>

# 🛠️ Tech Stack

## Machine Learning

* Scikit-learn
* NumPy
* Pandas
* Statsmodels

## Visualization

* Matplotlib
* Seaborn

## Deployment

* Streamlit
* Pickle Serialization

## Development

* Python
* Jupyter Notebook
* Kaggle

---

<br>

# 📂 Repository Structure

```text
smartphone-valuation-pipeline/
│
├── application.py
├── phone_price_pipeline.pkl
├── notebook.ipynb
├── requirements.txt
├── image.png
└── README.md
```

---

<br>

# ⚡ Installation

## Clone Repository

```bash
git clone https://github.com/preetndr/smartphone-valuation-pipeline.git
cd smartphone-valuation-pipeline
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit Application

```bash
streamlit run app.py
```

---

<br>

# 🔍 Example Prediction Workflow

1. Select smartphone hardware specifications
2. Submit configuration through the Streamlit interface
3. Pipeline performs:

   * preprocessing
   * feature engineering
   * scaling
   * inference
4. Predicted smartphone valuation is generated instantly

---

<br>

# 📌 Future Improvements

Potential future enhancements include:

* Expanding dataset size
* I**ncorporating modern smartphone ben**chmark scores
* Testing ensemble and boosting models
* Integrating deep learning approaches
* API deployment with FastAPI
* Real-time market price scraping
* Automated retraining workflows

---

<br>

# 🙌 Acknowledgements

* Kaggle Dataset Provider
* Streamlit
* Scikit-learn Documentation
* Open-source Python ML ecosystem

---

<br>

# 👨‍💻 Author

### Preetinder Singh

* GitHub: [https://github.com/preetndr](https://github.com/preetndr)
* Kaggle: [https://www.kaggle.com/preetndr/code](https://www.kaggle.com/preetndr/code)

---

<div align="center">

---

### ⭐ If you found this project interesting, consider starring the repository.

</div>
