##🤖 AutoML Intelligence Platform

## Overview
The <b>AutoML Intelligence Platform</b> is an interactive web application built with <b>Streamlit</b> that enables users to perform <b>Automated Machine Learning (AutoML)</b> for both <b>classification</b> and <b>regression</b> tasks. Users can upload datasets, preprocess data, train multiple models, evaluate their performance, and download predictions—all without writing any code.

The platform features a <b>dark modern theme</b>, interactive visualizations using <b>Plotly Express</b>, and a clean, intuitive interface designed for an enhanced user experience.

---

## Key Features

### 1. File Upload
- Supports <b>CSV</b> and <b>Excel</b> formats.  
- Drag-and-drop functionality with a clean, intuitive interface.  
- Automatic detection of the target column (manual selection is also available).

### 2. Problem Configuration
- <b>Auto-detect</b> mode identifies whether the task is classification or regression.  
- Manual selection of <b>Classification</b> or <b>Regression</b> modes.  
- Configurable <b>test size</b>, <b>KNN neighbors</b>, <b>tree depth</b>, and <b>number of estimators</b>.

### 3. Automated Preprocessing
- Handles missing values via <b>mean</b> or <b>most frequent</b> imputation.  
- Standardizes numeric columns using <b>StandardScaler</b>.  
- One-hot encodes categorical variables automatically.

### 4. Exploratory Data Analysis (EDA)
- Dataset preview, info, and summary statistics.  
- Correlation heatmaps and numeric feature histograms with <b>Plotly Express</b>.  
- All visualizations maintain the <b>dark theme</b> and are fully interactive.

### 5. AutoML Model Training
- Supports multiple models:
  - <b>Classification:</b> Logistic Regression, Random Forest, Gradient Boosting, KNN, Decision Tree  
  - <b>Regression:</b> Linear Regression, Random Forest, Gradient Boosting, Ridge Regression, KNN, Decision Tree  
- Automatically evaluates models using <b>F1 Score</b> (classification) or <b>R² / MAE</b> (regression).  
- Highlights the <b>best-performing model</b>.

### 6. Evaluation & Visualization
- Metric cards display performance in a visually appealing format.  
- Optional <b>feature importance</b> visualization for tree-based models.  
- Confusion matrix for classification tasks.

### 7. Download Predictions
- Download predictions in <b>CSV</b>, <b>Excel</b>, or <b>JSON</b> format.

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/dharejoaneesa-debug/Auto-ml1
cd AutoML-Streamlit

## Lisence
MIT



 ## Acknowledgements
Built with Streamlit

Visualizations by Plotly

Machine learning models by scikit-learn

## Contact
MuzammilAhmed – dharejomuzammil22@gmail.com
Project Link: https://github.com/yourusername/automl-platform
