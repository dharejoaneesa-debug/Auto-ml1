 AutoML Intelligence Platform
Overview

The AutoML Intelligence Platform is a fully interactive web application built with Streamlit that allows users to perform Automated Machine Learning (AutoML) for both classification and regression tasks without writing any code. The platform automatically handles data preprocessing, model training, evaluation, and visualization, providing users with insights and predictions in real-time.

The app emphasizes a dark modern theme with a visually appealing dashboard powered by Plotly Express for charts and Streamlit custom styling.

Features
1. Interactive File Upload
Supports CSV and Excel files.
Drag-and-drop interface with clear instructions.
Automatic detection of the dataset’s target column (can be manually selected).
2. Problem Configuration
Supports Auto-detection, Classification, or Regression modes.
Configurable test size, model hyperparameters (e.g., KNN neighbors, tree depth, number of estimators).
3. Automated Preprocessing
Handles missing values with mean or most frequent imputation.
Standard scaling for numeric features.
One-hot encoding for categorical features.
4. Exploratory Data Analysis (EDA)
Dataset preview, info, and summary statistics.
Correlation heatmaps using Plotly Express.
Histograms for numeric features.
All visualizations match the dark theme and maintain clarity.
5. AutoML Model Training
Supports multiple models:
Classification: Logistic Regression, Random Forest, Gradient Boosting, KNN, Decision Tree.
Regression: Linear Regression, Random Forest, Gradient Boosting, Ridge Regression, KNN, Decision Tree.
Automatically evaluates models based on F1 score (classification) or R²/MAE (regression).
Highlights the best-performing model.
6. Evaluation & Visualization
Display metrics in visually appealing metric cards.
Optional feature importance visualization.
Confusion matrix for classification tasks.
7. Download Predictions
Users can download predictions in CSV, Excel, or JSON formats.
Installation
Clone the repository
git clone https://github.com/dharejoaneesa-debug/Auto-ml1
cd AutoML-Streamlit
Install dependencies
pip install -r requirements.txt
Requirements (requirements.txt)
streamlit>=1.25.0
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.3.0
plotly>=5.17.0
openpyxl>=3.1.2
Usage
Run the Streamlit app:
streamlit run app.py
Open the URL provided in the terminal (usually http://localhost:8501).
Upload your dataset, configure problem type, select target column, adjust model parameters, and click Run AutoML.
Explore EDA visualizations, view model metrics, and download predictions.
Design & Styling
Dark theme with purple accents (#7C5CFF) for an elegant dashboard appearance.
Interactive Plotly charts for correlation heatmaps, histograms, and feature importance.
Clean cards, sliders, and buttons for a modern user experience.
File uploader styled for light background with black text to enhance readability.
Notes
Best suited for small to medium datasets due to browser/Streamlit performance.
Auto-detect feature works by checking if the target is categorical or numeric.
Supports both classification and regression in a single unified interface.
Authors

Aleeza Bashir – Full-stack AutoML & Data Science Enthusiast
Contact: aleezabashir3@gmail.com
