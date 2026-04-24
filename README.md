# Customer Segmentation ML Pipeline

A comprehensive machine learning pipeline for customer segmentation, featuring two interactive dashboards (HTML/JS and Streamlit) and a robust FastAPI backend.

## 🚀 Features

- **Multi-model Pipeline**: Logistic Regression, Decision Tree, and Random Forest classifiers.
- **FastAPI Backend**: Serving real-time statistics, chart data, and model predictions.
- **HTML Dashboard**: A high-performance, dark-themed dashboard using Chart.js.
- **Streamlit App**: A complete data science application with EDA, model evaluation, and hyperparameter tuning.
- **Automated Preprocessing**: KNN Imputation for missing values and stratified data splitting.

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic
- **ML/Data**: Scikit-Learn, Pandas, Numpy, Matplotlib, Seaborn
- **Frontend**: 
    - **HTML/JS**: Chart.js, CSS3 (Custom Dark Theme)
    - **Python**: Streamlit

## 📋 Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd week-3-intership
   ```

2. **Set up Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Running the Pipeline

Use the included `run.sh` script to manage services:

- **Start Everything**:
  ```bash
  ./run.sh all
  ```
  Starts HTML Dashboard (8500), API (8000), and Streamlit (8501).

- **Start API only**:
  ```bash
  ./run.sh api
  ```

- **Start Streamlit only**:
  ```bash
  ./run.sh streamlit
  ```

- **Start HTML Dashboard only**:
  ```bash
  ./run.sh html
  ```

## 🌐 Endpoints

- **HTML Dashboard**: [http://localhost:8500/dashboard.html](http://localhost:8500/dashboard.html)
- **Streamlit App**: [http://localhost:8501](http://localhost:8501)
- **API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

## 📊 API Endpoints

- `GET /data/summary`: Returns dataset statistics and segment counts.
- `GET /data/charts`: Returns formatted data for visualization.
- `POST /predict`: Predicts customer segment from input features.
