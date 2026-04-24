# Dashboard Guide

The project includes two complementary frontends. Both are designed to provide a unified user experience while serving different use cases.

## 1. HTML Executive Dashboard
**Best for**: Presentation and high-level reporting.
- **Technology**: Vanilla HTML/CSS/JS, Chart.js.
- **Port**: 8500
- **Features**:
    - **KPI Grid**: Real-time metrics from the API (Total Revenue, Customers, etc.).
    - **Interactive Charts**: Responsive charts for revenue trends and segmentation.
    - **Sidebar Navigation**: Quick switching between EDA, Model Building, and API views.
    - **Real-time API Sync**: Uses the `fetch` API to synchronize with the FastAPI backend.

## 2. Streamlit ML App
**Best for**: Data Scientists and technical stakeholders.
- **Technology**: Python, Streamlit.
- **Port**: 8501
- **Features**:
    - **Executive View**: Matches the aesthetic and metrics of the HTML dashboard.
    - **Deep EDA**: Automated histograms, correlation heatmaps, and dataset samples.
    - **Model Evaluation**: Confusion matrices, classification reports, and feature importance.
    - **Hyperparameter Tuning**: Interactive GridSearchCV on the Random Forest model.
    - **Prediction Tool**: A form-based tool to predict segments with probability visualization.

## Consistency
Both dashboards use a shared color palette for segments:
- **Premium**: `#667eea`
- **Standard**: `#764ba2`
- **Basic**: `#11998e`
- **Inactive**: `#38ef7d`
