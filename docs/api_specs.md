# API Specifications

The backend is built with FastAPI and provides endpoints for both data visualization and real-time inference.

## Base URL
`http://localhost:8000`

## Endpoints

### 1. Data Summary
- **URL**: `/data/summary`
- **Method**: `GET`
- **Description**: Returns general statistics about the customer dataset.
- **Response Example**:
  ```json
  {
    "total_customers": 2000,
    "total_revenue": 4108000.50,
    "avg_spend": 2054.00,
    "segment_counts": {
        "Standard": 700,
        "Premium": 450,
        "Basic": 450,
        "Inactive": 400
    }
  }
  ```

### 2. Chart Data
- **URL**: `/data/charts`
- **Method**: `GET`
- **Description**: Returns pre-processed data ready for consumption by frontend charting libraries.
- **Data included**:
    - `segments`: Count of customers per segment.
    - `revenue_by_segment`: Average annual spend per segment.
    - `correlations`: Feature-to-feature correlation matrix.

### 3. Prediction
- **URL**: `/predict`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "annual_spend": 2500.0,
    "visits_per_month": 10,
    "items_per_order": 5,
    "support_tickets": 0
  }
  ```
- **Response**:
  ```json
  {
    "prediction": "Premium",
    "probabilities": {
        "Basic": 0.05,
        "Inactive": 0.01,
        "Premium": 0.85,
        "Standard": 0.09
    }
  }
  ```

## Development
To run in development mode with auto-reload:
```bash
uvicorn api:app --reload --port 8000
```
