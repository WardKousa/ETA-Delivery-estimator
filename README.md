# Delivery ETA Estimator

*Video demo comming very soon*

Predict delivery Estimated Time of Arrival (ETA) using Linear Regression and Random Forest models with a multi-page Streamlit app.

## Features

- Multi-page Streamlit app:
  - Home: ETA predictions
  - Linear Regression Analysis: coefficients, feature contributions, residual diagnostics
  - Random Forest Analysis: permutation importances, top contributing features, residual diagnostics
  - Comparison: side-by-side model metrics and feature importance
  - Metrics Info: explanations of MAE, residuals, and feature importance
- Dockerized setup for reproducible demos
- Training script (`training.py`) to regenerate models and metrics

## Installation and Run

Clone the repo and (optionally) create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

# Install dependencies```
```pip install streamlit==1.30.0 pandas==2.1.0 numpy==1.27.0 matplotlib==3.8.0 scikit-learn==1.3.0 joblib==1.3.2```


# Run the app
```streamlit run app.py```

## Run with Docker

If you have Docker installed, you can run the app without installing Python or any libraries:

```bash
# Build the Docker image
docker build -t delivery-eta-app .
```
# Run the container
```docker run -p 8501:8501 delivery-eta-app```
