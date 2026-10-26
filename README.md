# PredictX — Predictive Maintenance & Failure Intelligence

PredictX is a machine-learning-powered predictive maintenance application that estimates the likelihood of machine failure from operational sensor readings. It combines a trained classification model, a FastAPI backend, PostgreSQL persistence, and a React + TypeScript dashboard.

The goal is to help users identify potentially risky machine conditions before failures occur.

## Features

- **Machine failure prediction:** Estimate failure probability from machine operating conditions.
- **Machine learning:** Train and compare baseline and classification models, including Dummy Classifier, Logistic Regression, and Random Forest.
- **Risk classification:** Classify predictions using a configurable probability threshold.
- **Interactive dashboard:** Submit sensor readings and view prediction results.
- **Prediction history:** Store and retrieve previous predictions using PostgreSQL.
- **REST API:** Expose health, prediction, and history endpoints through FastAPI.
- **Input validation:** Reject invalid requests with appropriate API responses.
- **Automated tests:** Test core API behavior and frontend interactions.

## Tech Stack

| Layer | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| UI and charts | Recharts, Lucide React |
| Backend | Python, FastAPI |
| Machine learning | pandas, scikit-learn, joblib |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Testing | pytest, Vitest, Testing Library |
| Version control | Git, GitHub |

## Architecture

```text
User
  |
  v
React + TypeScript Dashboard
  |
  | HTTP / REST API
  v
FastAPI Backend
  |
  +------> Trained ML Model
  |          |
  |          v
  |     Failure Probability
  |
  +------> PostgreSQL
              |
              v
       Prediction History
```

## Dataset

PredictX uses the AI4I 2020 Predictive Maintenance Dataset.

The dataset contains machine operating measurements and failure-related labels. The project explores relationships between operating conditions and machine failures, then trains classification models to predict the failure label.

The dataset includes fields such as:

- Product type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine failure label

The dataset is included in `data/ai4i2020.csv`.

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Analyze failure distribution and sensor relationships.
3. Prepare features and the target label.
4. Split data for model training and evaluation.
5. Train baseline and candidate classification models.
6. Compare performance using classification metrics, including precision, recall, F1-score, average precision, and ROC-AUC.
7. Examine feature importance to understand which inputs contribute most to the trained model's predictions.
8. Save the trained model for use by the API.
9. Apply a probability threshold to determine the displayed risk classification.

**Important:** Model performance depends on the data split, class imbalance, threshold, and evaluation methodology. Reported metrics should be treated as experimental results until confirmed with an independent final test set.

## Project Structure

```text
PredictX/
├── data/
│   └── ai4i2020.csv
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   ├── test/
│   │   ├── types/
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
├── models/
│   └── predictx_model.joblib
├── reports/
│   ├── failure_distribution.png
│   ├── sensor_comparison.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   └── feature_importance.png
├── src/
│   ├── api.py
│   ├── inspect_data.py
│   ├── explore_data.py
│   ├── train_model.py
│   ├── tune_threshold.py
│   └── explain_model.py
├── tests/
│   └── test_api.py
├── .gitignore
└── README.md
```

*The structure above summarizes the main project files; adjust it if your actual repository differs.*

## Getting Started

### Prerequisites

Install the following:

- Python
- Node.js and npm
- PostgreSQL

Run commands from the repository root unless otherwise specified.

### 1. Clone the repository

```bash
git clone https://github.com/coderXsahin/PredictX.git
cd PredictX
```

### 2. Set up the Python environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation, you can invoke the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Configure PostgreSQL

Create a PostgreSQL database named `predictx`.

Set the database password in your terminal session rather than committing credentials to Git:

```powershell
$env:DB_PASSWORD = "YOUR_POSTGRES_PASSWORD"
```

Configure the database connection to match your local PostgreSQL username, host, port, and database settings.

### 4. Start the backend

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --reload
```

The API should be available at:

- API base URL: `http://localhost:8000`
- Interactive API documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

The backend requires the trained model artifact and the configured database to be available.

### 5. Start the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, typically `http://localhost:5173`.

For a local frontend environment, set `VITE_API_BASE_URL` to `http://localhost:8000` if necessary.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/api/v1/predict` | Generate a machine failure prediction |
| GET | `/api/v1/predictions?limit=10` | Retrieve recent prediction history |

Use the interactive documentation at `/docs` to inspect request schemas and try the endpoints.

### Example prediction request

The exact fields must match the API's request schema.

```json
{
  "type": "M",
  "air_temperature": 300.0,
  "process_temperature": 310.0,
  "rotational_speed": 1500,
  "torque": 40.0,
  "tool_wear": 100
}
```

The API validates the submitted input, calculates a prediction, and returns a failure probability and classification. Successful predictions are persisted for history retrieval.

## Running Tests

### Backend

From the repository root, with the Python environment and required test dependencies installed:

```powershell
python -m pytest -v
```

### Frontend

```powershell
cd frontend
npm run test:run
npm run lint
npm run build
```

## Responsible Interpretation

PredictX is a demonstration and learning project, not a certified industrial safety system.

- A predicted probability is not a guarantee that a machine will or will not fail.
- Feature importance describes model behavior; it does not establish causation.
- Performance metrics should be evaluated with attention to class imbalance and data leakage.
- Independent validation on appropriate operational data is required before real-world deployment.

## Future Improvements

- Add independent final-test evaluation and cross-validation.
- Improve model calibration and threshold selection based on operational costs.
- Add model versioning and experiment tracking.
- Containerize the frontend, API, and database with Docker Compose.
- Add authentication, deployment automation, and monitoring.
- Deploy the application to a cloud platform.

## Author

**Sahin Khatoon**

GitHub: [@coderXsahin](https://github.com/coderXsahin)

---

**Project:** PredictX — ML-Powered Predictive Maintenance & Failure Intelligence
