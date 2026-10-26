# PredictX - Predictive Maintenance & Failure Intelligence



PredictX is a machine-learning-powered predictive maintenance application that estimates machine failure risk from operating and sensor data. It combines a trained classification model, a FastAPI backend, PostgreSQL persistence, and a React + TypeScript dashboard.



The goal is to help users explore machine readings, identify potentially high-risk operating conditions, and review previous predictions through a web interface.



## Features



- **Failure prediction:** Estimate failure probability from machine operating parameters.

- **Machine-learning pipeline:** Compare baseline models with a Random Forest classifier.

- **Risk classification:** Convert predicted probabilities into risk categories using a configurable decision threshold.

- **Interactive dashboard:** Submit machine readings and view prediction results.

- **Prediction history:** Persist and retrieve previous predictions using PostgreSQL.

- **REST API:** Expose prediction and health-check endpoints through FastAPI.

- **Input validation:** Reject invalid requests with appropriate validation errors.

- **Automated tests:** Test core backend and frontend functionality.



## Technology Stack



| Component | Technologies |

|---|---|

| Machine learning | Python, pandas, NumPy, scikit-learn |

| Backend API | FastAPI, Uvicorn, Pydantic |

| Database | PostgreSQL, SQLAlchemy |

| Model persistence | Joblib |

| Frontend | React, TypeScript, Vite |

| UI and charts | Lucide React, Recharts |

| Testing | pytest, Vitest, Testing Library |

| Version control | Git, GitHub |



## Architecture



```text

Machine Sensor Readings

&#x20;         |

&#x20;         v

React + TypeScript Dashboard

&#x20;         |

&#x20;         v

FastAPI REST API

&#x20;         |

&#x20;         v

Input Validation

&#x20;         |

&#x20;         v

Trained ML Model

&#x20;         |

&#x20;         v

Failure Probability + Risk Status

&#x20;         |

&#x20;         v

PostgreSQL Prediction History

&#x20;         |

&#x20;         v

Dashboard Results

```



## Dataset



PredictX uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.



The dataset contains simulated industrial machine operating measurements and failure-related labels.



Important features include:



- Air temperature

- Process temperature

- Rotational speed

- Torque

- Tool wear

- Machine type



The dataset contains 10,000 records, with 339 machine-failure cases (3.39%). This class imbalance makes precision, recall, F1-score, and Average Precision important evaluation metrics alongside ROC-AUC.



Dataset source: [UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)



## Machine-Learning Workflow



1. Load and inspect the dataset.

2. Explore failure distributions and relationships between machine measurements.

3. Prepare the input features and target label.

4. Train baseline and candidate classification models.

5. Compare model performance using classification metrics.

6. Tune the decision threshold to balance missed failures against false alarms.

7. Save the trained model for API inference.

8. Generate feature-importance reports to help interpret the model.



**Important:** Feature importance describes how the trained model uses its inputs; it does not establish that a feature causes machine failure.



## Project Structure



```text

PredictX/

├── data/

│   └── ai4i2020.csv

├── frontend/

│   ├── src/

│   ├── package.json

│   └── vite.config.ts

├── models/

│   └── predictx_model.joblib

├── reports/

│   ├── model_comparison.csv

│   ├── feature_importance.csv

│   ├── failure_distribution.png

│   ├── sensor_comparison.png

│   └── correlation_heatmap.png

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

├── requirements.txt

└── README.md

```



The structure above highlights the main project components. Check the actual repository if any path differs.



## Getting Started



### Prerequisites



Install the following before running the application:



- Python

- Node.js and npm

- PostgreSQL

- Git



### 1. Clone the repository



```powershell

git clone https://github.com/coderXsahin/PredictX.git

cd PredictX

```



### 2. Create the Python environment



```powershell

python -m venv .venv

.\.venv\\Scripts\\Activate.ps1

python -m pip install -r requirements.txt

```



If PowerShell blocks virtual-environment activation, use the environment's Python executable directly instead.



### 3. Configure PostgreSQL



Create a PostgreSQL database named `predictx`.



Set the database password in your current PowerShell session:



```powershell

$env:DB_PASSWORD = "YOUR_POSTGRES_PASSWORD"

```



Replace the placeholder with your actual local PostgreSQL password. Do not commit passwords or `.env` secrets to GitHub.



Confirm that the database connection settings in the backend match your local PostgreSQL configuration.



### 4. Start the backend



From the project root:



```powershell

.\.venv\\Scripts\\python.exe -m uvicorn src.api:app --reload

```



The API should be available at:



- API base URL: `http://localhost:8000`

- Interactive API documentation: `http://localhost:8000/docs`

- Health check: `http://localhost:8000/health`



The database must be running, and the required model artifact must be present before using prediction functionality.



### 5. Start the frontend



Open a second PowerShell terminal:



```powershell

cd C:\\Users\\moham\\Downloads\\PredictX\\frontend

npm install

npm run dev

```



Open the local URL printed by Vite, normally `http://localhost:5173`.



If necessary, configure the frontend API base URL using its local environment file:



```text

VITE_API_BASE_URL=http://localhost:8000

```



Do not commit local environment files containing secrets.



## API Endpoints



| Method | Endpoint | Purpose |

|---|---|---|

| GET | `/health` | Check backend health |

| POST | `/api/v1/predict` | Generate a machine-failure prediction |

| GET | `/api/v1/predictions?limit=10` | Retrieve recent prediction history |



For the exact request schema and response fields, refer to the running API documentation at `/docs`.



## Testing



### Backend tests



From the project root:



```powershell

.\.venv\\Scripts\\python.exe -m pytest -v

```



### Frontend tests



From the `frontend` directory:



```powershell

npm run test:run

npm run lint

npm run build

```



These commands run the frontend tests, check code quality, and create a production build.



## Responsible Use and Limitations



- A predicted probability is a model estimate, not a guarantee that a machine will fail.

- A decision threshold affects the balance between false alarms and missed failures.

- The model was developed using the AI4I 2020 dataset; performance on real industrial equipment requires independent validation.

- The application is a learning and demonstration project, not a certified industrial safety system.

- Reported model performance should be evaluated on an appropriately held-out dataset before making production claims.



## Future Improvements



- Add model versioning and experiment tracking.

- Add drift monitoring and model-performance reports.

- Improve validation using an untouched test set.

- Add authentication and role-based access control.

- Containerize the application with Docker.

- Add continuous integration for automated testing.

- Deploy the application to a cloud environment.



## Author



**Sahin Khatoon**



GitHub: [@coderXsahin](https://github.com/coderXsahin)



---



*PredictX — turning machine data into actionable failure-risk insights.*
