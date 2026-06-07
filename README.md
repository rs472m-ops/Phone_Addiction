# 📱 Phone Addiction Level Predictor
### An End-to-End Machine Learning Project with MLOps

##  Problem Statement

Smartphone overuse is a growing concern affecting millions of people worldwide — impacting mental health, sleep quality, academic performance, and social relationships. This project builds a **production-ready ML system** that predicts a person's phone addiction level **(0–10)** based on their behavioral and psychological patterns.

## Objective

> To design, build, evaluate, and deploy a machine learning solution using a structured **4-Sprint Agile methodology** with complete **MLOps integration** — covering pipelines, experiment tracking, hyperparameter tuning, model comparison, and deployment.

---

##  Dataset

| Property | Detail |
|---|---|
| **File** | `phone_addiction_dataset.csv` |
| **Rows** | 6,000 records |
| **Columns** | 22 features |
| **Target** | `Addiction_Level` (continuous, 0–10) |
| **Problem Type** | Regression |

### Features Used

| Feature | Description |
|---|---|
| `Age` | Age of the user |
| `Gender` | Male / Female / Other |
| `Daily_Usage_Hours` | Hours of phone use per day |
| `Sleep_Hours` | Hours of sleep per night |
| `Interllectual_Performance` | Academic/cognitive performance score |
| `Social_Interactions` | Number of social interactions per day |
| `Exercise_Hours` | Hours of exercise per day |
| `Anxiety_Level` | Self-reported anxiety (0–10) |
| `Depression_Level` | Self-reported depression (0–10) |
| `Self_Esteem` | Self-esteem score (0–10) |
| `Screen_Time_Before_Bed` | Hours of screen use before sleeping |
| `Phone_Checks_Per_Day` | Number of times phone checked daily |
| `Apps_Used_Daily` | Number of apps used per day |
| `Time_on_Social_Media` | Hours on social media daily |
| `Time_on_Gaming` | Hours on gaming daily |
| `Time_on_Education` | Hours on education apps daily |
| `Phone_Usage_Purpose` | Primary purpose of phone use |
| `Family_Communication` | Hours spent communicating with family |
| `Weekend_Usage_Hours` | Hours of phone use on weekends |
| `Addiction_Level` | **Target** — addiction score (0–10) |

### Engineered Features (Sprint 3)

| Feature | Formula |
|---|---|
| `Usage_Sleep_Ratio` | `Daily_Usage_Hours / Sleep_Hours` |
| `Total_Screen_Exposure` | `Daily_Usage_Hours + Weekend_Usage_Hours` |
| `Total_Content_Time` | `Social Media + Gaming + Education` |
| `High_Usage_Flag` | `1 if Daily_Usage > median else 0` |
| `Stress_Score` | `Anxiety_Level + Depression_Level` |

---

## 🏃 4-Sprint Methodology

### Sprint 1 — Data Understanding & Preprocessing
- Loaded and inspected dataset (`.head()`, `.info()`, `.describe()`)
- Handled missing values and duplicates
- Performed EDA — Univariate, Bivariate, Multivariate analysis
- Detected and treated outliers using IQR and boxplots
- Label Encoded categorical features
- Applied Feature Scaling (StandardScaler)
- Train / Test Split — 80% / 20%

### Sprint 2 — Model Building & Evaluation
- Trained 4 baseline regression models
- Evaluated using MAE, RMSE, R²
- Checked overfitting via Train vs Test score comparison
- Built Model Comparison Table

### Sprint 3 — Optimization & Final Model
- Created 5 domain-specific engineered features
- Removed highly correlated features (threshold > 0.9)
- Applied Recursive Feature Elimination (RFE)
- Hyperparameter tuning with GridSearchCV & RandomizedSearchCV
- Saved final optimized model as `final_pipeline.joblib`

### Sprint 4 — Deployment & MLOps
- Built end-to-end Sklearn Pipeline (Scaler + Model)
- Tracked all experiments with MLflow
- Compared 8 models using Optuna (20 trials each)
- Deployed live Streamlit web app
- Logged predictions and monitoring plots
- Wrote modular source code
- Pushed to GitHub

---

## 🤖 Models Trained & Compared

| Model | Type | Tracked in MLflow |
|---|---|---|
| Linear Regression | Baseline | ✅ |
| Ridge Regression | Regularized | ✅ |
| Lasso Regression | Regularized | ✅ |
| Decision Tree | Tree-based | ✅ |
| K-Nearest Neighbors | Distance-based | ✅ |
| Support Vector Regression | Kernel-based | ✅ |
| **Random Forest** | **Final Model ✅** | ✅ |
| Gradient Boosting | Ensemble | ✅ |

---

## 🛠️ Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.13 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| Pipeline | Sklearn Pipeline |
| Model Saving | Joblib |
| Frontend | Streamlit |
| Version Control | Git & GitHub |
| IDE | VS Code + Jupyter Notebook |

---

## ⚙️ MLOps Practices

| Practice | Tool | Details |
|---|---|---|
| Experiment Tracking | MLflow | Params, metrics, artifacts per run |
| Hyperparameter Tuning | Optuna + MLflow | 20 trials per model, nested runs |
| Multi-model Comparison | MLflow UI | 8 models in one experiment |
| Pipeline | Sklearn Pipeline | Prevents data leakage |
| Model Serialization | Joblib | `.joblib` file for deployment |
| Prediction Logging | CSV | Every prediction saved with timestamp |
| Performance Monitoring | Matplotlib | Actual vs Predicted + Residual plots |
| Modular Code | Python modules | `preprocessing.py`, `train.py`, `predict.py` |
| Version Control | GitHub | Full project versioned |
| Documentation | README.md | This file |

---

## 📁 Project Structure

```
phone_addiction_ml/
│
├── data/
│   └── phone_addiction_dataset.csv
│
├── notebooks/
│   ├── Sprint_2_model_building_evaluation.ipynb
│   ├── Sprint_3_optimization_final_model.ipynb
│   └── Sprint_4_deployment_mlops.ipynb
│
├── src/
│   ├── preprocessing.py       ← cleaning + feature engineering
│   ├── train.py               ← model training script
│   └── predict.py             ← prediction script
│
├── models/
│   └── final_pipeline.joblib  ← saved final model
│
├── app/
│   └── app.py                 ← Streamlit web app
│
├── logs/
│   ├── experiment_log.json    ← params + metrics log
│   └── prediction_log.csv     ← prediction history
│
├── mlruns/                    ← auto-created by MLflow
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/phone-addiction-ml.git
cd phone-addiction-ml
```

### 2. Create Virtual Environment
```bash
python -m venv myenv
source myenv/bin/activate        # Mac/Linux
myenv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Jupyter Notebooks
```bash
jupyter notebook
```
Run notebooks in order:
- `Sprint_2_model_building_evaluation.ipynb`
- `Sprint_3_optimization_final_model.ipynb`
- `Sprint_4_deployment_mlops.ipynb`

### 5. Launch Streamlit App
```bash
streamlit run app/app.py
```
Open browser at: `http://localhost:8501`

### 6. View MLflow Dashboard
```bash
mlflow ui
```
Open browser at: `http://localhost:5000`

### 7. Run Multi-Model Comparison
```bash
python phone_pipeline_hpt.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
mlflow
optuna
optuna-integration[mlflow]
joblib
jupyter
ipykernel
```

Install all:
```bash
pip install -r requirements.txt
```

---

##  Streamlit App

The web app accepts 19 user inputs, automatically computes 5 engineered features, and returns a predicted addiction level with a color-coded result:

| Score | Level | Indicator |
|---|---|---|
| 0 – 3 | Low Addiction |  Green |
| 3 – 6 | Moderate Addiction | Orange |
| 6 – 10 | High Addiction |  Red |

---

## 📈 Results

| Metric | Score |
|---|---|
| Train R² | ~0.94 |
| Test R² | ~0.81 |
| MAE | ~0.31 |
| RMSE | ~0.47 |

---

## 💡 Key Learnings

- Data quality matters more than model complexity
- Feature engineering often beats hyperparameter tuning
- Always compare multiple models before choosing one
- MLOps is not optional — it is what makes ML production-ready
- A Pipeline prevents data leakage — always use one
- Logging and monitoring are as important as model accuracy

---

##  Future Improvements

- Add deep learning model (Neural Network / LSTM)
- Build REST API using FastAPI for mobile integration
- Set up automated model retraining pipeline
- Add real-time data collection through the app
- Implement A/B testing between model versions
- Add user authentication and prediction history

  ##  Author: **Rahul Singh**
## 🔗 Project Resources

I believe projects become more valuable when others can explore, learn from, and interact with them. Here are the project resources:

🎥 **Demo Video**
Watch the complete walkthrough of the project, including model training, MLflow experiment tracking, and the Streamlit application.
👉 [[Demo Video Link](https://drive.google.com/file/d/1cLmNCzmt0lAVg14e66RemaLQ13bzLmzZ/view?usp=sharing)]

💻 **GitHub Repository**
Explore the complete source code, project structure, notebooks, MLflow experiments, and deployment files.
👉 [[GitHub Repository Link](https://github.com/rs472m-ops/Phone_Addiction/new/main?filename=README.md)]

🌐 **Live Streamlit Application**
Try the deployed application and predict smartphone addiction levels in real time.
👉 [[Streamlit App Link](https://phoneaddiction-bbcympmufarboke7ssfwli.streamlit.app/)]

I would love to hear your feedback, suggestions, or ideas for improving the project. Feel free to connect and share your thoughts!

## 🏢 Organization

**Innomatics Research Labs**
Machine Learning Project — 4-Sprint Methodology
