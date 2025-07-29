## ⚽ Football Performance Predictor

This project uses machine learning to predict whether a football player will perform well in upcoming matches, based on historical statistics. It supports model training, evaluation, and an interactive prediction interface via a Streamlit app.

---

### 📁 Project Structure

```
.
├── main.py                      # Trains and evaluates models
├── app.py                       # Streamlit web app for predictions
├── requirements.txt             # Python dependencies
├── output/                      # Stores trained models, features, results
├── data/raw/FPL_logs.csv        # Input dataset (not included here)
├── config/system_config.yaml    # Model hyperparameters
└── src/
    ├── models/                  # Model implementations (RF, SVM, etc.)
    ├── data/                    # Data loader & preprocessor
    ├── features/                # Feature selector
    ├── evaluation/              # Visualizations
    └── utils/                   # Logger and utilities
```

---

### ✅ Prerequisites

* Python 3.8+
* pip (or virtualenv recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 🧠 Step 1: Train the Models

To train models and generate predictions:

```bash
python main.py --config config/system_config.yaml --output_dir output --models RandomForest SVM NeuralNetwork GradientBoosting --feature_selection SelectKBest --n_features 30
```

This will:

* Preprocess the dataset
* Select features
* Train all models
* Save `.joblib` models and metrics to the `output/` folder

---

### 📊 Step 2: Launch the Web App

Start the prediction UI with:

```bash
streamlit run app.py
```

Use the sidebar to:

* Choose a trained model
* Upload a CSV or enter individual player details
* View prediction results and visualizations

---

### 🗂 Required Files

Before running the app, ensure the following files exist in the `output/` folder:

* `randomforest_model.joblib`
* `svm_model.joblib`
* `neuralnetwork_model.joblib`
* `gradientboosting_model.joblib`
* `selected_features.joblib`
* Optional: `*_feature_importance.csv` for feature charts

---

### 📌 Notes

* The dataset `FPL_logs.csv` must be placed in `data/raw/` and should follow the expected format.
* To add a custom model, extend `model_factory.py` and implement `fit`, `predict`, and `predict_proba`.

