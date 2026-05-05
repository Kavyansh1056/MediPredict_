# MediPredict v7 — AI Symptom Checker

## Quick Start (VS Code / Windows)

### 1. Open the project
Open the `MediPredict_v7` folder in VS Code.

### 2. Create a virtual environment
```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Run the app (model is already trained — no need to re-train)
```
python app.py
```

### 5. Open in browser
http://localhost:5000

---

## Re-train the model (optional)
Only needed if you want to retrain from scratch:
```
python train_model.py
```

## Project Structure
```
MediPredict_v7/
├── app.py                   Flask web app
├── train_model.py           ML training pipeline
├── setup_model.py           Alternative training script
├── requirements.txt         Python dependencies
├── model/                   Trained model files (pre-included)
│   ├── model.pkl            RF+SVM Ensemble model
│   ├── metadata.json        Accuracy and CV scores
│   └── ...
├── dataset/
│   └── kaggle_symptoms_disease.csv
├── static/
│   ├── style.css
│   └── script.js
└── templates/
    ├── index.html
    └── result.html
```

## Requirements
- Python 3.9+
- Flask, scikit-learn, pandas, numpy (see requirements.txt)

## Model Performance
- Algorithm: RF + SVM Soft-Voting Ensemble
- CV Accuracy: 94.5% ± 0.5%
- Macro F1: 95.1%
- Diseases: 41

**For educational use only. Not for clinical diagnosis.**
