import pandas as pd
import numpy as np
import os, random

random.seed(42)
np.random.seed(42)

symptoms = [
    'fever', 'high_fever', 'mild_fever', 'cough', 'runny_nose', 'sneezing',
    'sore_throat', 'fatigue', 'headache', 'severe_headache', 'body_aches',
    'joint_pain', 'chills', 'sweating', 'nausea', 'vomiting', 'diarrhea',
    'abdominal_pain', 'chest_pain', 'shortness_of_breath', 'skin_rash',
    'loss_of_taste', 'loss_of_smell', 'blurred_vision', 'frequent_urination',
    'excessive_thirst', 'burning_urination', 'cloudy_urine', 'yellowing_skin',
    'dark_urine', 'loss_of_appetite', 'pale_skin', 'weakness', 'dizziness',
    'sensitivity_to_light', 'slow_healing', 'cold_hands'
]

disease_profiles = {
    'Common Cold':      ['mild_fever', 'runny_nose', 'sneezing', 'sore_throat', 'cough', 'fatigue'],
    'Influenza':        ['high_fever', 'body_aches', 'fatigue', 'headache', 'cough', 'sore_throat', 'chills'],
    'COVID-19':         ['fever', 'cough', 'fatigue', 'loss_of_taste', 'loss_of_smell', 'shortness_of_breath', 'body_aches'],
    'Pneumonia':        ['high_fever', 'chest_pain', 'cough', 'shortness_of_breath', 'fatigue', 'chills'],
    'Bronchitis':       ['cough', 'chest_pain', 'fatigue', 'mild_fever', 'shortness_of_breath'],
    'Malaria':          ['high_fever', 'chills', 'sweating', 'headache', 'nausea', 'vomiting', 'body_aches'],
    'Dengue':           ['high_fever', 'severe_headache', 'joint_pain', 'skin_rash', 'nausea', 'fatigue'],
    'Typhoid':          ['high_fever', 'abdominal_pain', 'nausea', 'fatigue', 'headache', 'diarrhea'],
    'Gastroenteritis':  ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain', 'mild_fever', 'fatigue'],
    'Migraine':         ['severe_headache', 'nausea', 'sensitivity_to_light', 'vomiting'],
    'Diabetes':         ['frequent_urination', 'excessive_thirst', 'fatigue', 'blurred_vision', 'slow_healing'],
    'Hypertension':     ['headache', 'dizziness', 'chest_pain', 'shortness_of_breath', 'blurred_vision'],
    'Anemia':           ['fatigue', 'weakness', 'pale_skin', 'dizziness', 'shortness_of_breath', 'cold_hands'],
    'UTI':              ['burning_urination', 'frequent_urination', 'abdominal_pain', 'mild_fever', 'cloudy_urine'],
    'Jaundice':         ['yellowing_skin', 'fatigue', 'abdominal_pain', 'nausea', 'dark_urine', 'loss_of_appetite'],
}

rows = []
for disease, core_syms in disease_profiles.items():
    for _ in range(120):
        row = {s: 0 for s in symptoms}
        for s in core_syms:
            if random.random() < 0.85:
                row[s] = 1
        for s in symptoms:
            if row[s] == 0 and random.random() < 0.04:
                row[s] = 1
        row['disease'] = disease
        rows.append(row)

df = pd.DataFrame(rows)
os.makedirs('dataset', exist_ok=True)
df.to_csv('dataset/disease_dataset.csv', index=False)
print(f"✅ Dataset generated: {len(df)} rows, {len(symptoms)} symptom features")
print(df['disease'].value_counts().to_string())
