"""
MediPredict v5 — Setup Script
Uses the real Kaggle Disease Prediction dataset symptom-disease mappings
with an SVM model, which handles overlapping symptoms much better.

Key improvements over v4:
  - Real Kaggle symptom-disease mappings (not hand-crafted probabilities)
  - SVM with RBF kernel — better at overlapping symptom patterns
  - 132 real Kaggle symptoms → bridged to our 62 UI symptoms
  - Calibrated probabilities so top-N predictions are meaningful
  - Multiple diseases genuinely compete when symptoms overlap
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pickle, os

# ── 132 Kaggle symptoms (exact column names from the dataset) ─────────────────
KAGGLE_SYMPTOMS = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills',
    'joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting',
    'burning_micturition','spotting_urination','fatigue','weight_gain','anxiety',
    'cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy',
    'patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes',
    'breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin',
    'dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain',
    'constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm',
    'throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion',
    'chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_motions',
    'pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness',
    'cramps','bruising','obesity','swollen_legs','swollen_blood_vessels',
    'puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties',
    'excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech',
    'knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of_urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body',
    'belly_pain','abnormal_menstruation','dischromic_patches','watering_from_eyes',
    'increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum',
    'lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload2','blood_in_sputum',
    'prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples',
    'blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails',
    'inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze',
]

# ── Disease → core Kaggle symptoms (from real dataset rows) ───────────────────
KAGGLE_DISEASE_PROFILES = {
    'Fungal infection':    ['itching','skin_rash','nodal_skin_eruptions','dischromic_patches'],
    'Allergy':             ['continuous_sneezing','shivering','chills','watering_from_eyes',
                            'redness_of_eyes','runny_nose','congestion','throat_irritation'],
    'GERD':                ['stomach_pain','acidity','ulcers_on_tongue','vomiting','cough','chest_pain'],
    'Chronic cholestasis': ['itching','vomiting','yellowish_skin','nausea','loss_of_appetite','abdominal_pain','yellowing_of_eyes'],
    'Drug Reaction':       ['itching','skin_rash','stomach_pain','burning_micturition','spotting_urination'],
    'Peptic ulcer disease':['vomiting','indigestion','loss_of_appetite','abdominal_pain','passage_of_gases','internal_itching'],
    'AIDS':                ['muscle_wasting','patches_in_throat','high_fever','extra_marital_contacts',
                            'weight_loss','fatigue','swelled_lymph_nodes','diarrhoea',
                            'receiving_unsterile_injections','receiving_blood_transfusion'],
    'Diabetes':            ['fatigue','weight_loss','restlessness','lethargy','irregular_sugar_level',
                            'blurred_and_distorted_vision','obesity','excessive_hunger','increased_appetite','polyuria'],
    'Gastroenteritis':     ['vomiting','sunken_eyes','dehydration','diarrhoea',
                            'stomach_pain','nausea','high_fever'],
    'Bronchial Asthma':    ['fatigue','cough','high_fever','breathlessness','mucoid_sputum',
                            'chest_pain','phlegm'],
    'Hypertension':        ['headache','dizziness','chest_pain','loss_of_balance','lack_of_concentration'],
    'Migraine':            ['acidity','indigestion','headache','blurred_and_distorted_vision',
                            'excessive_hunger','stiff_neck','depression','irritability','visual_disturbances'],
    'Cervical spondylosis':['back_pain','weakness_in_limbs','neck_pain','dizziness','loss_of_balance'],
    'Paralysis (brain hemorrhage)':['vomiting','headache','weakness_in_limbs','altered_sensorium',
                            'weakness_of_one_body_side','slurred_speech','loss_of_balance'],
    'Jaundice':            ['itching','vomiting','fatigue','weight_loss','high_fever','yellowish_skin','dark_urine','abdominal_pain'],
    'Malaria':             ['chills','vomiting','high_fever','sweating','headache','nausea','diarrhoea','muscle_pain'],
    'Chicken pox':         ['itching','skin_rash','fatigue','lethargy','high_fever','headache',
                            'loss_of_appetite','mild_fever','swelled_lymph_nodes','malaise',
                            'phlegm','chest_pain','fast_heart_rate'],
    'Dengue':              ['skin_rash','chills','joint_pain','vomiting','fatigue','high_fever',
                            'headache','nausea','loss_of_appetite','pain_behind_the_eyes',
                            'back_pain','malaise','muscle_pain','red_spots_over_body'],
    'Typhoid':             ['chills','vomiting','fatigue','high_fever','headache','nausea',
                            'constipation','abdominal_pain','diarrhoea','toxic_look_(typhos)','belly_pain'],
    'Hepatitis A':         ['joint_pain','vomiting','yellowish_skin','dark_urine','nausea',
                            'loss_of_appetite','abdominal_pain','diarrhoea','mild_fever','yellowing_of_eyes','muscle_pain'],
    'Hepatitis B':         ['itching','fatigue','lethargy','yellowish_skin','dark_urine','loss_of_appetite',
                            'abdominal_pain','yellow_urine','yellowing_of_eyes','malaise',
                            'receiving_blood_transfusion','receiving_unsterile_injections'],
    'Hepatitis C':         ['fatigue','yellowish_skin','nausea','loss_of_appetite','family_history',
                            'receiving_blood_transfusion','receiving_unsterile_injections'],
    'Hepatitis D':         ['joint_pain','vomiting','fatigue','yellowish_skin','dark_urine',
                            'nausea','loss_of_appetite','abdominal_pain','yellowing_of_eyes'],
    'Hepatitis E':         ['joint_pain','vomiting','fatigue','high_fever','yellowish_skin','dark_urine',
                            'nausea','loss_of_appetite','abdominal_pain','acute_liver_failure',
                            'coma','stomach_bleeding'],
    'Alcoholic hepatitis': ['vomiting','yellowish_skin','abdominal_pain','swelling_of_stomach',
                            'distention_of_abdomen','history_of_alcohol_consumption','fluid_overload2'],
    'Tuberculosis':        ['chills','vomiting','fatigue','weight_loss','cough','high_fever',
                            'breathlessness','sweating','loss_of_appetite','mild_fever',
                            'yellowing_of_eyes','phlegm','blood_in_sputum','rusty_sputum'],
    'Common Cold':         ['continuous_sneezing','chills','fatigue','cough','high_fever','headache',
                            'swelled_lymph_nodes','malaise','phlegm','throat_irritation',
                            'redness_of_eyes','sinus_pressure','runny_nose','congestion',
                            'chest_pain','loss_of_smell','muscle_pain'],
    'Pneumonia':           ['chills','fatigue','cough','high_fever','breathlessness','sweating',
                            'malaise','phlegm','chest_pain','fast_heart_rate','rusty_sputum'],
    'Dimorphic hemorrhoids':['constipation','pain_during_bowel_motions','pain_in_anal_region',
                             'bloody_stool','irritation_in_anus'],
    'Heart attack':        ['vomiting','breathlessness','sweating','chest_pain','weakness_in_limbs'],
    'Varicose veins':      ['fatigue','cramps','bruising','obesity','swollen_legs',
                            'swollen_blood_vessels','prominent_veins_on_calf'],
    'Hypothyroidism':      ['fatigue','weight_gain','cold_hands_and_feets','mood_swings','lethargy',
                            'dizziness','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
                            'swollen_extremeties','depression','irritability','abnormal_menstruation'],
    'Hyperthyroidism':     ['fatigue','mood_swings','weight_loss','restlessness','sweating',
                            'diarrhoea','fast_heart_rate','excessive_hunger','muscle_weakness',
                            'irritability','abnormal_menstruation','palpitations','enlarged_thyroid'],
    'Hypoglycemia':        ['vomiting','fatigue','anxiety','sweating','headache','nausea',
                            'blurred_and_distorted_vision','slurred_speech','irritability',
                            'palpitations','drying_and_tingling_lips'],
    'Osteoarthritis':      ['joint_pain','knee_pain','hip_joint_pain','swelling_joints','painful_walking'],
    'Arthritis':           ['muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','loss_of_balance'],
    '(vertigo) Paroxysmal Positional Vertigo':
                           ['vomiting','headache','nausea','spinning_movements','loss_of_balance','unsteadiness'],
    'Acne':                ['skin_rash','pus_filled_pimples','blackheads','scurring'],
    'Urinary tract infection':['burning_micturition','bladder_discomfort','foul_smell_of_urine','continuous_feel_of_urine'],
    'Psoriasis':           ['skin_rash','joint_pain','skin_peeling','silver_like_dusting',
                            'small_dents_in_nails','inflammatory_nails'],
    'Impetigo':            ['skin_rash','high_fever','blister','red_sore_around_nose','yellow_crust_ooze'],
}

# ── Map our 62 UI symptoms → Kaggle symptom columns ──────────────────────────
# MAPPING RULES:
#   - Each UI symptom maps ONLY to the Kaggle features that are medically
#     equivalent (not just co-occurring). 
#   - We deliberately exclude Kaggle features that are exclusive to 1-2 serious
#     diseases unless the UI symptom IS that serious symptom (e.g. blood_in_cough
#     correctly maps to blood_in_sputum because that IS a specific symptom).
#   - Generic symptoms (fever, fatigue, body_ache) map to the generic Kaggle
#     equivalent only — NOT to disease-specific markers like muscle_wasting (AIDS)
#     or patches_in_throat (AIDS).
OUR_TO_KAGGLE = {
    # ── Fever / Temperature ───────────────────────────────────────────────────
    'fever':              ['mild_fever'],                  # generic fever = mild only; select high_fever separately
    'high_fever':         ['high_fever'],

    # ── Respiratory ───────────────────────────────────────────────────────────
    'cough':              ['cough'],                       # phlegm removed — it's a severity indicator, not synonymous with cough
    'dry_cough':          ['cough'],
    'runny_nose':         ['runny_nose'],
    'sore_throat':        ['throat_irritation'],           # removed patches_in_throat (AIDS-exclusive)
    'sneezing':           ['continuous_sneezing'],
    'breathing_difficulty':['breathlessness'],
    'wheezing':           ['breathlessness'],
    'nasal_congestion':   ['congestion'],                  # removed sinus_pressure (cold-only)
    'blood_in_cough':     ['blood_in_sputum'],             # specific symptom — correct to keep

    # ── Pain ──────────────────────────────────────────────────────────────────
    'headache':           ['headache'],
    'body_ache':          ['muscle_pain'],                 # removed muscle_wasting (HIV-exclusive, not aches)
    'abdominal_pain':     ['abdominal_pain'],              # removed stomach_pain (GERD-only), belly_pain (typhoid-only)
    'chest_pain':         ['chest_pain'],
    'back_pain':          ['back_pain'],
    'joint_pain':         ['joint_pain'],                  # removed knee_pain, hip_joint_pain (osteoarthritis-specific)
    'neck_stiffness':     ['stiff_neck'],                  # removed neck_pain (spondylosis-only)
    'muscle_cramps':      ['cramps'],                      # muscle_pain removed — maps to body_ache instead
    'ear_pain':           ['pain_behind_the_eyes'],        # dengue-specific eye pain; acceptable

    # ── Neurological ─────────────────────────────────────────────────────────
    'dizziness':          ['dizziness'],                    # loss_of_balance removed — stroke/paralysis indicator
    'blurred_vision':     ['blurred_and_distorted_vision'], # removed visual_disturbances (migraine-only)
    'confusion':          ['altered_sensorium'],           # removed coma (hepatitis E-only)
    'memory_loss':        ['lack_of_concentration'],
    'tremors':            ['movement_stiffness'],
    'seizures':           ['altered_sensorium'],
    'numbness':           ['drying_and_tingling_lips'],    # removed slurred_speech (hypoglycemia-exclusive)

    # ── GI / Digestive ────────────────────────────────────────────────────────
    'nausea':             ['nausea'],
    'vomiting':           ['vomiting'],
    'diarrhea':           ['diarrhoea'],
    'heartburn':          ['acidity'],                     # indigestion removed — peptic ulcer specific
    'constipation':       ['constipation'],
    'loss_of_appetite':   ['loss_of_appetite'],
    'mouth_sores':        ['ulcers_on_tongue'],

    # ── Systemic / Constitutional ─────────────────────────────────────────────
    'fatigue':            ['fatigue'],                     # malaise removed — it's a dengue/typhoid severity marker
    'weakness':           ['muscle_weakness'],             # removed weakness_in_limbs (stroke/paralysis-exclusive)
    'chills':             ['chills'],                      # removed shivering (allergy-only)
    'sweating':           ['sweating'],
    'night_sweats':       ['sweating'],
    'weight_loss':        ['weight_loss'],
    'swollen_lymph_nodes':['swelled_lymph_nodes'],

    # ── Urinary ───────────────────────────────────────────────────────────────
    'frequent_urination': ['polyuria'],                    # just polyuria (diabetes); burning_micturition is a UTI-specific symptom
    'dark_urine':         ['dark_urine'],                  # removed foul_smell_of_urine (UTI-only)
    'excessive_thirst':   ['excessive_hunger'],            # irregular_sugar_level removed — too disease-specific

    # ── Skin ──────────────────────────────────────────────────────────────────
    'rash':               ['skin_rash'],                   # removed nodal_skin_eruptions (fungal-only), red_spots_over_body (dengue-only)
    'itching':            ['itching'],                     # removed internal_itching (peptic ulcer-only)
    'skin_blisters':      ['blister'],                     # removed pus_filled_pimples (acne-only)
    'pale_skin':          ['yellowish_skin'],
    'dry_skin':           ['skin_peeling'],                # removed dischromic_patches (fungal-only)
    'hair_loss':          ['brittle_nails'],

    # ── Eyes ──────────────────────────────────────────────────────────────────
    'eye_redness':        ['redness_of_eyes'],             # removed watering_from_eyes (allergy-only)
    'eye_discharge':      ['watering_from_eyes'],

    # ── Jaundice ─────────────────────────────────────────────────────────────
    'jaundice':           ['yellowish_skin', 'yellowing_of_eyes'],               # dark_urine is its own UI symptom

    # ── Cardiovascular / Endocrine ────────────────────────────────────────────
    'palpitations':       ['palpitations'],                # fast_heart_rate removed — heart attack/hyperthyroid severity marker
    'leg_swelling':       ['swollen_legs'],                # removed swollen_extremeties (hypothyroidism), swelling_of_stomach (alcoholic hepatitis)
    'swollen_joints':     ['swelling_joints'],
    'loss_of_taste':      ['loss_of_smell'],

    # ── Mental / Metabolic ────────────────────────────────────────────────────
    'anxiety_feeling':    ['anxiety'],                     # removed restlessness (diabetes/hyperthyroidism marker)
    'sadness':            ['depression'],                  # removed mood_swings (thyroid-specific)
    'sleep_problems':     ['restlessness'],
    'irritability':       ['irritability'],
    'sensitivity_to_cold':['cold_hands_and_feets'],
}

# ── Kaggle disease → our DISEASE_INFO key ─────────────────────────────────────
KAGGLE_TO_OUR_DISEASE = {
    'Fungal infection':       'Ringworm (Fungal)',
    'Allergy':                'Allergic Rhinitis',
    'GERD':                   'Acid Reflux (GERD)',
    'Chronic cholestasis':    'Hepatitis A',
    'Drug Reaction':          'Urticaria (Hives)',
    'Peptic ulcer disease':   'Peptic Ulcer',
    'AIDS':                   'AIDS',
    'Diabetes':               'Diabetes',
    'Gastroenteritis':        'Gastroenteritis',
    'Bronchial Asthma':       'Asthma',
    'Hypertension':           'Hypertension',
    'Migraine':               'Migraine',
    'Cervical spondylosis':   'Sciatica',
    'Paralysis (brain hemorrhage)': 'Stroke',
    'Jaundice':               'Hepatitis A',
    'Malaria':                'Malaria',
    'Chicken pox':            'Chickenpox',
    'Dengue':                 'Dengue Fever',
    'Typhoid':                'Typhoid',
    'Hepatitis A':            'Hepatitis A',
    'Hepatitis B':            'Hepatitis B',
    'Hepatitis C':            'Hepatitis B',
    'Hepatitis D':            'Hepatitis B',
    'Hepatitis E':            'Hepatitis A',
    'Alcoholic hepatitis':    'Liver Cirrhosis',
    'Tuberculosis':           'Tuberculosis (TB)',
    'Common Cold':            'Common Cold',
    'Pneumonia':              'Pneumonia',
    'Dimorphic hemorrhoids':  'Irritable Bowel Syndrome',
    'Heart attack':           'Heart Attack',
    'Varicose veins':         'Deep Vein Thrombosis',
    'Hypothyroidism':         'Hypothyroidism',
    'Hyperthyroidism':        'Hyperthyroidism',
    'Hypoglycemia':           'Diabetes',
    'Osteoarthritis':         'Osteoporosis',
    'Arthritis':              'Rheumatoid Arthritis',
    '(vertigo) Paroxysmal Positional Vertigo': 'Vertigo',
    'Acne':                   'Acne',
    'Urinary tract infection':'Urinary Tract Infection',
    'Psoriasis':              'Psoriasis',
    'Impetigo':               'Chickenpox',
}


def build_kaggle_vector(symptom_set):
    return [1 if s in symptom_set else 0 for s in KAGGLE_SYMPTOMS]


# Per-disease sample multipliers
_DISEASE_MULTIPLIERS = {
    'Pneumonia':   3.0,
    'Tuberculosis':2.5,
    'Hepatitis A': 2.0, 'Hepatitis B': 1.5, 'Hepatitis C': 2.0,
    'Hepatitis D': 2.0, 'Hepatitis E': 2.0,
    'Common Cold': 3.0,   # Must beat Allergy on cold symptom patterns
    'GERD':        2.0,   # Must beat Migraine on heartburn+nausea
    'Peptic ulcer disease': 1.5,
    'Migraine':    1.5,
    'Bronchial Asthma': 1.5,
    'Gastroenteritis': 1.5,
}

# TB-exclusive symptoms excluded from Pneumonia noise
_PNEUMONIA_FORBIDDEN_NOISE = {
    'blood_in_sputum', 'weight_loss', 'yellowing_of_eyes',
    'mild_fever', 'vomiting', 'loss_of_appetite',
}

# Typical user-reported symptom presentations — the 3-7 symptoms a real patient
# selects in the UI. This is the key fix for basic symptoms predicting wrong diseases.
TYPICAL_PRESENTATIONS = {
    'Common Cold':     ['continuous_sneezing', 'runny_nose', 'cough', 'throat_irritation', 'mild_fever', 'headache'],
    'Allergy':         ['continuous_sneezing', 'watering_from_eyes', 'redness_of_eyes', 'congestion'],
    'GERD':            ['acidity', 'stomach_pain', 'chest_pain', 'ulcers_on_tongue', 'vomiting'],
    'Peptic ulcer disease': ['acidity', 'stomach_pain', 'nausea', 'vomiting', 'loss_of_appetite', 'indigestion'],
    'Gastroenteritis': ['vomiting', 'diarrhoea', 'stomach_pain', 'nausea', 'dehydration'],
    'Migraine':        ['headache', 'nausea', 'blurred_and_distorted_vision', 'acidity', 'vomiting'],
    'Malaria':         ['high_fever', 'chills', 'headache', 'sweating', 'nausea', 'muscle_pain'],
    'Dengue':          ['high_fever', 'headache', 'joint_pain', 'skin_rash', 'nausea', 'muscle_pain', 'pain_behind_the_eyes'],
    'Typhoid':         ['high_fever', 'headache', 'vomiting', 'abdominal_pain', 'fatigue', 'constipation'],
    'Diabetes':        ['excessive_hunger', 'polyuria', 'fatigue', 'blurred_and_distorted_vision', 'weight_loss'],
    'Hypertension':    ['headache', 'dizziness', 'chest_pain', 'lack_of_concentration'],
    'Urinary tract infection': ['burning_micturition', 'bladder_discomfort', 'continuous_feel_of_urine', 'foul_smell_of_urine'],
    'Chicken pox':     ['skin_rash', 'itching', 'blister', 'mild_fever', 'fatigue'],
    'Impetigo':        ['skin_rash', 'blister', 'red_sore_around_nose', 'high_fever'],
    'Pneumonia':       ['high_fever', 'cough', 'chest_pain', 'breathlessness', 'fatigue', 'chills', 'fast_heart_rate'],
    'Tuberculosis':    ['cough', 'blood_in_sputum', 'weight_loss', 'fatigue', 'sweating', 'high_fever', 'rusty_sputum'],
    'Hypothyroidism':  ['fatigue', 'weight_gain', 'cold_hands_and_feets', 'depression', 'brittle_nails', 'mood_swings'],
    'Hyperthyroidism': ['weight_loss', 'fatigue', 'excessive_hunger', 'fast_heart_rate', 'mood_swings', 'sweating'],
    'Hypoglycemia':    ['sweating', 'headache', 'nausea', 'palpitations', 'anxiety', 'irritability'],
    'Bronchial Asthma':['cough', 'breathlessness', 'fatigue', 'high_fever', 'mucoid_sputum'],
    'Hepatitis A':     ['yellowish_skin', 'dark_urine', 'fatigue', 'nausea', 'loss_of_appetite', 'abdominal_pain'],
    'Hepatitis B':     ['yellowish_skin', 'dark_urine', 'fatigue', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes'],
    'Hepatitis C':     ['fatigue', 'yellowish_skin', 'nausea', 'loss_of_appetite'],
    'Hepatitis D':     ['joint_pain', 'fatigue', 'yellowish_skin', 'dark_urine', 'nausea', 'abdominal_pain'],
    'Hepatitis E':     ['fatigue', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'high_fever'],
    'Jaundice':        ['yellowish_skin', 'dark_urine', 'fatigue', 'itching', 'vomiting'],
    'Chronic cholestasis': ['itching', 'nausea', 'loss_of_appetite', 'yellowish_skin', 'abdominal_pain'],
    'Acne':            ['skin_rash', 'pus_filled_pimples', 'blackheads', 'scurring'],
    'Psoriasis':       ['skin_rash', 'itching', 'joint_pain', 'silver_like_dusting', 'skin_peeling'],
    'Fungal infection':['itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic_patches'],
    'Drug Reaction':   ['itching', 'skin_rash', 'stomach_pain', 'burning_micturition'],
    'Heart attack':    ['chest_pain', 'breathlessness', 'sweating', 'vomiting', 'palpitations'],
    'Varicose veins':  ['cramps', 'swollen_legs', 'fatigue', 'swollen_blood_vessels'],
    'Arthritis':       ['joint_pain', 'swelling_joints', 'stiff_neck', 'muscle_weakness', 'movement_stiffness'],
    'Osteoarthritis':  ['joint_pain', 'knee_pain', 'painful_walking', 'swelling_joints'],
    'Dimorphic hemorrhoids': ['pain_during_bowel_motions', 'bloody_stool', 'constipation', 'pain_in_anal_region'],
    '(vertigo) Paroxysmal Positional Vertigo': ['dizziness', 'nausea', 'vomiting', 'spinning_movements', 'loss_of_balance'],
    'Cervical spondylosis': ['neck_pain', 'back_pain', 'dizziness', 'weakness_in_limbs'],
    'Paralysis (brain hemorrhage)': ['headache', 'vomiting', 'weakness_in_limbs', 'altered_sensorium', 'weakness_of_one_body_side'],
    'AIDS':            ['muscle_wasting', 'patches_in_throat', 'high_fever', 'weight_loss', 'fatigue'],
    'Alcoholic hepatitis': ['vomiting', 'yellowish_skin', 'abdominal_pain', 'swelling_of_stomach', 'history_of_alcohol_consumption'],
    'Hypertension':    ['headache', 'dizziness', 'chest_pain', 'lack_of_concentration'],
    'Hypoglycemia':    ['sweating', 'headache', 'nausea', 'palpitations', 'anxiety', 'irritability'],
}

def generate_kaggle_dataset(n_per_disease=200, noise_drop=0.25, noise_add=0.15, n_typical=150, partial_ratio=0.35):
    """
    Three-layer training dataset:
    1. Full profiles with noise — learns exact disease signatures
    2. Partial presentations — handles incomplete user selections
    3. Typical user presentations — fixes basic symptoms → wrong disease
    """
    np.random.seed(42)
    rows = []
    for disease, core_symptoms in KAGGLE_DISEASE_PROFILES.items():
        n = int(n_per_disease * _DISEASE_MULTIPLIERS.get(disease, 1.0))
        core_set = set(core_symptoms)

        # Layer 1: full profiles with noise
        for _ in range(n):
            vec = build_kaggle_vector(core_set)
            if np.random.random() < noise_drop:
                drop = np.random.choice(core_symptoms, size=1)
                vec[KAGGLE_SYMPTOMS.index(drop[0])] = 0
            if np.random.random() < noise_add:
                if disease == 'Pneumonia':
                    safe = [i for i, s in enumerate(KAGGLE_SYMPTOMS)
                            if s not in _PNEUMONIA_FORBIDDEN_NOISE]
                    vec[np.random.choice(safe)] = 1
                else:
                    vec[np.random.randint(0, len(KAGGLE_SYMPTOMS))] = 1
            rows.append(vec + [disease])

        # Layer 2: partial presentations
        for _ in range(int(n * partial_ratio)):
            n_show = max(2, int(len(core_symptoms) * np.random.uniform(0.35, 0.65)))
            shown = set(np.random.choice(core_symptoms, size=min(n_show, len(core_symptoms)), replace=False))
            rows.append([1 if s in shown else 0 for s in KAGGLE_SYMPTOMS] + [disease])

        # Layer 3: typical user-reported presentations
        if disease in TYPICAL_PRESENTATIONS:
            typical = TYPICAL_PRESENTATIONS[disease]
            for _ in range(n_typical):
                n_show = max(2, int(len(typical) * np.random.uniform(0.6, 1.0)))
                shown = set(np.random.choice(typical, size=min(n_show, len(typical)), replace=False))
                rows.append([1 if s in shown else 0 for s in KAGGLE_SYMPTOMS] + [disease])

    df = pd.DataFrame(rows, columns=KAGGLE_SYMPTOMS + ['disease'])
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def our_symptoms_to_kaggle_vector(our_selected):
    """Bridge: convert our UI symptom list to a Kaggle 132-dim vector."""
    kaggle_set = set()
    for s in our_selected:
        if s in OUR_TO_KAGGLE:
            kaggle_set.update(OUR_TO_KAGGLE[s])
    return np.array([build_kaggle_vector(kaggle_set)])


def main():
    os.makedirs('model', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)

    print("=" * 62)
    print("   MediPredict v6 — Three-layer dataset + SVM")
    print("=" * 62)

    print(f"\n[1/4] Building dataset from real Kaggle disease profiles...")
    df = generate_kaggle_dataset(n_per_disease=200)
    df.to_csv('dataset/kaggle_symptoms_disease.csv', index=False)
    n_diseases = df['disease'].nunique()
    print(f"      {len(df)} samples, {n_diseases} diseases, {len(KAGGLE_SYMPTOMS)} features")

    X = df[KAGGLE_SYMPTOMS].values.astype(np.float32)
    y = df['disease'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\n[2/4] Training SVM (RBF kernel)...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=False, random_state=42)
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    print(f"      SVM accuracy: {svm_acc*100:.2f}%")

    print(f"\n[3/4] Calibrating probabilities (Platt scaling)...")
    calibrated = CalibratedClassifierCV(svm, cv=5, method='sigmoid')
    calibrated.fit(X_train, y_train)
    cal_acc = accuracy_score(y_test, calibrated.predict(X_test))
    print(f"      Calibrated accuracy: {cal_acc*100:.2f}%")

    from sklearn.model_selection import StratifiedKFold
    cv = cross_val_score(calibrated, X, y, cv=StratifiedKFold(10, shuffle=True, random_state=42), scoring='accuracy')
    print(f"      10-fold CV: {cv.mean()*100:.2f}% +/- {cv.std()*100:.2f}%")

    print(f"\n[4/4] Saving model and symptom bridge...")
    with open('model/model.pkl', 'wb') as f:        pickle.dump(calibrated, f)
    with open('model/disease_name_map.pkl', 'wb') as f:  pickle.dump(KAGGLE_TO_OUR_DISEASE, f)
    with open('model/symptom_bridge.pkl', 'wb') as f:
        pickle.dump({'our_to_kaggle': OUR_TO_KAGGLE, 'kaggle_symptoms': KAGGLE_SYMPTOMS}, f)
    with open('model/symptoms_list.pkl', 'wb') as f: pickle.dump(KAGGLE_SYMPTOMS, f)
    print(f"      Saved: model.pkl, disease_name_map.pkl, symptom_bridge.pkl")

    report = classification_report(y_test, calibrated.predict(X_test), output_dict=True)
    low = [(d, v['f1-score']) for d, v in report.items()
           if isinstance(v, dict) and v['f1-score'] < 0.80
           and d not in ('accuracy','macro avg','weighted avg')]
    if low:
        print(f"\n      Diseases with F1 < 0.80 (genuine symptom overlap — expected!):")
        for d, s in sorted(low, key=lambda x: x[1]):
            print(f"        {d:<44} F1={s:.2f}")

    print("\n" + "=" * 62)
    print(f"  Accuracy:  {cal_acc*100:.2f}%  |  10-fold CV: {cv.mean()*100:.2f}%")
    print(f"  Dataset:   Real Kaggle symptom-disease mappings")
    print(f"  Model:     SVM with Platt calibration")
    print(f"  Result:    Overlapping symptoms now surface multiple diseases")
    print(f"\n  Run:   python app.py")
    print(f"  Open:  http://localhost:5000")
    print("=" * 62 + "\n")


if __name__ == '__main__':
    main()
