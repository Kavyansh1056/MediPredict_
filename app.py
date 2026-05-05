"""MediPredict v7 — Flask App (RF+SVM Ensemble, upgraded pipeline)"""
from flask import Flask, render_template, request, jsonify
import pickle, numpy as np, os, json

app = Flask(__name__)
 
# ── Load model once at startup (not per request) ──────────────────────────────
_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

def _load(filename):
    path = os.path.join(_MODEL_DIR, filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def _load_json(filename):
    path = os.path.join(_MODEL_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

# Model and feature columns (from upgraded train_model.py)
model = _load('model.pkl')
if model is None:
    raise FileNotFoundError("Run 'python train_model.py' first to train the model.")

# Feature columns: prefer new feature_cols.pkl, fall back to symptoms_list.pkl
_FEATURE_COLS = _load('feature_cols.pkl') or _load('symptoms_list.pkl') or []

# Symptom bridge (UI checkboxes → Kaggle feature vector)
_BRIDGE = _load('symptom_bridge.pkl')
_DISEASE_MAP = _load('disease_name_map.pkl') or {}
_OUR_TO_KAGGLE   = _BRIDGE['our_to_kaggle']   if _BRIDGE else {}
_KAGGLE_SYMPTOMS = _BRIDGE['kaggle_symptoms']  if _BRIDGE else _FEATURE_COLS
_USE_BRIDGE = bool(_BRIDGE)

# Model metadata (CV scores, accuracy, etc.)
MODEL_META = _load_json('metadata.json')

def _build_input_vector(selected_symptoms):
    """Convert UI symptom list → model feature vector (132-dim Kaggle space)."""
    if _USE_BRIDGE:
        kaggle_set = set()
        for s in selected_symptoms:
            if s in _OUR_TO_KAGGLE:
                kaggle_set.update(_OUR_TO_KAGGLE[s])
        return np.array([[1 if k in kaggle_set else 0 for k in _KAGGLE_SYMPTOMS]])
    return np.array([[1 if s in selected_symptoms else 0 for s in _FEATURE_COLS]])

def _map_disease_name(kaggle_name):
    """Kaggle disease label → our DISEASE_INFO key."""
    return _DISEASE_MAP.get(kaggle_name, kaggle_name)

SYMPTOM_LABELS = {
    'fever':'Fever','high_fever':'High Fever','cough':'Cough','dry_cough':'Dry Cough',
    'runny_nose':'Runny Nose','sore_throat':'Sore Throat','sneezing':'Sneezing',
    'headache':'Headache','body_ache':'Body Ache','fatigue':'Fatigue','weakness':'Weakness',
    'dizziness':'Dizziness','nausea':'Nausea','vomiting':'Vomiting','diarrhea':'Diarrhea',
    'abdominal_pain':'Abdominal Pain','chest_pain':'Chest Pain',
    'breathing_difficulty':'Breathing Difficulty','joint_pain':'Joint Pain','rash':'Skin Rash',
    'chills':'Chills','loss_of_taste':'Loss of Taste/Smell','frequent_urination':'Frequent Urination',
    'excessive_thirst':'Excessive Thirst','pale_skin':'Pale Skin','sweating':'Excessive Sweating',
    'wheezing':'Wheezing','blurred_vision':'Blurred Vision','weight_loss':'Weight Loss',
    'night_sweats':'Night Sweats','blood_in_cough':'Blood in Cough','jaundice':'Jaundice',
    'itching':'Itching','skin_blisters':'Skin Blisters','back_pain':'Back Pain',
    'neck_stiffness':'Neck Stiffness','confusion':'Confusion','memory_loss':'Memory Loss',
    'tremors':'Tremors','seizures':'Seizures','muscle_cramps':'Muscle Cramps',
    'swollen_lymph_nodes':'Swollen Lymph Nodes','swollen_joints':'Swollen Joints',
    'loss_of_appetite':'Loss of Appetite','dark_urine':'Dark Urine','constipation':'Constipation',
    'heartburn':'Heartburn','eye_redness':'Eye Redness','ear_pain':'Ear Pain',
    'nasal_congestion':'Nasal Congestion','mouth_sores':'Mouth Sores','hair_loss':'Hair Loss',
    'dry_skin':'Dry Skin','palpitations':'Heart Palpitations','leg_swelling':'Leg Swelling',
    'numbness':'Numbness','sensitivity_to_cold':'Sensitivity to Cold',
    'anxiety_feeling':'Anxiety / Panic','sadness':'Persistent Sadness',
    'sleep_problems':'Sleep Problems','irritability':'Irritability','eye_discharge':'Eye Discharge',
}

SYMPTOM_CATEGORIES = {
    '🌡️ Fever & Temperature': ['fever','high_fever','chills','sweating','night_sweats'],
    '🫁 Respiratory':         ['cough','dry_cough','blood_in_cough','runny_nose','sore_throat','sneezing','breathing_difficulty','wheezing','nasal_congestion'],
    '🤕 Head & Neuro':        ['headache','dizziness','neck_stiffness','confusion','memory_loss','tremors','seizures','numbness','blurred_vision'],
    '🤢 Digestive':           ['nausea','vomiting','diarrhea','constipation','abdominal_pain','heartburn','jaundice','dark_urine','loss_of_appetite'],
    '💓 Heart & Circulation': ['chest_pain','palpitations','leg_swelling','pale_skin'],
    '🦴 Bone, Muscle & Joint':['joint_pain','swollen_joints','body_ache','back_pain','muscle_cramps'],
    '🧴 Skin':                ['rash','itching','skin_blisters','dry_skin','hair_loss'],
    '👁️ Eye & Ear':           ['eye_redness','eye_discharge','ear_pain'],
    '💧 Urinary':             ['frequent_urination','excessive_thirst'],
    '⚡ General & Mental':    ['fatigue','weakness','weight_loss','loss_of_taste','swollen_lymph_nodes','mouth_sores','sensitivity_to_cold','anxiety_feeling','sadness','sleep_problems','irritability'],
}

# doctor_visit: "home" | "soon" | "urgent" | "emergency"
DISEASE_INFO = {
    'Common Cold': {
        'emoji':'🤧','sev':'Mild','sc':'mild','cat':'Infectious',
        'desc':'Viral upper respiratory infection. Usually resolves in 7–10 days without treatment.',
        'causes':['Rhinovirus (most common)','Adenovirus or coronavirus','Contact with infected person or surfaces'],
        'precautions':['Wash hands frequently for 20 seconds','Avoid touching eyes, nose, mouth','Cover coughs and sneezes; stay home if unwell'],
        'risk_level':'Low',
        'recommendation':'Rest, stay hydrated, and use saline nasal spray. See a doctor if symptoms worsen beyond 10 days.',
        'home_remedies':[
            'Drink warm fluids — ginger tea, honey-lemon water, or warm broth',
            'Gargle with warm salt water (½ tsp salt in 1 glass warm water) 3–4 times/day',
            'Steam inhalation with a few drops of eucalyptus oil for congestion',
            'Get plenty of rest — your immune system heals while you sleep',
            'Eat light foods like dal-rice, khichdi; avoid cold drinks and oily food',
            'Tulsi (holy basil) and ginger kadha boosts immunity naturally',
        ],
        'doctor_visit':'home',
        'doctor_note':'See a doctor if fever exceeds 39°C, symptoms last beyond 10 days, or breathing becomes difficult.',
    },
    'Influenza (Flu)': {
        'emoji':'🤒','sev':'Moderate','sc':'moderate','cat':'Infectious',
        'desc':'Contagious respiratory illness with sudden onset of high fever, body ache, and fatigue.',
        'home_remedies':[
            'Rest completely — avoid work/school for at least 5 days',
            'Stay very well hydrated with ORS, coconut water, or warm soups',
            'Take paracetamol (Crocin/Dolo) for fever and body pain as directed',
            'Steam inhalation 2–3 times daily to ease congestion',
            'Ginger-tulsi-black pepper kadha twice daily',
            'Keep warm, avoid cold air and fans directly on body',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor within 24–48 hours — antivirals are most effective early. Go immediately if breathing is affected.',
    },
    'COVID-19': {
        'emoji':'😷','sev':'Moderate–Severe','sc':'moderate','cat':'Infectious',
        'desc':'Respiratory illness caused by SARS-CoV-2. Ranges from mild to life-threatening.',
        'home_remedies':[
            'Isolate yourself immediately to protect others',
            'Monitor oxygen levels with a pulse oximeter — below 94% needs urgent care',
            'Drink warm fluids, do steam inhalation 2–3 times daily',
            'Paracetamol for fever — avoid ibuprofen unless prescribed',
            'Sleep on your stomach (prone position) to help breathing',
            'Vitamin C, Zinc, and Vitamin D supplements may support recovery',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'Consult a doctor immediately. Go to emergency if SpO2 drops below 94%, breathing is difficult, or lips turn blue.',
    },
    'Tuberculosis (TB)': {
        'emoji':'🫁','sev':'Severe','sc':'severe','cat':'Infectious',
        'desc':'Serious bacterial lung infection caused by Mycobacterium tuberculosis. Requires 6-month antibiotic course.',
        'causes':['Airborne bacterial transmission','Close contact with an active TB patient','Weakened immune system'],
        'precautions':['Wear a surgical mask in crowded areas','Ensure proper ventilation at home','Complete the full course of treatment without breaks'],
        'risk_level':'High',
        'recommendation':'Seek medical attention immediately. TB is treatable but requires doctor-supervised antibiotic therapy (DOTS program).',
        'home_remedies':[
            'There is no home cure for TB — medical treatment is mandatory',
            'Eat a high-protein, nutritious diet (eggs, milk, dal, fish) to support recovery',
            'Take prescribed medicines without missing a single dose (DOTS therapy)',
            'Get plenty of sunlight and fresh air — avoid crowded, closed spaces',
            'Avoid smoking and alcohol completely',
            'Wear a mask and cover your mouth while coughing to prevent spreading',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'See a doctor immediately. TB requires 6+ months of prescription antibiotics — home remedies alone will not work.',
    },
    'Malaria': {
        'emoji':'🦟','sev':'Severe','sc':'severe','cat':'Infectious',
        'desc':'Life-threatening parasitic disease spread by mosquito bites.',
        'causes':['Bite from infected Anopheles mosquito','Travel to endemic regions','Weakened immune system'],
        'precautions':['Sleep under mosquito nets','Use DEET repellent','Take prophylactics when travelling'],
        'risk_level':'High',
        'recommendation':'Go to a clinic immediately for a blood smear test. Malaria needs prescription antimalarials — do not delay.',
        'home_remedies':[
            'Home remedies cannot cure malaria — see a doctor first',
            'While awaiting treatment: stay very hydrated with ORS or coconut water',
            'Take paracetamol (NOT aspirin/ibuprofen) for fever',
            'Use cold, wet cloths on forehead during high fever',
            'Rest completely — do not strain the body',
            'Use mosquito nets and repellents to prevent further bites',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'Go to a doctor TODAY. Malaria can become fatal within 24–48 hours without proper antimalarial drugs.',
    },
    'Dengue Fever': {
        'emoji':'🦟','sev':'Severe','sc':'severe','cat':'Infectious',
        'desc':'Mosquito-borne viral fever caused by dengue virus, spread by Aedes mosquitoes. Severe form can cause dangerous platelet drops.',
        'causes':['Bite from infected Aedes aegypti mosquito','Travel to tropical/subtropical regions','Second dengue infection (higher severity risk)'],
        'precautions':['Eliminate standing water near home','Wear full-sleeve clothes during day','Monitor platelet count if diagnosed'],
        'risk_level':'High',
        'recommendation':'See a doctor immediately for NS1 antigen test. Monitor hydration and platelets — severe dengue needs hospitalisation.',
        'home_remedies':[
            'Stay extremely hydrated — ORS, coconut water, fresh fruit juices, soups',
            'Take ONLY paracetamol for fever — strictly avoid aspirin and ibuprofen',
            'Papaya leaf extract (fresh juice) may help increase platelet count',
            'Rest completely — no strenuous activity',
            'Eat easily digestible food — khichdi, bananas, boiled vegetables',
            'Monitor for warning signs: bleeding gums, blood in stool, severe abdominal pain',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'See a doctor immediately for platelet count monitoring. Hospitalisation may be needed. Watch for bleeding signs.',
    },
    'Typhoid': {
        'emoji':'🦠','sev':'Severe','sc':'severe','cat':'Infectious',
        'desc':'Bacterial infection spread through contaminated food and water.',
        'home_remedies':[
            'Strictly eat only boiled/properly cooked food — no outside/street food',
            'Drink only boiled or filtered water',
            'Eat light, easily digestible foods — khichdi, curd, boiled rice, banana',
            'Stay well hydrated — ORS is very helpful',
            'Complete bed rest — do not strain yourself',
            'Avoid dairy products, fried/spicy food during illness',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'See a doctor immediately. Typhoid requires prescription antibiotics. Delaying treatment can lead to complications.',
    },
    'Cholera': {
        'emoji':'💧','sev':'Severe','sc':'severe','cat':'Infectious',
        'desc':'Acute diarrheal infection causing severe dehydration. Can be fatal within hours.',
        'home_remedies':[
            'Start ORS (Oral Rehydration Solution) immediately — drink constantly',
            'Make home ORS: 1L boiled water + 6 tsp sugar + ½ tsp salt',
            'Drink coconut water, rice water (maad), or diluted fruit juice',
            'Eat nothing until vomiting stops, then eat bland foods like rice/khichdi',
            'Keep the patient very warm and comfortable',
            'Maintain strict hygiene — wash hands frequently',
        ],
        'doctor_visit':'emergency',
        'doctor_note':'Go to hospital immediately. Severe dehydration from cholera can be fatal within hours without IV fluids.',
    },
    'Chickenpox': {
        'emoji':'🔴','sev':'Mild–Moderate','sc':'mild','cat':'Infectious',
        'desc':'Highly contagious viral infection with itchy blister-like rash.',
        'home_remedies':[
            'Apply calamine lotion on blisters to relieve itching',
            'Add neem leaves or baking soda to lukewarm bath water',
            'Trim fingernails short to prevent scratching and infection',
            'Wear loose, soft cotton clothes to avoid irritation',
            'Eat soft, bland foods — avoid salty and spicy food',
            'Neem paste applied gently on blisters has traditional antiviral properties',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor to confirm diagnosis. Seek urgent care if blisters get infected, fever spikes, or breathing is affected.',
    },
    'Measles': {
        'emoji':'🔴','sev':'Moderate','sc':'moderate','cat':'Infectious',
        'desc':'Highly contagious viral disease preventable by MMR vaccination.',
        'home_remedies':[
            'Rest in a dark, quiet room — eyes may be very sensitive to light',
            'Stay well hydrated with water, coconut water, warm soups',
            'Take paracetamol for fever control',
            'Eat soft, nutritious foods — rice, dal, fruits',
            'Vitamin A supplementation is recommended — consult doctor',
            'Isolate the patient to prevent spreading to others',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'See a doctor for proper care and to rule out pneumonia/encephalitis complications. Measles is a notifiable disease.',
    },
    'Mumps': {
        'emoji':'😮','sev':'Moderate','sc':'moderate','cat':'Infectious',
        'desc':'Viral infection swelling the salivary glands.',
        'home_remedies':[
            'Apply warm or cold compress on swollen jaw/cheeks for pain relief',
            'Eat soft foods — mashed foods, yoghurt, soup; avoid chewing hard food',
            'Stay well hydrated with water and warm fluids',
            'Paracetamol for pain and fever relief',
            'Rest and avoid strenuous activity',
            'Isolate for 5 days after swelling begins to prevent spread',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor to confirm diagnosis. Urgent care if severe headache, stiff neck, or testicular pain develops.',
    },
    'Meningitis': {
        'emoji':'🧠','sev':'Life-Threatening','sc':'severe','cat':'Infectious',
        'desc':'Inflammation of membranes around the brain. Can be fatal within hours.',
        'home_remedies':[
            'There are NO safe home remedies for meningitis',
            'Call an ambulance or go to emergency immediately',
            'Do not give food or water if consciousness is reduced',
            'Keep the person lying down, comfortable, and calm',
            'Note time symptoms started — this is critical for doctors',
        ],
        'doctor_visit':'emergency',
        'doctor_note':'EMERGENCY — Call ambulance immediately. Bacterial meningitis can cause death or permanent disability within hours.',
    },
    'Hepatitis A': {
        'emoji':'🟡','sev':'Moderate','sc':'moderate','cat':'Infectious',
        'desc':'Liver inflammation from Hepatitis A virus. Spread via contaminated food/water.',
        'home_remedies':[
            'Rest completely — avoid all physical exertion',
            'Stay very well hydrated with water, coconut water, fresh juices',
            'Eat small, frequent, easily digestible meals',
            'Strictly avoid alcohol — even small amounts seriously damage the liver',
            'Avoid fatty, fried, and spicy foods',
            'Eat more fruits, vegetables, and light carbohydrates',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor within 1–2 days for liver enzyme tests and monitoring. Most cases resolve in 2 months.',
    },
    'Hepatitis B': {
        'emoji':'🟡','sev':'Moderate–Severe','sc':'moderate','cat':'Infectious',
        'desc':'Viral liver infection that can become chronic and lead to liver failure.',
        'home_remedies':[
            'Complete bed rest during acute phase',
            'Stay very well hydrated; drink 8–10 glasses of water daily',
            'Eat a liver-friendly diet: fruits, vegetables, whole grains',
            'Strictly avoid alcohol and smoking — permanent liver damage risk',
            'Avoid paracetamol in large doses — hard on the liver',
            'Take all prescribed medications without fail',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'Consult a gastroenterologist/hepatologist promptly. Chronic Hep B needs antiviral treatment to prevent cirrhosis.',
    },
    'Leptospirosis': {
        'emoji':'🐀','sev':'Moderate–Severe','sc':'moderate','cat':'Infectious',
        'desc':'Bacterial infection from contact with water contaminated by animal urine.',
        'home_remedies':[
            'Stay well hydrated with clean water and ORS',
            'Paracetamol for fever and body pain',
            'Rest completely; avoid strenuous activity',
            'Avoid wading through floodwater or dirty water',
            'Eat nutritious, easily digestible food',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'See a doctor urgently for antibiotics (doxycycline/penicillin). Severe cases need hospitalisation.',
    },
    'Pneumonia': {
        'emoji':'🫁','sev':'Severe','sc':'severe','cat':'Respiratory',
        'desc':'Lung infection causing air sacs to fill with fluid. Can be life-threatening.',
        'causes':['Streptococcus pneumoniae bacteria','Influenza or COVID-19 virus','Weakened immunity or very young/old age'],
        'precautions':['Take the full antibiotic course','Rest in upright position to ease breathing','Monitor SpO2 — seek help if below 95%'],
        'risk_level':'High',
        'recommendation':'See a doctor today for a chest X-ray. Bacterial pneumonia responds well to prompt antibiotic treatment.',
        'home_remedies':[
            'Rest completely — do not push through tiredness',
            'Stay very well hydrated to thin mucus secretions',
            'Take prescribed antibiotics on time — do NOT stop midway',
            'Steam inhalation to ease breathing (add eucalyptus oil)',
            'Sleep with head slightly elevated on pillows',
            'Warm soups and broths provide nutrition and hydration',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'See a doctor today. Pneumonia requires prescription antibiotics. Hospitalisation needed if oxygen levels drop.',
    },
    'Asthma': {
        'emoji':'💨','sev':'Moderate','sc':'moderate','cat':'Respiratory',
        'desc':'Chronic airway inflammation causing recurrent breathing episodes.',
        'causes':['Allergens: pollen, dust, pet dander','Air pollution, smoke, or chemical fumes','Exercise or cold air trigger'],
        'precautions':['Always carry your rescue inhaler','Avoid smoke, dust, and strong perfumes','Use air purifier in sleeping areas'],
        'risk_level':'Medium',
        'recommendation':'See a pulmonologist for spirometry testing. Use controller inhalers daily — not just during attacks.',
        'home_remedies':[
            'Use your prescribed reliever inhaler (Salbutamol) at first sign of attack',
            'Identify and avoid triggers: dust, pollen, smoke, cold air, pets',
            'Breathing exercises (Pranayama) like Anulom-Vilom daily help long-term',
            'Keep indoor air clean — use air purifier, avoid incense/agarbatti',
            'Ginger tea and honey may soothe airways mildly',
            'Stay away from cigarette smoke completely',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor for proper asthma management plan. Go to emergency if reliever inhaler does not work within 20 minutes.',
    },
    'Bronchitis': {
        'emoji':'🫁','sev':'Mild–Moderate','sc':'mild','cat':'Respiratory',
        'desc':'Inflammation of bronchial tubes. Acute bronchitis usually resolves in 2–3 weeks.',
        'home_remedies':[
            'Drink warm honey-ginger-lemon tea 3–4 times daily',
            'Steam inhalation with eucalyptus oil twice daily',
            'Gargle with warm salt water for throat irritation',
            'Use a humidifier in the room to keep air moist',
            'Avoid smoking, dust, and pollution completely',
            'Turmeric milk (haldi doodh) before bedtime can reduce inflammation',
        ],
        'doctor_visit':'home',
        'doctor_note':'See a doctor if cough lasts more than 3 weeks, you have a high fever, or yellow/green mucus persists.',
    },
    'Sinusitis': {
        'emoji':'👃','sev':'Mild','sc':'mild','cat':'Respiratory',
        'desc':'Inflammation of sinus cavities, usually following a cold or allergies.',
        'home_remedies':[
            'Steam inhalation 3–4 times daily (add eucalyptus or peppermint oil)',
            'Neti pot nasal rinse with warm saline water clears blockage very effectively',
            'Warm compress on face (forehead, nose, cheeks) relieves pressure',
            'Stay well hydrated to thin mucus',
            'Elevate your head while sleeping',
            'Avoid cold drinks, dairy products, and cold air during illness',
        ],
        'doctor_visit':'home',
        'doctor_note':'See a doctor if symptoms last more than 10 days, pain is severe, or fever develops.',
    },
    'Allergic Rhinitis': {
        'emoji':'🤧','sev':'Mild','sc':'mild','cat':'Respiratory',
        'desc':'Nasal allergy triggered by pollen, dust, pet dander, etc.',
        'home_remedies':[
            'Use a HEPA air purifier indoors to reduce allergens',
            'Regular neti pot nasal rinse washes away allergens',
            'Keep windows closed during high pollen seasons',
            'Steam inhalation provides temporary relief',
            'Honey (local variety) taken daily may reduce pollen sensitivity over time',
            'Wash bedsheets weekly in hot water to eliminate dust mites',
        ],
        'doctor_visit':'home',
        'doctor_note':'See a doctor if OTC antihistamines do not help, or for allergy testing and long-term immunotherapy.',
    },
    'Sleep Apnea': {
        'emoji':'😴','sev':'Moderate','sc':'moderate','cat':'Respiratory',
        'desc':'Breathing repeatedly stops during sleep. Often undiagnosed for years.',
        'home_remedies':[
            'Sleep on your side instead of your back',
            'Lose weight if overweight — even 10% reduction significantly helps',
            'Avoid alcohol and sedatives before bedtime',
            'Maintain a consistent sleep schedule',
            'Elevate the head of your bed by 4–6 inches',
            'Humming, singing, or playing wind instruments tones throat muscles',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor for a sleep study. CPAP therapy is the gold standard treatment and dramatically improves quality of life.',
    },
    'Gastroenteritis': {
        'emoji':'🤢','sev':'Mild–Moderate','sc':'mild','cat':'Digestive',
        'desc':'Stomach flu — viral or bacterial infection causing vomiting and diarrhea.',
        'home_remedies':[
            'ORS (Oral Rehydration Solution) every time you vomit or have diarrhea',
            'Eat BRAT diet: Bananas, Rice (plain), Applesauce/Apples, Toast',
            'Jeera (cumin) water: boil 1 tsp cumin in water, sip slowly',
            'Ginger tea settles nausea and stomach cramps',
            'Curd/yoghurt (once vomiting stops) restores good gut bacteria',
            'Avoid dairy, fatty foods, caffeine, and spicy food until fully recovered',
        ],
        'doctor_visit':'home',
        'doctor_note':'See a doctor if symptoms last more than 3 days, blood appears in stool/vomit, or dehydration signs appear (dry mouth, no urine).',
    },
    'Appendicitis': {
        'emoji':'🔴','sev':'Emergency','sc':'severe','cat':'Digestive',
        'desc':'Inflammation of the appendix. Can rupture and be life-threatening.',
        'home_remedies':[
            'DO NOT attempt to treat at home',
            'DO NOT take pain medication — it can mask symptoms from doctors',
            'DO NOT apply heat to the abdomen',
            'Do not eat or drink anything — surgery may be needed',
            'Go to emergency room immediately',
        ],
        'doctor_visit':'emergency',
        'doctor_note':'EMERGENCY — Go to hospital immediately. Surgery is the only treatment. Do NOT delay.',
    },
    'Irritable Bowel Syndrome': {
        'emoji':'🫃','sev':'Chronic','sc':'mild','cat':'Digestive',
        'desc':'Chronic large intestine disorder with pain, bloating, and altered bowel habits.',
        'home_remedies':[
            'Follow a low-FODMAP diet — avoid onion, garlic, wheat, legumes, dairy',
            'Eat smaller, more frequent meals instead of large ones',
            'Peppermint tea or peppermint oil capsules reduce cramping',
            'Increase soluble fibre: oats, bananas, psyllium husk (Isabgol)',
            'Stress management — yoga, meditation, and deep breathing help significantly',
            'Keep a food diary to identify your personal trigger foods',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a gastroenterologist to rule out other conditions (IBD, celiac). IBS is manageable long-term with guidance.',
    },
    'Acid Reflux (GERD)': {
        'emoji':'🔥','sev':'Mild–Moderate','sc':'mild','cat':'Digestive',
        'desc':'Stomach acid frequently flows back into the esophagus, causing heartburn.',
        'home_remedies':[
            'Eat smaller meals and sit upright for 30 minutes after eating',
            'Do not lie down within 2–3 hours of meals',
            'Elevate the head end of your bed by 6–8 inches',
            'Cold milk or banana provides instant heartburn relief',
            'Avoid tea, coffee, spicy/oily food, carbonated drinks, and chocolate',
            'Aloe vera juice (1–2 tbsp before meals) soothes esophagus lining',
        ],
        'doctor_visit':'home',
        'doctor_note':'See a doctor if heartburn is severe, frequent (more than twice a week), or you have difficulty swallowing.',
    },
    'Peptic Ulcer': {
        'emoji':'🔴','sev':'Moderate','sc':'moderate','cat':'Digestive',
        'desc':'Sores on the stomach lining or small intestine. Caused by H. pylori or NSAIDs.',
        'home_remedies':[
            'Eat small, frequent meals; do not remain on an empty stomach',
            'Cold milk and banana provide temporary relief from burning pain',
            'Cabbage juice has natural compounds that may help heal ulcers',
            'Avoid NSAIDs (aspirin, ibuprofen) completely',
            'Strictly avoid alcohol, spicy/fried food, and carbonated drinks',
            'Manage stress — stress increases acid secretion and worsens ulcers',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor for testing and antibiotics if H. pylori is the cause. Seek emergency care if you vomit blood.',
    },
    'Liver Cirrhosis': {
        'emoji':'🟡','sev':'Severe','sc':'severe','cat':'Digestive',
        'desc':'Severe, irreversible liver scarring from long-term damage.',
        'home_remedies':[
            'Absolute zero alcohol — even one drink causes further damage',
            'Low-sodium diet to reduce fluid buildup (ascites)',
            'Eat small, frequent meals high in protein (eggs, fish, dal)',
            'Stay well hydrated with water and fresh juices (not packaged)',
            'Avoid paracetamol and all NSAIDs — use only prescribed medications',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'Consult a hepatologist urgently for monitoring and management. Advanced cases may need liver transplant evaluation.',
    },
    'Hypertension': {
        'emoji':'💓','sev':'Chronic','sc':'moderate','cat':'Heart & Blood',
        'desc':'Persistently high blood pressure. Major risk factor for heart attack and stroke.',
        'causes':['High sodium diet and obesity','Chronic stress and sedentary lifestyle','Genetic predisposition and smoking'],
        'precautions':['Limit salt to under 5g/day','Exercise 30+ minutes most days','Monitor blood pressure regularly at home'],
        'risk_level':'Medium',
        'recommendation':'See a doctor for BP measurement and cardiovascular assessment. Lifestyle changes plus medication are very effective.',
        'home_remedies':[
            'DASH diet: reduce salt intake to less than 5g/day; eat more fruits and vegetables',
            'Daily 30-minute brisk walk or light exercise significantly lowers BP',
            'Practice meditation and deep breathing (Pranayama) to reduce stress',
            'Garlic and hibiscus tea have mild BP-lowering properties',
            'Lose excess weight — every 1 kg lost reduces BP by 1 mmHg',
            'Quit smoking and reduce alcohol completely',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor to start monitoring and treatment. Never stop prescribed BP medication without doctor approval.',
    },
    'Anemia': {
        'emoji':'🩸','sev':'Mild–Moderate','sc':'mild','cat':'Heart & Blood',
        'desc':'Insufficient red blood cells to carry oxygen. Iron deficiency is most common.',
        'home_remedies':[
            'Eat iron-rich foods: spinach, beetroot, pomegranate, dates, jaggery, lentils',
            'Eat Vitamin C with iron-rich foods (lemon juice on dal) to enhance absorption',
            'Avoid tea/coffee with meals — tannins block iron absorption',
            'Cook in cast iron pans — increases iron content of food',
            'Amla (Indian gooseberry) juice daily provides Vitamin C and iron',
            'Sesame seeds (til) mixed with jaggery is an excellent home remedy',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor for a blood test to identify the type of anaemia. Iron supplements may be needed.',
    },
    'Heart Attack': {
        'emoji':'❤️','sev':'Emergency','sc':'severe','cat':'Heart & Blood',
        'desc':'Blockage of blood flow to the heart muscle. Every minute of delay causes damage.',
        'causes':['Coronary artery blockage by cholesterol plaque','Atherosclerosis — artery hardening over years','Hypertension, diabetes, and smoking'],
        'precautions':['CALL EMERGENCY SERVICES IMMEDIATELY','Do not drive yourself to hospital','Chew aspirin (325mg) if not allergic while waiting'],
        'risk_level':'Critical',
        'recommendation':'CALL EMERGENCY SERVICES NOW. Every minute counts. Angioplasty within 90 minutes dramatically improves survival.',
        'home_remedies':[
            'There is NO home treatment for a heart attack',
            'Call 108 (ambulance) or rush to hospital IMMEDIATELY',
            'Chew (do not swallow whole) one 325mg aspirin if not allergic',
            'Loosen any tight clothing around chest and neck',
            'Make the person sit or lie in a comfortable position',
            'Begin CPR if the person becomes unconscious and stops breathing',
        ],
        'doctor_visit':'emergency',
        'doctor_note':'EMERGENCY — Call ambulance immediately. Time is critical. Every minute without treatment causes permanent heart damage.',
    },
    'Heart Failure': {
        'emoji':'💔','sev':'Severe','sc':'severe','cat':'Heart & Blood',
        'desc':'Heart cannot pump blood efficiently. Requires long-term management.',
        'home_remedies':[
            'Strictly limit sodium (salt) to less than 2g per day',
            'Weigh yourself daily — sudden weight gain (1–2 kg overnight) means fluid retention',
            'Monitor fluid intake as directed by your doctor',
            'Elevate legs when sitting to reduce leg swelling',
            'Light walking as tolerated — do not overexert',
            'Take all prescribed medications without missing doses',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'Consult a cardiologist urgently. Go to emergency if breathlessness worsens suddenly or legs swell rapidly.',
    },
    'Deep Vein Thrombosis': {
        'emoji':'🦵','sev':'Moderate–Severe','sc':'moderate','cat':'Heart & Blood',
        'desc':'Blood clot in a deep vein (usually leg). Can travel to lungs causing death.',
        'home_remedies':[
            'DO NOT massage the leg — this can dislodge the clot',
            'Elevate the affected leg above heart level',
            'Stay hydrated — dehydration thickens blood',
            'Gentle walking (if advised) improves blood flow',
            'Wear graduated compression stockings as recommended',
            'Avoid sitting still for long periods — move every hour',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'See a doctor immediately for anticoagulant (blood thinner) treatment. Go to emergency if you develop chest pain or breathlessness.',
    },
    'Migraine': {
        'emoji':'🧠','sev':'Moderate','sc':'moderate','cat':'Neurological',
        'desc':'Intense, debilitating headaches often with nausea and light sensitivity.',
        'causes':['Stress, hormonal changes, or sleep disruption','Dietary triggers: caffeine, chocolate, aged cheese','Bright lights, loud sounds, or strong smells'],
        'precautions':['Keep a migraine diary to identify triggers','Maintain consistent sleep and meal schedule','Rest in a quiet, dark room at onset'],
        'risk_level':'Medium',
        'recommendation':'See a neurologist if migraines occur more than 4 times a month. Preventive medication reduces frequency significantly.',
        'home_remedies':[
            'Lie down in a dark, quiet room immediately at the onset',
            'Apply a cold pack or ice wrapped in cloth on the forehead/neck',
            'Peppermint oil applied to temples provides quick relief for many people',
            'Ginger tea reduces nausea and may shorten migraine duration',
            'Stay well hydrated — dehydration is a major trigger',
            'Maintain regular sleep, meal times, and reduce screen time',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a neurologist for recurrent migraines. Seek emergency care if headache is the "worst of your life" or comes with fever/stiff neck.',
    },
    'Epilepsy': {
        'emoji':'⚡','sev':'Moderate–Severe','sc':'moderate','cat':'Neurological',
        'desc':'Neurological disorder causing recurrent, unprovoked seizures.',
        'home_remedies':[
            'During seizure: keep the person safe — move sharp objects away',
            'Turn the person on their side to prevent choking',
            'Do NOT put anything in the mouth during a seizure',
            'Time the seizure — if it lasts more than 5 minutes, call emergency',
            'Maintain regular sleep and avoid alcohol and flashing lights (known triggers)',
            'Take prescribed anti-epileptic medications every day without fail',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'See a neurologist. Call emergency if a seizure lasts more than 5 minutes or the person does not regain consciousness.',
    },
    'Parkinson\'s Disease': {
        'emoji':'🤲','sev':'Chronic–Progressive','sc':'moderate','cat':'Neurological',
        'desc':'Progressive nervous system disorder affecting movement.',
        'home_remedies':[
            'Regular exercise (walking, swimming, tai chi) significantly slows progression',
            'Speech therapy exercises help maintain voice strength',
            'Occupational therapy helps adapt daily activities',
            'High-fibre diet and plenty of water prevent constipation (a major issue)',
            'Balance exercises reduce fall risk',
            'Take medications at the same time every day for consistent effect',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a neurologist for diagnosis and medication management. Early treatment improves long-term outcomes significantly.',
    },
    'Alzheimer\'s Disease': {
        'emoji':'🧩','sev':'Chronic–Progressive','sc':'moderate','cat':'Neurological',
        'desc':'Progressive brain disease destroying memory and cognitive function.',
        'home_remedies':[
            'Mental stimulation: puzzles, reading, learning new skills may slow decline',
            'Regular physical exercise improves blood flow to the brain',
            'Mediterranean diet (olive oil, fish, vegetables, nuts) is brain-protective',
            'Maintain social connections and regular routines',
            'Quality sleep is essential — sleep disorders worsen Alzheimer\'s',
            'Turmeric (curcumin) has shown promising anti-amyloid properties in research',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a geriatrician or neurologist for early diagnosis. Medications can slow progression if started early.',
    },
    'Stroke': {
        'emoji':'🔴','sev':'Emergency','sc':'severe','cat':'Neurological',
        'desc':'Brain cells die due to blocked or burst blood vessel. Use FAST test.',
        'home_remedies':[
            'There is NO home treatment for a stroke',
            'Remember FAST: Face drooping, Arm weakness, Speech difficulty, Time to call 108',
            'Call ambulance immediately — do not drive the person yourself',
            'Keep the person calm and lying down with head slightly elevated',
            'Do not give food, water, or any medication',
            'Note the time symptoms started — critical for clot-busting treatment',
        ],
        'doctor_visit':'emergency',
        'doctor_note':'EMERGENCY — Call ambulance immediately. Clot-busting drugs must be given within 4.5 hours. Time = Brain cells.',
    },
    'Vertigo': {
        'emoji':'🌀','sev':'Mild–Moderate','sc':'mild','cat':'Neurological',
        'desc':'Spinning sensation, usually caused by inner ear problems.',
        'home_remedies':[
            'Epley maneuver (YouTube it for your affected ear) resolves most BPPV cases',
            'Sit or lie down immediately when an episode starts to prevent falls',
            'Ginger tea reduces vertigo-associated nausea effectively',
            'Move slowly — avoid quick head movements',
            'Sleep with your head slightly elevated',
            'Stay hydrated — dehydration worsens dizziness',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See an ENT specialist or neurologist. Seek emergency care if vertigo is accompanied by hearing loss, double vision, or difficulty walking.',
    },
    'Anxiety Disorder': {
        'emoji':'😰','sev':'Moderate','sc':'moderate','cat':'Mental Health',
        'desc':'Persistent excessive worry interfering with daily activities.',
        'home_remedies':[
            'Deep breathing: 4-count inhale, 4-count hold, 6-count exhale — do 10 cycles',
            'Daily 30-minute exercise is as effective as some medications for anxiety',
            'Reduce caffeine and alcohol — both worsen anxiety significantly',
            'Journaling: write down worries to externalise them',
            'Ashwagandha (500mg) has strong clinical evidence for reducing anxiety',
            'Limit news and social media — set specific times to check, not constantly',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a mental health professional. CBT (Cognitive Behavioural Therapy) is highly effective. Medication available if needed.',
    },
    'Depression': {
        'emoji':'😔','sev':'Moderate–Severe','sc':'moderate','cat':'Mental Health',
        'desc':'Persistent sadness and loss of interest. Very common and treatable.',
        'home_remedies':[
            'Talk to someone you trust — isolation worsens depression significantly',
            'Daily sunlight exposure (20 min) boosts serotonin naturally',
            'Exercise — even a 20-minute walk shows antidepressant effects',
            'Maintain regular sleep and meal schedules',
            'Reduce alcohol — it is a depressant that worsens mood',
            'Set small, achievable daily goals to rebuild a sense of accomplishment',
        ],
        'doctor_visit':'soon',
        'doctor_note':'Please talk to a mental health professional. Therapy and/or medications work well. If you have thoughts of self-harm, seek help immediately.',
    },
    'Insomnia': {
        'emoji':'🌙','sev':'Mild–Moderate','sc':'mild','cat':'Mental Health',
        'desc':'Difficulty falling or staying asleep, affecting daily functioning.',
        'home_remedies':[
            'Keep a fixed sleep-wake time every day, even weekends',
            'No screens (phone, TV) 1 hour before bed — use blue light filter',
            'Warm turmeric milk or chamomile tea 30 minutes before bed',
            'Keep bedroom cool, dark, and quiet',
            'Ashwagandha and Brahmi supplements improve sleep quality over time',
            'Progressive muscle relaxation: tense and release each muscle group',
        ],
        'doctor_visit':'home',
        'doctor_note':'See a doctor if insomnia lasts more than 1 month, severely impacts functioning, or is accompanied by depression.',
    },
    'Bipolar Disorder': {
        'emoji':'🔄','sev':'Moderate–Severe','sc':'moderate','cat':'Mental Health',
        'desc':'Extreme mood swings between mania (highs) and depression (lows).',
        'home_remedies':[
            'Maintain a strict daily routine — sleep, meals, exercise at same times',
            'Mood diary: track mood, sleep, and activities to identify patterns/triggers',
            'Avoid alcohol and recreational drugs completely — they destabilise mood',
            'Regular moderate exercise helps regulate mood',
            'Omega-3 fatty acids (fish oil) may have mild mood-stabilising effects',
            'Build a support network of trusted family and friends',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'See a psychiatrist promptly for diagnosis and mood stabiliser medications. Do not stop medications without doctor guidance.',
    },
    'Panic Disorder': {
        'emoji':'😱','sev':'Moderate','sc':'moderate','cat':'Mental Health',
        'desc':'Recurrent unexpected panic attacks with intense physical symptoms.',
        'home_remedies':[
            'Box breathing during attack: inhale 4s → hold 4s → exhale 4s → hold 4s',
            'Grounding technique: name 5 things you see, 4 you can touch, 3 you hear',
            'Cold water on face or wrists interrupts the panic response',
            'Regular yoga and Pranayama reduces overall anxiety baseline',
            'Reduce caffeine intake — it directly triggers panic attacks',
            'Remind yourself: "This is a panic attack. It is temporary. I am safe."',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a mental health professional. CBT and breathing retraining cure most people of panic disorder effectively.',
    },
    'Eczema': {
        'emoji':'🧴','sev':'Mild–Moderate','sc':'mild','cat':'Skin',
        'desc':'Chronic inflammatory skin condition causing itchy, dry, inflamed patches.',
        'home_remedies':[
            'Moisturise 2–3 times daily with fragrance-free cream (Cetaphil, coconut oil)',
            'Apply cold, damp cloth to itchy areas — do not scratch',
            'Take short, lukewarm (not hot) showers; pat dry gently',
            'Wear only soft, 100% cotton clothing',
            'Identify and avoid triggers: soaps, perfumes, certain foods, stress',
            'Neem leaf paste or aloe vera gel applied topically soothes inflammation',
        ],
        'doctor_visit':'home',
        'doctor_note':'See a dermatologist if large areas are affected, skin gets infected (pus, increased redness), or it severely affects sleep.',
    },
    'Psoriasis': {
        'emoji':'🔴','sev':'Mild–Moderate','sc':'mild','cat':'Skin',
        'desc':'Autoimmune condition causing rapid skin cell buildup and scaly patches.',
        'home_remedies':[
            'Moisturise heavily with petroleum jelly (Vaseline) to reduce scaling',
            'Aloe vera gel applied directly reduces redness and scaling',
            'Dead Sea salt or oatmeal baths soothe irritated skin',
            'Limited, careful sun exposure can improve psoriasis plaques',
            'Reduce stress — it is a major psoriasis trigger',
            'Avoid alcohol and smoking — both worsen psoriasis significantly',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a dermatologist for treatment options. Severe or joint-affecting psoriasis needs specialist management.',
    },
    'Acne': {
        'emoji':'🫧','sev':'Mild','sc':'mild','cat':'Skin',
        'desc':'Common skin condition from clogged pores with oil and dead cells.',
        'home_remedies':[
            'Wash face twice daily with a gentle, non-comedogenic cleanser',
            'Apply diluted tea tree oil on individual spots — natural antibacterial',
            'Raw honey applied as a mask has antibacterial properties',
            'Change pillow covers every 2 days',
            'Do not touch or pop pimples — causes scarring and spreads bacteria',
            'Drink plenty of water and reduce sugar intake',
        ],
        'doctor_visit':'home',
        'doctor_note':'See a dermatologist for severe/cystic acne that does not respond to OTC treatment, or when scarring is a concern.',
    },
    'Ringworm (Fungal)': {
        'emoji':'⭕','sev':'Mild','sc':'mild','cat':'Skin',
        'desc':'Contagious fungal skin infection causing ring-shaped rash.',
        'home_remedies':[
            'Apply OTC antifungal cream (Clotrimazole/Miconazole) twice daily for 4 weeks',
            'Keep area clean and dry — fungi thrive in moist environments',
            'Apply raw garlic or tea tree oil for natural antifungal effect',
            'Wear loose, breathable cotton clothing',
            'Do not share towels, combs, or clothing',
            'Continue treatment for 2 weeks after rash disappears to prevent relapse',
        ],
        'doctor_visit':'home',
        'doctor_note':'See a doctor if the rash spreads, affects scalp or nails, or does not improve after 2 weeks of antifungal cream.',
    },
    'Urticaria (Hives)': {
        'emoji':'🔴','sev':'Mild–Moderate','sc':'mild','cat':'Skin',
        'desc':'Itchy red welts triggered by allergic reactions or other causes.',
        'home_remedies':[
            'Apply cold compress or ice pack to reduce itching and swelling',
            'Take OTC antihistamine (Cetirizine) for quick relief',
            'Wear loose, cool clothing to prevent irritation',
            'Calamine lotion application provides soothing relief',
            'Identify and strictly avoid triggers (foods, soaps, medications)',
            'Aloe vera gel applied to hives reduces inflammation and itching',
        ],
        'doctor_visit':'home',
        'doctor_note':'Call emergency if you develop throat swelling, difficulty breathing, or dizziness — this is anaphylaxis, a life-threatening emergency.',
    },
    'Scabies': {
        'emoji':'🪲','sev':'Mild','sc':'mild','cat':'Skin',
        'desc':'Contagious skin mite infestation causing intense nighttime itching.',
        'home_remedies':[
            'Apply prescription scabicide (Permethrin cream) from neck down overnight',
            'Wash all clothing, bedsheets, and towels in hot water and dry on high heat',
            'Items that cannot be washed: seal in plastic bag for 1 week',
            'All close contacts must be treated simultaneously',
            'Neem oil and clove oil have antiparasitic properties (supportive use)',
            'Calamine lotion reduces itch while awaiting treatment',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor for prescription scabicide cream. Scabies does not resolve without proper treatment.',
    },
    'Rheumatoid Arthritis': {
        'emoji':'🦴','sev':'Chronic','sc':'moderate','cat':'Bone & Joint',
        'desc':'Autoimmune disease causing joint inflammation, pain, and potential deformity.',
        'home_remedies':[
            'Warm or cold packs on affected joints provide pain relief',
            'Gentle exercise and stretching maintains joint mobility',
            'Turmeric with black pepper (curcumin) has anti-inflammatory properties',
            'Fish oil (Omega-3) reduces joint inflammation — take daily',
            'Maintain a healthy weight to reduce stress on joints',
            'Rest during flare-ups — do not push through severe pain',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'See a rheumatologist. Early treatment with DMARDs prevents permanent joint damage. Do not delay treatment.',
    },
    'Osteoporosis': {
        'emoji':'💀','sev':'Moderate','sc':'moderate','cat':'Bone & Joint',
        'desc':'Weak, brittle bones increasing fracture risk. Often silent until a fracture occurs.',
        'home_remedies':[
            'Eat calcium-rich foods: milk, paneer, yoghurt, ragi, sesame seeds, almonds',
            'Get daily sun exposure for 20–30 minutes for natural Vitamin D',
            'Weight-bearing exercises (walking, dancing) strengthen bones',
            'Avoid smoking and alcohol — both accelerate bone loss',
            'Reduce excessive caffeine consumption — leaches calcium from bones',
            'Include Vitamin K foods: leafy greens, broccoli (supports bone mineralisation)',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor for a DEXA bone density scan. Calcium and Vitamin D supplements, and possibly medications, will be recommended.',
    },
    'Gout': {
        'emoji':'🦶','sev':'Mild–Moderate','sc':'mild','cat':'Bone & Joint',
        'desc':'Uric acid crystal buildup in joints causing sudden, intense pain.',
        'home_remedies':[
            'Rest the affected joint — elevate it above heart level',
            'Apply ice pack (20 minutes on, 20 off) to reduce swelling',
            'Drink lots of water — 8–12 glasses daily to flush out uric acid',
            'Cherry juice or fresh cherries reduce uric acid levels significantly',
            'Avoid red meat, organ meats, shellfish, beer, and sugary drinks',
            'Take prescribed NSAIDs or colchicine during acute attack',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor for long-term uric acid management with Allopurinol. Repeated attacks cause permanent joint damage.',
    },
    'Sciatica': {
        'emoji':'🦵','sev':'Moderate','sc':'moderate','cat':'Bone & Joint',
        'desc':'Pain radiating from lower back through the sciatic nerve into the leg.',
        'home_remedies':[
            'Apply alternating hot and cold packs to lower back (20 min each)',
            'Gentle stretches: knee-to-chest, piriformis stretch, cat-cow yoga pose',
            'Keep moving — prolonged bed rest worsens sciatica',
            'Sleep on your side with a pillow between knees',
            'Maintain good posture while sitting — use lumbar support',
            'Ibuprofen or Diclofenac gel on lower back for pain relief',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor if pain is severe, lasts more than 6 weeks, or affects bladder/bowel control (emergency — see doctor immediately).',
    },
    'Fibromyalgia': {
        'emoji':'🩹','sev':'Chronic','sc':'moderate','cat':'Bone & Joint',
        'desc':'Widespread musculoskeletal pain with fatigue, sleep, and mood issues.',
        'home_remedies':[
            'Low-impact aerobic exercise (walking, swimming) reduces pain over time',
            'Warm baths with Epsom salt soothes muscle pain',
            'Consistent sleep schedule — poor sleep dramatically worsens fibromyalgia',
            'Mindfulness meditation reduces pain perception significantly',
            'Magnesium supplement may reduce muscle cramps and improve sleep',
            'Pace your activities — alternate rest and activity throughout the day',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a rheumatologist for diagnosis. Fibromyalgia benefits from a combination of medications, physical therapy, and CBT.',
    },
    'Conjunctivitis': {
        'emoji':'👁️','sev':'Mild','sc':'mild','cat':'Eye & ENT',
        'desc':'Inflammation of the eye\'s outer membrane. Highly contagious if viral/bacterial.',
        'home_remedies':[
            'Clean eyes with clean cotton dipped in cooled boiled water — wipe outward',
            'Cold compress on closed eyes reduces swelling and discomfort',
            'Rose water (Gulab jal) drops — natural, soothing eye cleanser',
            'Wash hands thoroughly and frequently',
            'Do not share towels, pillows, or eye cosmetics',
            'Avoid touching eyes; remove contact lenses until fully healed',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor for antibiotic eye drops if bacterial. Avoid school/work for 24 hours after starting treatment.',
    },
    'Glaucoma': {
        'emoji':'👁️','sev':'Moderate–Severe','sc':'moderate','cat':'Eye & ENT',
        'desc':'Optic nerve damage from high eye pressure. Leading cause of blindness.',
        'home_remedies':[
            'There is no home cure for glaucoma — prescribed eye drops must not be skipped',
            'Regular exercise (walking) reduces eye pressure mildly',
            'Avoid inverted yoga poses (headstands) — they raise eye pressure',
            'Do not smoke — smoking is a glaucoma risk factor',
            'Protect eyes from UV light with sunglasses outdoors',
            'Take all prescribed eye drops exactly as directed — never miss a dose',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'See an ophthalmologist immediately. Glaucoma damage is permanent and irreversible — early treatment prevents blindness.',
    },
    'Tonsillitis': {
        'emoji':'😮','sev':'Mild–Moderate','sc':'mild','cat':'Eye & ENT',
        'desc':'Inflammation of the tonsils, commonly from viral or bacterial infection.',
        'home_remedies':[
            'Gargle with warm salt water every 2 hours (very effective)',
            'Honey and ginger tea with tulsi leaves soothes throat pain',
            'Cold ice cream or ice chips can reduce inflammation and pain temporarily',
            'Stay well hydrated — warm soups, broths, warm water',
            'Turmeric mixed in warm milk is a traditional anti-inflammatory remedy',
            'Rest your voice — talking worsens throat irritation',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor to rule out Strep throat (requires antibiotics). Urgent care if throat swells enough to affect breathing or swallowing.',
    },
    'Ear Infection (Otitis)': {
        'emoji':'👂','sev':'Mild–Moderate','sc':'mild','cat':'Eye & ENT',
        'desc':'Infection of the middle ear. Very common in children but affects adults too.',
        'home_remedies':[
            'Warm (not hot) compress or heating pad against the ear for 20 minutes provides pain relief',
            'A few drops of warm (body temperature) olive oil in the ear canal may soothe',
            'Garlic has antimicrobial properties — garlic-infused oil drops are a traditional remedy',
            'Keep the ear dry — avoid swimming and getting water in the ear',
            'Elevate the head slightly while sleeping to promote drainage',
            'Paracetamol or ibuprofen for pain and fever management',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor within 1–2 days. Bacterial ear infections require antibiotic ear drops or oral antibiotics.',
    },
    'Urinary Tract Infection': {
        'emoji':'💧','sev':'Mild–Moderate','sc':'mild','cat':'Urinary',
        'desc':'Bacterial infection in the urinary system. Very common, especially in women.',
        'home_remedies':[
            'Drink 3–4 litres of water daily to flush bacteria out',
            'Unsweetened cranberry juice reduces bacterial adhesion to bladder walls',
            'Curd/yoghurt (probiotic) maintains healthy urinary tract bacteria balance',
            'Avoid holding urine — urinate when you feel the urge',
            'Warm compress on lower abdomen reduces pain and discomfort',
            'Avoid caffeine, alcohol, and spicy food during infection',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor for antibiotics. UTIs rarely clear on their own and can spread to kidneys if untreated.',
    },
    'Kidney Stones': {
        'emoji':'🪨','sev':'Moderate–Severe','sc':'moderate','cat':'Urinary',
        'desc':'Hard mineral deposits in the kidneys. Passing a stone is extremely painful.',
        'home_remedies':[
            'Drink 10–12 glasses of water daily — most essential treatment',
            'Fresh lemon juice (citrate) prevents calcium oxalate stone formation',
            'Barley water and coconut water help flush urinary tract',
            'Avoid high-oxalate foods: spinach, nuts, chocolate, beets',
            'Reduce salt (sodium) intake — increases calcium in urine',
            'Regular exercise helps prevent stone formation',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See a doctor for imaging and treatment. Emergency care if you have unbearable pain, fever with chills, or complete inability to urinate.',
    },
    'Chronic Kidney Disease': {
        'emoji':'🫘','sev':'Severe','sc':'severe','cat':'Urinary',
        'desc':'Gradual loss of kidney function. Often asymptomatic until advanced stages.',
        'home_remedies':[
            'Strictly follow a kidney-friendly diet — limit protein, potassium, phosphorus, and salt',
            'Stay well hydrated (as advised — not excessive if kidneys are very weak)',
            'Control blood pressure and blood sugar — they are the main causes of CKD',
            'Exercise regularly (light walking) to maintain cardiovascular health',
            'Avoid NSAIDs (ibuprofen, aspirin) completely — they damage kidneys',
            'Quit smoking and alcohol completely',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'Consult a nephrologist immediately for staging and treatment. CKD is progressive — early management prevents dialysis.',
    },
    'Diabetes': {
        'emoji':'🩸','sev':'Chronic','sc':'moderate','cat':'Metabolic',
        'desc':'Chronic condition affecting blood sugar regulation. Requires lifelong management.',
        'causes':['Insulin resistance (Type 2)','Autoimmune destruction of beta cells (Type 1)','Obesity and sedentary lifestyle'],
        'precautions':['Monitor blood glucose daily','Follow a low-GI diet','Exercise at least 30 minutes most days'],
        'risk_level':'Medium',
        'recommendation':'See a doctor for HbA1c and fasting glucose tests. Lifestyle changes plus medication effectively manage diabetes.',
        'home_remedies':[
            'Follow a low-GI diet: whole grains, vegetables, legumes; avoid white rice, sugar, maida',
            'Methi (fenugreek) seeds soaked overnight — drink water in the morning on empty stomach',
            'Bitter gourd (karela) juice is a traditional blood sugar reducer',
            'Amla (Indian gooseberry) juice daily has anti-diabetic properties',
            'Exercise 30 minutes daily — walking after meals reduces blood sugar significantly',
            'Manage stress — cortisol directly raises blood glucose levels',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'Consult a doctor for blood tests and medication. Home remedies support but cannot replace prescribed medications for diabetes.',
    },
    'Hypothyroidism': {
        'emoji':'🦋','sev':'Moderate','sc':'moderate','cat':'Metabolic',
        'desc':'Underactive thyroid producing insufficient hormones. Causes fatigue and weight gain.',
        'home_remedies':[
            'Eat selenium-rich foods: Brazil nuts, sunflower seeds, eggs (supports thyroid)',
            'Iodine-rich foods: sea vegetables, iodised salt, fish (in moderation)',
            'Avoid eating soy, raw cruciferous vegetables (cauliflower, cabbage) in excess',
            'Exercise regularly to boost metabolism and energy',
            'Manage stress — chronic stress disrupts thyroid function',
            'Take medications at the same time every day, on an empty stomach',
        ],
        'doctor_visit':'soon',
        'doctor_note':'See an endocrinologist. Levothyroxine replacement therapy is very effective. Regular blood TSH monitoring is needed.',
    },
    'Hyperthyroidism': {
        'emoji':'🦋','sev':'Moderate','sc':'moderate','cat':'Metabolic',
        'desc':'Overactive thyroid producing excess hormones. Can cause heart complications.',
        'home_remedies':[
            'Bugleweed and lemon balm herbal teas may mildly reduce thyroid activity',
            'Eat calcium-rich foods — hyperthyroidism leaches calcium from bones',
            'Avoid iodine-rich foods: seaweed, iodised salt, fish, dairy in excess',
            'Reduce stress through yoga, meditation — stress worsens hyperthyroidism',
            'Limit caffeine and alcohol — they worsen palpitations and anxiety',
            'Rest adequately — hyperthyroidism causes physical exhaustion',
        ],
        'doctor_visit':'urgent',
        'doctor_note':'See an endocrinologist promptly for antithyroid medications or other treatment. Untreated hyperthyroidism causes heart complications.',
    },
}
DEFAULT_INFO = {
    'emoji':'🏥','sev':'Unknown','sc':'mild','cat':'General',
    'desc':'Please consult a healthcare professional for a proper diagnosis.',
    'home_remedies':['Rest adequately','Stay well hydrated','Eat nutritious, light meals','Monitor symptoms and note any changes'],
    'doctor_visit':'soon',
    'doctor_note':'Please see a doctor for proper diagnosis and treatment guidance.',
}

DOCTOR_VISIT_CONFIG = {
    'home':      {'label':'Can Manage at Home','color':'#16a34a','bg':'#dcfce7','icon':'🏠'},
    'soon':      {'label':'See a Doctor Soon','color':'#d97706','bg':'#fef3c7','icon':'📅'},
    'urgent':    {'label':'See a Doctor Today','color':'#ea580c','bg':'#ffedd5','icon':'⚠️'},
    'emergency': {'label':'EMERGENCY — Go Now','color':'#dc2626','bg':'#fee2e2','icon':'🚨'},
}

@app.route('/')
def index():
    return render_template('index.html',
        symptom_categories=SYMPTOM_CATEGORIES,
        symptom_labels=SYMPTOM_LABELS,
        model_meta=MODEL_META)

@app.route('/predict', methods=['POST'])
def predict():
    selected = request.form.getlist('symptoms')
    if not selected:
        return render_template('result.html',
            result_mode='unclear', prediction='', confidence=0,
            top_predictions=[], info=DEFAULT_INFO, dv=DOCTOR_VISIT_CONFIG['home'],
            selected_symptoms=[], symptom_labels=SYMPTOM_LABELS, symptom_count=0,
            split_disease=None, split_info=None, model_meta=MODEL_META)

    input_vector = _build_input_vector(selected)
    kaggle_prediction = model.predict(input_vector)[0]
    probabilities = model.predict_proba(input_vector)[0]
    classes = model.classes_

    sorted_idx = np.argsort(probabilities)[::-1]
    top_indices = sorted_idx[:5]
    top_predictions = [
        {'disease': _map_disease_name(classes[i]), 'prob': round(float(probabilities[i])*100, 1)}
        for i in top_indices
    ]

    top1_prob = float(probabilities[sorted_idx[0]])
    top2_prob = float(probabilities[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0

    if top1_prob >= 0.40:
        result_mode = 'clear'
    elif top1_prob >= 0.25 and top2_prob >= 0.20 and (top1_prob - top2_prob) < 0.15:
        result_mode = 'split'
    elif top1_prob >= 0.25:
        result_mode = 'clear'
    else:
        result_mode = 'unclear'

    prediction  = _map_disease_name(kaggle_prediction)
    confidence  = round(top1_prob * 100, 1)
    info        = DISEASE_INFO.get(prediction, DEFAULT_INFO)
    dv          = DOCTOR_VISIT_CONFIG[info.get('doctor_visit', 'soon')]
    split_disease = top_predictions[1]['disease'] if result_mode == 'split' and len(top_predictions) > 1 else None
    split_info    = DISEASE_INFO.get(split_disease, DEFAULT_INFO) if split_disease else None

    return render_template('result.html',
        prediction=prediction, confidence=confidence,
        top_predictions=top_predictions,
        info=info, dv=dv,
        selected_symptoms=selected,
        symptom_labels=SYMPTOM_LABELS,
        symptom_count=len(selected),
        result_mode=result_mode,
        split_disease=split_disease, split_info=split_info,
        model_meta=MODEL_META)

@app.route('/api/model-info')
def model_info():
    return jsonify(MODEL_META)

if __name__ == '__main__':
    meta_acc = MODEL_META.get('cv_mean', 0)
    print('  MediPredict v7')
    print('  Model: ' + MODEL_META.get('model_type','Ensemble') + ' | CV ' + str(round(meta_acc*100,1)) + '%')
    print('  Open: http://localhost:5000')
    app.run(debug=True)
