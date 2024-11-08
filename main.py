import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from groq import Groq

# Groq setup (use your actual API key)
client = Groq(api_key="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")

# Load dataset
data = pd.read_csv('symbipredict_2022.csv')  # Replace with actual path
symptom_list = data.columns[:-1].tolist()   # List of symptoms from dataset

# Initialize a language model for sentence embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Compact BERT model

# Create mappings of symptoms to possible phrases
symptom_mapping = {
    "itching": ["tickling", "prickling", "tingling", "crawling", "irritation", "burning", "stinging", "scratchiness", "annoyance", "pins and needles"],
    "skin_rash": ["hives", "redness", "irritation", "eruption", "inflammation", "bumps", "blisters", "welts", "dermatitis", "lesions"],
    "nodal_skin_eruptions": ["nodules", "lumps", "swellings", "bumps", "raised lesions", "papules", "knots", "nodes", "cysts", "masses"],
    "continuous_sneezing": ["recurrent sneezing", "frequent sneezing", "persistent sneezing"],
    "shivering": ["chills", "trembling", "shaking", "quivering", "convulsing"],
    "chills": ["shivering", "coldness", "goosebumps", "chattering teeth"],
    "joint_pain": ["arthralgia", "joint discomfort", "musculoskeletal pain", "joint stiffness"],
    "stomach_pain": ["abdominal pain", "bellyache", "cramps", "stomach cramps", "gastric discomfort"],
    "acidity": ["heartburn", "acid reflux", "dyspepsia", "indigestion"],
    "ulcers_on_tongue": ["tongue sores", "oral ulcers", "canker sores", "blisters on tongue"],
    "muscle_wasting": ["atrophy", "muscle loss", "muscular degeneration", "muscle shrinkage"],
    "vomiting": ["throwing up", "retching", "emesis", "puking"],
    "burning_micturition": ["painful urination", "stinging urination", "burning sensation during urination"],
    "spotting_urination": ["blood in urine", "hematuria", "urinary spotting"],
    "fatigue": ["exhaustion", "tiredness", "weakness", "weariness", "lack of energy"],
    "weight_gain": ["weight increase", "fat accumulation", "gain in body mass", "putting on weight"],
    "anxiety": ["nervousness", "unease", "apprehension", "restlessness", "worry"],
    "cold_hands_and_feets": ["cold extremities", "chilled hands and feet", "frostbite sensation"],
    "mood_swings": ["emotional fluctuations", "mood changes", "emotional instability"],
    "weight_loss": ["unintentional weight loss", "losing weight", "reduction in body mass"],
    "restlessness": ["unease", "nervous energy", "agitation", "fidgeting", "uneasiness"],
    "lethargy": ["fatigue", "tiredness", "weakness", "lack of energy", "sluggishness"],
    "patches_in_throat": ["throat lesions", "sores in the throat", "oral patches", "throat ulcers"],
    "irregular_sugar_level": ["blood sugar imbalance", "unstable blood sugar", "fluctuating glucose levels"],
    "cough": ["persistent cough", "dry cough", "wet cough", "hack", "throat clearing"],
    "high_fever": ["fever", "elevated temperature", "high body temperature", "pyrexia"],
    "sunken_eyes": ["hollow eyes", "dark circles", "sunken gaze", "tired eyes"],
    "breathlessness": ["shortness of breath", "dyspnea", "labored breathing", "difficulty in breathing"],
    "sweating": ["perspiration", "sweaty palms", "excessive sweating", "sweaty skin"],
    "dehydration": ["fluid loss", "dryness", "thirst", "lack of hydration"],
    "indigestion": ["dyspepsia", "stomach upset", "gastritis", "bloating", "acid reflux"],
    "headache": ["migraine", "head pain", "cephalalgia", "tension headache", "cluster headache"],
    "yellowish_skin": ["jaundice", "yellow skin tint", "pale yellow skin", "yellowing of skin"],
    "dark_urine": ["colored urine", "brown urine", "amber urine", "dark-colored urine"],
    "nausea": ["sick feeling", "queasiness", "stomach upset", "vomiting sensation", "feeling nauseous"],
    "loss_of_appetite": ["anorexia", "lack of hunger", "reduced appetite", "no desire to eat"],
    "pain_behind_the_eyes": ["ocular pain", "eye discomfort", "sinus pain", "headache behind the eyes"],
    "back_pain": ["backache", "spinal pain", "lumbar pain", "lower back pain"],
    "constipation": ["difficulty in passing stools", "hard stools", "bowel irregularity", "infrequent bowel movements"],
    "abdominal_pain": ["stomach ache", "cramping", "gastric pain", "belly discomfort"],
    "diarrhoea": ["loose stools", "watery stools", "frequent bowel movements", "runny stool"],
    "mild_fever": ["low-grade fever", "slight fever", "temperature increase", "slightly elevated body temperature"],
    "yellow_urine": ["bilirubinuria", "yellowish urine", "amber-colored urine"],
    "yellowing_of_eyes": ["jaundice", "yellow eyes", "icterus"],
    "acute_liver_failure": ["liver failure", "hepatic failure", "severe liver damage"],
    "fluid_overload": ["edema", "fluid retention", "swelling", "water retention"],
    "swelling_of_stomach": ["abdominal bloating", "distended stomach", "gassy abdomen", "swollen belly"],
    "swelled_lymph_nodes": ["lymphadenopathy", "swollen glands", "enlarged lymph nodes"],
    "malaise": ["discomfort", "general ill feeling", "weakness", "unwell sensation"],
    "blurred_and_distorted_vision": ["visual impairment", "double vision", "unclear vision"],
    "phlegm": ["mucus", "sputum", "spit", "expectorated material"],
    "throat_irritation": ["sore throat", "itchy throat", "scratchy throat", "throat discomfort"],
    "redness_of_eyes": ["conjunctival redness", "bloodshot eyes", "inflamed eyes"],
    "sinus_pressure": ["sinus congestion", "sinus discomfort", "nasal pressure"],
    "runny_nose": ["nasal discharge", "rhinorrhea", "stuffy nose", "mucous secretion"],
    "congestion": ["nasal congestion", "blocked nose", "stuffiness"],
    "chest_pain": ["angina", "chest discomfort", "thoracic pain", "heart pain"],
    "weakness_in_limbs": ["limb weakness", "muscle weakness", "weak arms and legs", "reduced strength in limbs"],
    "fast_heart_rate": ["tachycardia", "rapid heartbeat", "increased heart rate"],
    "pain_during_bowel_movements": ["rectal pain", "anus pain", "painful stools"],
    "pain_in_anal_region": ["anal discomfort", "rectal pain", "anus soreness"],
    "bloody_stool": ["rectal bleeding", "blood in stools", "bloody feces"],
    "irritation_in_anus": ["anus discomfort", "rectal itching", "anal irritation"],
    "neck_pain": ["cervical pain", "neck ache", "stiff neck"],
    "dizziness": ["lightheadedness", "vertigo", "spinning sensation", "unsteadiness"],
    "cramps": ["muscle cramps", "abdominal cramps", "painful contractions", "stomach cramps"],
    "bruising": ["contusions", "bruises", "hematoma", "discoloration of skin"],
    "obesity": ["overweight", "excess weight", "obesity disorder", "being overweight"],
    "swollen_legs": ["edema in legs", "leg swelling", "puffy legs", "water retention in legs"],
    "swollen_blood_vessels": ["varicose veins", "vascular swelling", "enlarged veins"],
    "puffy_face_and_eyes": ["facial swelling", "swollen eyes", "puffy eyes", "edematous face"],
    "enlarged_thyroid": ["goiter", "thyroid enlargement", "swelling in neck", "thyroid mass"],
    "brittle_nails": ["fragile nails", "weak nails", "cracking nails", "nail brittleness"],
    "swollen_extremeties": ["swollen arms and legs", "edema in limbs", "puffy extremities"],
    "excessive_hunger": ["hyperphagia", "increased appetite", "voracious appetite"],
    "extra_marital_contacts": ["infidelity", "affairs", "cheating", "extramarital relationship"],
    "drying_and_tingling_lips": ["dry lips", "chapped lips", "tingling sensation on lips"],
    "slurred_speech": ["dysarthria", "impaired speech", "unclear speech", "mumbling"],
    "knee_pain": ["patellar pain", "pain in knee", "joint pain in knee"],
    "hip_joint_pain": ["hip pain", "pain in hip joint", "pelvic pain"],
    "muscle_weakness": ["muscle atrophy", "muscle fatigue", "loss of muscle strength"],
    "stiff_neck": ["neck rigidity", "tight neck", "difficulty moving neck"],
    "swelling_joints": ["joint swelling", "inflammation of joints", "arthritic swelling"],
    "movement_stiffness": ["rigidity", "stiffness in movement", "limited mobility"],
    "spinning_movements": ["vertigo", "dizziness", "spinning sensation"],
    "loss_of_balance": ["imbalance", "unsteadiness", "coordination loss"],
    "unsteadiness": ["imbalance", "dizziness", "lack of stability"],
    "weakness_of_one_body_side": ["hemiparesis", "one-sided weakness", "unilateral weakness"],
    "loss_of_smell": ["anosmia", "lack of smell", "inability to smell"],
    "bladder_discomfort": ["painful urination", "urinary discomfort", "bladder irritation"],
    "foul_smell_of_urine": ["strong-smelling urine", "odor in urine", "bad-smelling urine"],
    "continuous_feel_of_urine": ["urinary urgency", "constant need to urinate", "frequent urination"],
    "passage_of_gases": ["flatulence", "gas", "passing wind", "intestinal gas"],
    "internal_itching": ["pruritus", "internal scratchiness", "itching inside"],
    "toxic_look_(typhos)": ["toxicity", "ill appearance", "sickly look"],
    "depression": ["sadness", "mood disorder", "feeling down", "low mood"],
    "irritability": ["annoyance", "frustration", "impatience", "mood swings"],
    "muscle_pain": ["myalgia", "muscle soreness", "muscle ache"],
    "altered_sensorium": ["confusion", "altered consciousness", "disorientation"],
    "red_spots_over_body": ["petechiae", "skin lesions", "reddish marks"],
    "belly_pain": ["abdominal pain", "stomach pain", "cramps"],
    "abnormal_menstruation": ["irregular periods", "menstrual disturbances", "heavy periods"],
    "dischromic_patches": ["pigmented patches", "skin discoloration", "hyperpigmented areas"],
    "watering_from_eyes": ["watery eyes", "teary eyes", "eye discharge"],
    "increased_appetite": ["hyperphagia", "excessive hunger", "increased craving"],
    "polyuria": ["frequent urination", "excessive urination", "increased urine output"],
    "family_history": ["genetic predisposition", "hereditary condition", "family medical history"],
    "mucoid_sputum": ["mucous sputum", "phlegm", "thick mucus"],
    "rusty_sputum": ["brown sputum", "discolored phlegm", "bloody mucus"],
    "lack_of_concentration": ["inattention", "poor focus", "difficulty concentrating"],
    "visual_disturbances": ["blurry vision", "double vision", "vision problems"],
    "receiving_blood_transfusion": ["blood transfusion", "transfusion history"],
    "receiving_unsterile_injections": ["unsterile injections", "non-sterile shots"],
    "coma": ["unconsciousness", "vegetative state", "lack of responsiveness"],
    "stomach_bleeding": ["gastric bleeding", "internal bleeding", "hemorrhaging in stomach"],
    "distention_of_abdomen": ["abdominal bloating", "swollen stomach", "enlarged abdomen"],
    "history_of_alcohol_consumption": ["alcohol use history", "drinking history", "past alcohol consumption"],
    "blood_in_sputum": ["hemoptysis", "bloody mucus", "blood-tinged phlegm"],
    "prominent_veins_on_calf": ["varicose veins", "swollen veins", "enlarged veins in calf"],
    "palpitations": ["heart palpitations", "racing heart", "fluttering heart"],
    "painful_walking": ["painful ambulation", "difficulty walking", "pain in legs while walking"],
    "pus_filled_pimples": ["infected pimples", "pus-filled boils", "acne with pus"],
    "blackheads": ["comedones", "blocked pores", "clogged pores"],
    "scurring": ["scratching", "rubbing", "skin irritation"],
    "skin_peeling": ["exfoliation", "skin shedding", "flaky skin"],
    "silver_like_dusting": ["silver-colored dusting", "scaly patches", "flaky skin"],
    "small_dents_in_nails": ["nail pitting", "indentations in nails", "dents in nails"],
    "inflammatory_nails": ["red nails", "swollen nails", "inflamed nails"],
    "blister": ["buboes", "vesicle", "skin blister", "fluid-filled bump"],
    "red_sore_around_nose": ["nasal ulcer", "sore around nose", "painful redness around nose"],
    "yellow_crust_ooze": ["crusty discharge", "yellow discharge", "oozing yellow fluid"]
}

# Generate embeddings for symptoms and phrases for similarity comparison
symptom_embeddings = {symptom: embedding_model.encode(symptom) for symptom in symptom_list}

# Function to extract symptoms based on similarity
def extract_symptoms(user_input):
    # Use LLaMA3 via Groq to remove unwanted phrases and extract relevant keywords
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that filters out irrelevant phrases and provides only relevant symptom keywords from the user's input."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        model="llama3-8b-8192",  # Using LLaMA3 model
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )

    # Get filtered symptoms from LLaMA3 output
    relevant_keywords = chat_completion.choices[0].message.content.strip()
    
    print(relevant_keywords)
    
    # Convert the keywords into a vector format
    symptom_vector = np.zeros(len(symptom_list), dtype=int)
    
    for idx, symptom in enumerate(symptom_list):
        # Check if the symptom is in the filtered keywords
        if symptom in relevant_keywords.lower():
            symptom_vector[idx] = 1
    
    return symptom_vector

# Prepare the dataset
label_encoder = LabelEncoder()
data['prognosis'] = label_encoder.fit_transform(data['prognosis'])
X = data[symptom_list]
y = data['prognosis']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Model accuracy: {accuracy:.2f}%")

# Function to make predictions
def predict_disease(user_input):
    symptom_vector = extract_symptoms(user_input)
    predicted_disease_code = model.predict([symptom_vector])[0]
    predicted_disease = label_encoder.inverse_transform([predicted_disease_code])[0]
    return predicted_disease

# Example usage
user_input = str(input("Enter your Symptoms --> "))
predicted_disease = predict_disease(user_input)
print(f"Predicted Disease: {predicted_disease}")
