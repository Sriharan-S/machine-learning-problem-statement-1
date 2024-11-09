from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from groq import Groq

svm = joblib.load(r'C:\Users\Sriharan\Desktop\Hack Hustle\svm_model.joblib')
data = pd.read_csv('symbipredict_2022.csv')
symptom_list = data.columns[:-1].tolist()

client1 = Groq(api_key="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")
client2 = Groq(api_key="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")
client3 = Groq(api_key="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")
client4 = Groq(api_key="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")
client5 = Groq(api_key="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")
label_encoder = LabelEncoder()
data['prognosis'] = label_encoder.fit_transform(data['prognosis'])

app = Flask(__name__)
CORS(app)

def extract_symptoms(user_input):
    chat_completion = client1.chat.completions.create(
        messages=[
            {"role": "system", "content": """you are a helpful assistant. These were list of symptoms:
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", 
    "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting", 
    "burning_micturition", "spotting_urination", "fatigue", "weight_gain", "anxiety", 
    "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", "lethargy", 
    "patches_in_throat", "irregular_sugar_level", "cough", "high_fever", "sunken_eyes", 
    "breathlessness", "sweating", "dehydration", "indigestion", "headache", "yellowish_skin", 
    "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", 
    "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", "yellowing_of_eyes", 
    "acute_liver_failure", "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", 
    "malaise", "blurred_and_distorted_vision", "phlegm", "throat_irritation", "redness_of_eyes", 
    "sinus_pressure", "runny_nose", "congestion", "chest_pain", "weakness_in_limbs", 
    "fast_heart_rate", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", 
    "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising", "obesity", 
    "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid", 
    "brittle_nails", "swollen_extremeties", "excessive_hunger", "extra_marital_contacts", 
    "drying_and_tingling_lips", "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness", 
    "stiff_neck", "swelling_joints", "movement_stiffness", "spinning_movements", "loss_of_balance", 
    "unsteadiness", "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", 
    "foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases", "internal_itching", 
    "toxic_look_(typhos)", "depression", "irritability", "muscle_pain", "altered_sensorium", 
    "red_spots_over_body", "belly_pain", "abnormal_menstruation", "dischromic_patches", 
    "watering_from_eyes", "increased_appetite", "polyuria", "family_history", "mucoid_sputum", 
    "rusty_sputum", "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion", 
    "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen", 
    "history_of_alcohol_consumption", "fluid_overload", "blood_in_sputum", "prominent_veins_on_calf", 
    "palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurring", 
    "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister", 
    "red_sore_around_nose", "yellow_crust_ooze". You need to analyze the user input and return the symptoms from above which matches the given Input. The Output should be exact as given (including underscore). Format the each symptom in newline, do not include any other extra formatting. Just reply with symptoms, do not add any header and footer texts. If no symptoms matched, just reply with 0."""},
            {"role": "user", "content": f"'{user_input}'"}
        ],
        model="llama3-70b-8192",
        temperature=0.5,
        max_tokens=100,
        top_p=1,
        stop=None,
        stream=False,
    )
    relevant_keywords = chat_completion.choices[0].message.content.strip()
    return relevant_keywords

def generate_detail(relevant_keywords):
    chat_completion = client2.chat.completions.create(
        messages=[
            {"role": "system", "content": """you are a virtual Doctor. You should generate a 50 word response for the given symptom(s). Provide a disease description for them. Do not provide solution. Do not attach anything else apart from actual content."""},
            {"role": "user", "content": f"'{relevant_keywords}'"}
        ],
        model="llama3-70b-8192",
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        stop=None,
        stream=False,
    )
    brief = chat_completion.choices[0].message.content.strip()
    return brief

def generate_solution(relevant_keywords):
    chat_completion = client3.chat.completions.create(
        messages=[
            {"role": "system", "content": """you are a virtual Doctor. You should generate a 50 word natural solution for the given symptom(s). Do not provide description about the symptom. You should provide solution. Also insist to consult the doctor. Do not attach anything else apart from actual content."""},
            {"role": "user", "content": f"'{relevant_keywords}'"}
        ],
        model="llama3-70b-8192",
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        stop=None,
        stream=False,
    )
    solution = chat_completion.choices[0].message.content.strip()
    return solution

def generate_duration(relevant_keywords):
    chat_completion = client4.chat.completions.create(
        messages=[
            {"role": "system", "content": """you are a virtual Doctor. You should just give the time required to cure the given symptom. Do not provide solution. Do not give description. Just the Time requiered to cure. Like "4 Days" or "1-3 Days". Do not attach anything else apart from actual content."""},
            {"role": "user", "content": f"'{relevant_keywords}'"}
        ],
        model="llama3-70b-8192",
        temperature=0.5,
        max_tokens=50,
        top_p=1,
        stop=None,
        stream=False,
    )
    reason = chat_completion.choices[0].message.content.strip()
    return reason

def generate_reason(relevant_keywords):
    chat_completion = client5.chat.completions.create(
        messages=[
            {"role": "system", "content": """you are a virtual Doctor. You should just give the reason for the given symptom in 2-5 words. Do not provide solution. Do not give description. Just why he got the disease/symptom. Do not attach anything else apart from actual content."""},
            {"role": "user", "content": f"'{relevant_keywords}'"}
        ],
        model="llama3-70b-8192",
        temperature=0.5,
        max_tokens=20,
        top_p=1,
        stop=None,
        stream=False,
    )
    duration = chat_completion.choices[0].message.content.strip()
    return duration

def map_to_symptom_vector(relevant_keywords):
    symptom_vector = np.zeros(len(symptom_list), dtype=int)
    keywords = [kw.strip().lower() for kw in relevant_keywords.split("\n")]

    for keyword in keywords:
        if keyword in symptom_list:
            symptom_index = symptom_list.index(keyword)
            symptom_vector[symptom_index] = 1
    return symptom_vector

@app.route('/predict', methods=['POST'])
def predict_disease():
    data = request.get_json()
    user_input = data.get("symptoms", "")

    relevant_keywords = extract_symptoms(user_input)
    symptom_vector = map_to_symptom_vector(relevant_keywords)
    detail = generate_detail(relevant_keywords)
    solution = generate_solution(relevant_keywords)
    duration = generate_duration(relevant_keywords)
    reason = generate_reason(relevant_keywords)

    predicted_disease_code = svm.predict([symptom_vector])[0]
    predicted_disease = label_encoder.inverse_transform([predicted_disease_code])[0]

    response = {
        "predicted_disease": predicted_disease,
        "relevant_keywords": relevant_keywords,
        "symptom_vector": symptom_vector.tolist(),
        "detail": detail,
        "solution": solution,
        "duration": duration,
        "reason": reason
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
