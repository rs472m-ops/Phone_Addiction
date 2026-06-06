import streamlit as st
import joblib
import pandas as pd

# Load model
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📱 Phone Addiction Level Predictor")
st.write("Enter your details below:")

# Inputs
age                    = st.number_input("Age",                          min_value=10,   value=18)
gender                 = st.selectbox("Gender",                          ["Male", "Female", "Other"])
daily_usage_hours      = st.number_input("Daily Usage Hours",            min_value=0.0,  value=5.0)
sleep_hours            = st.number_input("Sleep Hours",                  min_value=0.0,  value=7.0)
intellectual_perf      = st.number_input("Intellectual Performance",     min_value=0,    value=70)
social_interactions    = st.number_input("Social Interactions",          min_value=0,    value=5)
exercise_hours         = st.number_input("Exercise Hours",               min_value=0.0,  value=0.5)
anxiety_level          = st.number_input("Anxiety Level (0-10)",         min_value=0,    value=5)
depression_level       = st.number_input("Depression Level (0-10)",      min_value=0,    value=5)
self_esteem            = st.number_input("Self Esteem (0-10)",           min_value=0,    value=5)
screen_time_before_bed = st.number_input("Screen Time Before Bed (hrs)", min_value=0.0,  value=1.0)
phone_checks_per_day   = st.number_input("Phone Checks Per Day",         min_value=0,    value=50)
apps_used_daily        = st.number_input("Apps Used Daily",              min_value=0,    value=10)
time_on_social_media   = st.number_input("Time on Social Media",         min_value=0.0,  value=2.0)
time_on_gaming         = st.number_input("Time on Gaming",               min_value=0.0,  value=1.0)
time_on_education      = st.number_input("Time on Education",            min_value=0.0,  value=1.0)
phone_usage_purpose    = st.selectbox("Phone Usage Purpose",             ["Browsing", "Social Media", "Gaming", "Education", "Work"])
family_communication   = st.number_input("Family Communication (hrs)",   min_value=0,    value=3)
weekend_usage_hours    = st.number_input("Weekend Usage Hours",          min_value=0.0,  value=8.0)

# Encode categoricals (same as training)
gender_encoded  = {"Male": 1, "Female": 0, "Other": 2}[gender]
purpose_encoded = {"Browsing": 0, "Social Media": 3, "Gaming": 1, "Education": 2, "Work": 4}[phone_usage_purpose]

if st.button("Predict"):

    # Feature engineering — must match Sprint 3 exactly
    usage_sleep_ratio     = daily_usage_hours / (sleep_hours + 1e-5)
    total_screen_exposure = daily_usage_hours + weekend_usage_hours
    total_content_time    = time_on_social_media + time_on_gaming + time_on_education
    high_usage_flag       = 1 if daily_usage_hours > 5.0 else 0
    stress_score          = anxiety_level + depression_level

    input_data = pd.DataFrame({
        "Age":                       [age],
        "Gender":                    [gender_encoded],
        "Daily_Usage_Hours":         [daily_usage_hours],
        "Sleep_Hours":               [sleep_hours],
        "Interllectual_Performance": [intellectual_perf],
        "Social_Interactions":       [social_interactions],
        "Exercise_Hours":            [exercise_hours],
        "Anxiety_Level":             [anxiety_level],
        "Depression_Level":          [depression_level],
        "Self_Esteem":               [self_esteem],
        "Screen_Time_Before_Bed":    [screen_time_before_bed],
        "Phone_Checks_Per_Day":      [phone_checks_per_day],
        "Apps_Used_Daily":           [apps_used_daily],
        "Time_on_Social_Media":      [time_on_social_media],
        "Time_on_Gaming":            [time_on_gaming],
        "Time_on_Education":         [time_on_education],
        "Phone_Usage_Purpose":       [purpose_encoded],
        "Family_Communication":      [family_communication],
        "Weekend_Usage_Hours":       [weekend_usage_hours],
        "Usage_Sleep_Ratio":         [usage_sleep_ratio],
        "Total_Screen_Exposure":     [total_screen_exposure],
        "Total_Content_Time":        [total_content_time],
        "High_Usage_Flag":           [high_usage_flag],
        "Stress_Score":              [stress_score],
    })

    prediction = round(model.predict(input_data)[0], 2)

    if prediction <= 3:
        st.success(f"Predicted Addiction Level: {prediction} / 10 — Low Addiction ✅")
    elif prediction <= 6:
        st.warning(f"Predicted Addiction Level: {prediction} / 10 — Moderate Addiction ⚠️")
    else:
        st.error(f"Predicted Addiction Level: {prediction} / 10 — High Addiction 🚨")
