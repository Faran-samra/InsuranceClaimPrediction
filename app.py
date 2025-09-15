import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Insurance Claim Classifier", layout="centered")

st.title("Insurance Claim Prediction")
st.write("Fill the form and click Predict. (Model trained WITHOUT charges)")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sex", options=[0,1], format_func=lambda x: "Female (0)" if x==0 else "Male (1)")
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
steps = st.number_input("Avg steps/day", min_value=0, max_value=50000, value=5000)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", options=[0,1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
region = st.selectbox("Region", options=[0,1,2,3], format_func=lambda x: ["NE(0)","NW(1)","SE(2)","SW(3)"][x])

if st.button("Predict"):
    input_dict = {'age':age,'sex':sex,'bmi':bmi,'steps':steps,'children':children,'smoker':smoker,'region':region}
    model = joblib.load("models/xgb_no_charges_joblib.pkl")
    df_input = pd.DataFrame([input_dict])
    df_input['bmi_category'] = pd.cut(df_input['bmi'], bins=[0,18.5,25,30,100], labels=['underweight','normal','overweight','obese'])
    df_input['age_bucket'] = pd.cut(df_input['age'], bins=[0,25,40,60,120], labels=['young','adult','midage','senior'])
    pred = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[:,1][0] if hasattr(model, "predict_proba") else None
    st.write("**Prediction (1 = will claim, 0 = won't claim):**", int(pred))
    if proba is not None:
        st.write(f"Prediction probability for class=1: {proba:.3f}")
