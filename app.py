"""
app.py  —  Titanic Survival Predictor
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "titanic_model.joblib"
    if not os.path.exists(path):
        st.error("Model not found. Run `python train_model.py` first.")
        st.stop()
    return joblib.load(path)

model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚢 Titanic Survival Predictor")
st.write("Fill in the passenger details below to predict survival chances.")
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3],
                          format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} Class")
    sex    = st.radio("Sex", ["Male", "Female"], horizontal=True)
    age    = st.slider("Age", 1, 80, 28)
    embarked = st.selectbox("Port of Embarkation",
                             ["Southampton", "Cherbourg", "Queenstown"])

with col2:
    sibsp  = st.number_input("Siblings / Spouses aboard", 0, 8, 0)
    parch  = st.number_input("Parents / Children aboard", 0, 6, 0)
    fare   = st.number_input("Ticket Fare (£)", 0.0, 520.0, 32.0, step=0.5)

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict Survival", use_container_width=True, type="primary"):

    sex_enc      = 1 if sex == "Female" else 0
    embarked_enc = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]

    features = np.array([[pclass, sex_enc, age, sibsp, parch, fare, embarked_enc]])

    prediction   = model.predict(features)[0]
    probability  = model.predict_proba(features)[0]

    survived_prob    = probability[1] * 100
    not_survived_prob = probability[0] * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ **Survived** — {survived_prob:.1f}% confidence")
    else:
        st.error(f"❌ **Did Not Survive** — {not_survived_prob:.1f}% confidence")

    # Probability bar
    st.write("**Survival Probability**")
    st.progress(int(survived_prob))
    st.caption(f"Survived: {survived_prob:.1f}%  |  Did not survive: {not_survived_prob:.1f}%")

    # Summary
    st.divider()
    st.write("**Passenger Summary**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Class",  f"{pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'}")
    c2.metric("Age",    age)
    c3.metric("Sex",    sex)
    c4.metric("Fare",   f"£{fare:.0f}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: Random Forest · Accuracy: 81% · Trained on Titanic dataset (891 passengers)")
