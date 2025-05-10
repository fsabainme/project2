import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
from io import BytesIO

# --------------------------------------------------
# 1. Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Quadratic Regression Predictor",
    layout="centered",
)

# --------------------------------------------------
# 2. Load pretrained model from GitHub
# --------------------------------------------------
@st.cache_resource
def load_model_from_github() -> object:
    url = (
        "https://raw.githubusercontent.com/"
        "fsabainme/project2/main/model.pkl"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    return pickle.load(BytesIO(resp.content))

model = load_model_from_github()

# --------------------------------------------------
# 3. Load the experimental data
# --------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    raw = np.array([
        [1, 10.00,  3.27, 6.00, 54.89],
        [2, 10.00, 10.00, 9.36, 77.45],
        [3,  6.00, 14.00, 4.00, 78.39],
        [4, 14.00,  6.00, 4.00, 47.89],
        [5,  6.00, 14.00, 8.00, 68.98],
        [6, 10.00, 10.00, 6.00, 94.28],
        [7, 14.00,  6.00, 8.00, 85.68],
        [8, 14.00, 14.00, 8.00, 77.34],
        [9, 10.00, 10.00, 6.00, 98.98],
        [10, 6.00,  6.00, 8.00, 68.98],
        [11,10.00, 10.00, 2.64, 53.89],
        [12,16.73, 10.00, 6.00, 52.67],
        [13,10.00, 10.00, 6.00, 97.89],
        [14,10.00, 10.00, 6.00, 91.98],
        [15,14.00, 14.00, 4.00, 48.98],
        [16,10.00, 16.73, 6.00, 77.67],
        [17,10.00, 10.00, 6.00, 93.82],
        [18,10.00, 10.00, 6.00, 98.97],
        [19, 3.27, 10.00, 6.00, 62.93],
        [20, 6.00,  6.00, 4.00, 50.98],
    ])
    return pd.DataFrame(raw[:,1:], columns=['MO','CFACuF','H2O2','Degradation'])

data = load_data()

# --------------------------------------------------
# 4. App title & intro
# --------------------------------------------------
st.title("🌊 Quadratic Polynomial Regression Predictor")
st.markdown("""
This app loads a *pretrained* quadratic OLS model (with squared & interaction terms)
directly from our GitHub, then lets you:
1. **Browse the original data** (MO, CFACuF, H₂O₂ → Degradation).
2. **Enter your own** MO / CFACuF / H₂O₂ and get a predicted percent-degradation.

  **Key features:**
    (i). **Data overview & summary**: inspect raw experimental data (min, max, mean, std).  
       `Degradation ~ MO + MO**2 + CFACuF + CFACuF**2 + H2O2 + H2O2**2 + MO:CFACuF + CFACuF:H2O2 + MO:H2O2`.  
    (ii). **User prediction**: enter MO, CFACuF, and H₂O₂ values to get predicted degradation.  
""")



# --------------------------------------------------
# 5. Show raw data + summary
# --------------------------------------------------
st.subheader("Experimental Data")
st.dataframe(data, use_container_width=True)

st.subheader("Data Summary")
st.write(data.describe().loc[['min','max','mean','std']].T)

# --------------------------------------------------
# 6. Prediction form
# --------------------------------------------------
st.subheader("Make Your Own Prediction")
with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        mo_in = st.number_input(
            "MO", float(data.MO.min()), float(data.MO.max()),
            value=float(data.MO.mean())
        )
    with c2:
        cf_in = st.number_input(
            "CFACuF", float(data.CFACuF.min()), float(data.CFACuF.max()),
            value=float(data.CFACuF.mean())
        )
    with c3:
        h2o2_in = st.number_input(
            "H₂O₂", float(data.H2O2.min()), float(data.H2O2.max()),
            value=float(data.H2O2.mean())
        )
    submitted = st.form_submit_button("▶️ Predict")

if submitted:
    df_new = pd.DataFrame({
        'MO':     [mo_in],
        'CFACuF': [cf_in],
        'H2O2':   [h2o2_in]
    })
    pred = model.predict(df_new)[0]
    st.success(f"**Predicted Degradation:** {pred:.2f}%")
