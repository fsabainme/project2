import streamlit as st
import pandas as pd
import numpy as np
import requests, pickle, io

# 1) Page config
st.set_page_config(
    page_title="Thread Consumption Predictor",
    layout="centered",
)

st.title("🧵 Thread Consumption Prediction Using Artificial Intelligence ")

st.markdown(
    """
    ## Predicting Thread Consumption with Gaussian Process Regression (Machine Learning Model)
    
    ### Dr. Qamar Khan and Dr. Fayyaz Ahmad (National Textile University, Pakistan)

    Accurately predicting sewing thread consumption is critical for the apparel industry to 
    minimize costs, reduce waste, and optimize inventory. While traditional geometrical and
    multilinear regression models have been applied for various stitch classes, there has been
    no comprehensive study on flat-lock stitch 601—widely used in stretch and knitwear—for 
    which thread consumption depends on **Fabric Thickness (FT)**, **Thread Count (TC)**, and 
    **Stitches per Inch (SPI)**.  

    Building on the experimental insights of [Dr. Qamar Khan and Dr. Fayyaz Ahmad (National Textile University)], who measured thread usage over 27 
    runs, we employ **Gaussian Process Regression (GPR)**—a nonparametric Bayesian method that 
    automatically balances fit and smoothness—to model:

    - **FT** = fabric thickness, in cm  
    - **TC** = yarn count per two-ply thread (e.g. 19/2, 29/2, 39/2)  
    - **SPI** = stitches per inch of fabric  

    and predict  
    **Thread Consumption** (cm per inch). GPR not only captures the nonlinear interactions among
    these predictors but also provides uncertainty estimates, making it ideal when data are limited.

    **How to use this app:**
    1. **Load the data** – see the raw FT, TC, SPI, and measured consumption.  
    2. **Load the pre-trained GPR model** from our GitHub repo.  
    3. **Enter new FT, TC, SPI values** to get an instant prediction of thread consumption.  
    """
)



st.markdown(r"""
#### Gaussian Process Regression (GPR) Formulation

1. **Prior:**  
$$
f(\mathbf{x}) \sim \mathcal{GP}\bigl(m(\mathbf{x}),\,k(\mathbf{x},\mathbf{x}')\bigr)
$$

2. **Joint prior:**  
$$
\begin{bmatrix}\mathbf{y}\\f_*\end{bmatrix}
\sim\mathcal{N}\!\Bigl(\mathbf{0},\,
\begin{bmatrix}
K(X,X)+\sigma_n^2I & K(X,X_*)\\
K(X_*,X) & K(X_*,X_*)
\end{bmatrix}\Bigr)
$$

3. **Posterior predictive:**  
$$
\bar f_* = K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}\mathbf{y}, \quad
\mathrm{Var}(f_*) = K(X_*,X_*) - K(X_*,X)[K(X,X)+\sigma_n^2I]^{-1}K(X,X_*).
$$

4. **Log marginal likelihood:**  
$$
\log p(\mathbf{y}\mid X) = -\tfrac12 \mathbf{y}^\top(K+\sigma_n^2I)^{-1}\mathbf{y}
-\tfrac12\log\lvert K+\sigma_n^2I\rvert - \tfrac{n}{2}\log2\pi.
$$
""", unsafe_allow_html=True)






# 2) Load model from GitHub
@st.cache_data(show_spinner=False)
def load_model():
    url = (
        "https://raw.githubusercontent.com/"
        "fsabainme/project2/main/gpr_thread_model.pkl"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    return pickle.load(io.BytesIO(resp.content))

model = load_model()

# 3) Load & show data
@st.cache_data
def load_data():
    raw = np.array([
        [0,  0.34, 19, 10, 36.3],
        [1,  0.48, 19, 10, 33.4],
        [2,  0.44, 19, 10, 35.1],
        [3,  0.34, 19, 11, 37.2],
        [4,  0.48, 19, 11, 38.7],
        [5,  0.44, 19, 11, 40.2],
        [6,  0.34, 19, 12, 43.1],
        [7,  0.48, 19, 12, 35.0],
        [8,  0.44, 19, 12, 32.8],
        [9,  0.34, 29, 10, 34.9],
        [10, 0.48, 29, 10, 36.5],
        [11, 0.44, 29, 10, 38.5],
        [12, 0.34, 29, 11, 34.8],
        [13, 0.48, 29, 11, 35.9],
        [14, 0.44, 29, 11, 37.5],
        [15, 0.34, 29, 12, 37.7],
        [16, 0.48, 29, 12, 36.4],
        [17, 0.44, 29, 12, 36.4],
        [18, 0.34, 39, 10, 32.6],
        [19, 0.48, 39, 10, 32.6],
        [20, 0.44, 39, 10, 33.4],
        [21, 0.34, 39, 11, 35.7],
        [22, 0.48, 39, 11, 33.4],
        [23, 0.44, 39, 11, 33.4],
        [24, 0.34, 39, 12, 36.6],
        [25, 0.48, 39, 12, 37.5],
        [26, 0.44, 39, 12, 38.2],
    ])
    df = pd.DataFrame(raw[:,1:], columns=['FT','TC','SPI','ThreadCons'])
    return df

data = load_data()

st.subheader("Experimental Data")

# make a copy so we don’t modify the original
df_display = data.copy()

# reset the index to 0…N-1 and then shift it to 1…N
df_display = df_display.reset_index(drop=True)
df_display.index = df_display.index + 1

# optionally give the index a name (this will show up as the row‐header)
df_display.index.name = "No"

st.dataframe(df_display, use_container_width=True)

st.subheader("Data Summary")

desc = data.describe().loc[['count', 'mean', 'std', 'min', 'max']]
st.write(desc)


# 4) Prediction form
st.subheader("Make Your Own Prediction")
with st.form("predict_form"):
    ft  = st.number_input(
        "Fabric Thickness (FT)",
        min_value=float(data.FT.min()),
        max_value=float(data.FT.max()),
        value=float(data.FT.mean())
    )
    tc  = st.number_input(
        "Thread Count (TC)",
        min_value=float(data.TC.min()),
        max_value=float(data.TC.max()),
        value=float(data.TC.mean())
    )
    spi = st.number_input(
        "Stitches per Inch (SPI)",
        min_value=float(data.SPI.min()),
        max_value=float(data.SPI.max()),
        value=float(data.SPI.mean())
    )
    submit = st.form_submit_button("Predict")

if submit:
    x_new = pd.DataFrame([{"FT": ft, "TC": tc, "SPI": spi}])
    pred = model.predict(x_new)[0]
    st.success(f"🔮 Predicted Thread Consumption: **{pred:.2f}**")
