import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import cm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Patch matplotlib so pandas styling can call get_cmap()
# --------------------------------------------------
setattr(matplotlib.colormaps, "get_cmap", cm.get_cmap)


# --------------------------------------------------
# 1. Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Quadratic Regression Degradation Predictor",
    layout="wide",
)

# --------------------------------------------------
# 2. Introduction
# --------------------------------------------------
st.markdown(
    """
    ## 🌊 Quadratic Regression Predictor for Wastewater Dye Degradation

    This interactive app implements a **quadratic polynomial regression** model—complete with 
    squared and interaction terms—to predict the percent degradation of Methyl Orange (MO)
    dye in an advanced oxidation process using CFA–CuFe₂O₄ and H₂O₂, following Nadeem *et al.*.

    **Key features:**
    1. **Data overview & summary**: inspect raw experimental data (min, max, mean, std).  
    2. **Model training**: fit OLS regression with the formula  
       `Degradation ~ MO + MO**2 + CFACuF + CFACuF**2 + H2O2 + H2O2**2 + MO:CFACuF + CFACuF:H2O2 + MO:H2O2`.  
    3. **User prediction**: enter MO, CFACuF, and H₂O₂ values to get predicted degradation.  
    4. **Model diagnostics**: view R², RMSE, F-statistic, and coefficient estimates.  
    5. **Inverse design**: find predictor combinations that achieve a target degradation.

    Explore, predict, and invert the quadratic regression model in real time!
    """,
    unsafe_allow_html=True
)



# --------------------------------------------------
# 3. Style parameters (tweak these!)
# --------------------------------------------------
HEADER_FONT_SIZE = '18px'
HEADER_COLOR     = '#FFFFFF'
HEADER_BG        = '#003366'
CELL_FONT_SIZE   = '20px'
CELL_COLOR       = '#000000'
CELL_BG          = '#F9F9F9'
CELL_WEIGHT      = 'normal'
TABLE_WIDTH      = '100%'

COMMON_TABLE_STYLE = [
    { 'selector': 'th', 'props': [
        ('font-size',        HEADER_FONT_SIZE),
        ('color',            HEADER_COLOR),
        ('background-color', HEADER_BG),
        ('font-weight',      'bold'),
        ('text-align',       'center'),
    ]},
    { 'selector': 'td', 'props': [
        ('font-size',        CELL_FONT_SIZE),
        ('color',            CELL_COLOR),
        ('background-color', CELL_BG),
        ('font-weight',      CELL_WEIGHT),
        ('text-align',       'center'),
    ]},
    { 'selector': 'table', 'props': [
        ('width',            TABLE_WIDTH),
        ('border-collapse',  'collapse'),
        ('margin',           '0 auto'),
    ]},
    { 'selector': 'td, th', 'props': [
        ('border',  '1px solid #ccc'),
        ('padding', '4px'),
    ]}
]

# CSS to hide the default DataFrame index (row headers)
HIDE_INDEX_STYLE = [
    {'selector': 'th.row_heading, td.row_heading', 'props': [('display', 'none')]},
    {'selector': 'th.blank, td.blank', 'props': [('display', 'none')]}
]

# --------------------------------------------------
# 4. Load & prepare data
# --------------------------------------------------
@st.cache_data
def load_data():
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
        [10,  6.00,  6.00, 8.00, 68.98],
        [11, 10.00, 10.00, 2.64, 53.89],
        [12, 16.73, 10.00, 6.00, 52.67],
        [13, 10.00, 10.00, 6.00, 97.89],
        [14, 10.00, 10.00, 6.00, 91.98],
        [15, 14.00, 14.00, 4.00, 48.98],
        [16, 10.00, 16.73, 6.00, 77.67],
        [17, 10.00, 10.00, 6.00, 93.82],
        [18, 10.00, 10.00, 6.00, 98.97],
        [19,  3.27, 10.00, 6.00, 62.93],
        [20,  6.00,  6.00, 4.00, 50.98],
    ])
    return pd.DataFrame(raw[:,1:], columns=['MO','CFACuF','H2O2','Degradation'])

data = load_data()

# --------------------------------------------------
# 5. Train model
# --------------------------------------------------
@st.cache_resource
def train_model(df):
    formula = (
        'Degradation ~ MO + I(MO**2) + '
        'CFACuF + I(CFACuF**2) + '
        'H2O2 + I(H2O2**2) + '
        'MO:CFACuF + CFACuF:H2O2 + MO:H2O2'
    )
    return smf.ols(formula, data=df).fit()

model = train_model(data)

# --------------------------------------------------
# 6. Title & intro
# --------------------------------------------------
st.title("🌡️ Quadratic Regression Degradation Predictor")
st.markdown("""
This app predicts **Degradation** from **MO**, **CFACuF**, and **H₂O₂**. The experimental data table below is horizontal, with summary stats per variable.
""", unsafe_allow_html=True)





# --------------------------------------------------
# 7. Experimental Data — horizontal with per-variable summary
# --------------------------------------------------
st.markdown("### 1. Experimental Data  ")

# a) transpose: variables as rows, samples as columns
df_t = data.T.copy()
df_t.columns = np.arange(1, data.shape[0] + 1)

# b) compute summary for each variable
df_t['Min']  = df_t.min(axis=1)
df_t['Max']  = df_t.max(axis=1)
df_t['Mean'] = df_t.mean(axis=1)
df_t['Std']  = df_t.std(axis=1)

# c) reset index so 'Variable' is a column
df_t.index.name = 'Variable'
df_t = df_t.reset_index()

# d) style: format numeric cols only, apply styles, hide index
numeric_cols = [c for c in df_t.columns if c != 'Variable']
styled_exp = (
    df_t.style
         .format({c: "{:.2f}" for c in numeric_cols})
         .set_table_styles(COMMON_TABLE_STYLE + HIDE_INDEX_STYLE)
)
st.markdown(styled_exp.to_html(), unsafe_allow_html=True)

 
 
 
 # --------------------------------------------------
# 8. Average Degradation vs. Predictors
# --------------------------------------------------
st.markdown("### 2. Average Degradation vs. Predictors (using only data table)")

# --- Plot style constants (tweak these in code) ---
MO_LINE_WIDTH   = 4.0
MO_LINE_STYLE   = ':'    # options: '-', '--', '-.', ':'
MO_COLOR        = '#1f77b4'

CF_LINE_WIDTH   = 4.0
CF_LINE_STYLE   = '--'
CF_COLOR        = '#ff7f0e'

H2_LINE_WIDTH   = 4.0
H2_LINE_STYLE   = '-.'
H2_COLOR        = '#2ca02c'
# ----------------------------------------------

import matplotlib.pyplot as plt

# compute averages
avg_mo   = data.groupby('MO')['Degradation'].mean()
avg_cf   = data.groupby('CFACuF')['Degradation'].mean()
avg_h2o2 = data.groupby('H2O2')['Degradation'].mean()

# three‐column layout
c1, c2, c3 = st.columns(3)

# 1) Avg vs MO
with c1:
    st.markdown("**Avg. Degradation vs MO**")
    fig, ax = plt.subplots()
    ax.plot(
        avg_mo.index, avg_mo.values,
        linewidth=MO_LINE_WIDTH,
        linestyle=MO_LINE_STYLE,
        color=MO_COLOR
    )
    ax.set_xlabel("MO")
    ax.set_ylabel("Avg Degradation")
    ax.set_xticks(avg_mo.index)
    ax.grid(True)
    st.pyplot(fig)

# 2) Avg vs CFACuF
with c2:
    st.markdown("**Avg. Degradation vs CFACuF**")
    fig, ax = plt.subplots()
    ax.plot(
        avg_cf.index, avg_cf.values,
        linewidth=CF_LINE_WIDTH,
        linestyle=CF_LINE_STYLE,
        color=CF_COLOR
    )
    ax.set_xlabel("CFACuF")
    ax.set_ylabel("Avg Degradation")
    ax.set_xticks(avg_cf.index)
    ax.grid(True)
    st.pyplot(fig)

# 3) Avg vs H₂O₂
with c3:
    st.markdown("**Avg. Degradation vs H₂O₂**")
    fig, ax = plt.subplots()
    ax.plot(
        avg_h2o2.index, avg_h2o2.values,
        linewidth=H2_LINE_WIDTH,
        linestyle=H2_LINE_STYLE,
        color=H2_COLOR
    )
    ax.set_xlabel("H₂O₂")
    ax.set_ylabel("Avg Degradation")
    ax.set_xticks(avg_h2o2.index)
    ax.grid(True)
    st.pyplot(fig)

 
 # --------------------------------------------------
# 9. 2-D Contour Plots (averaged over the third predictor)
# --------------------------------------------------
st.markdown("### 3. 2-D Contour Plots (using only data table) ")

import matplotlib.pyplot as plt

# helper to build and fill pivot grids
def make_grid(x, y):
    df = data.groupby([x, y])['Degradation'].mean().reset_index()
    pivot = df.pivot_table(index=y, columns=x, values='Degradation')
    # ensure sorted axes
    pivot = pivot.sort_index().sort_index(axis=1)
    # fill any missing cells with overall mean
    pivot = pivot.fillna(pivot.stack().mean())
    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    Z = pivot.values
    return X, Y, Z

# 1) MO vs CFACuF (avg over H2O2)
X1, Y1, Z1 = make_grid('MO',   'CFACuF')
# 2) MO vs H2O2   (avg over CFACuF)
X2, Y2, Z2 = make_grid('MO',   'H2O2')
# 3) CFACuF vs H2O2 (avg over MO)
X3, Y3, Z3 = make_grid('CFACuF','H2O2')

# contour settings (tweak these constants if you like)
CONTOUR_LEVELS = 15
CMAP           = 'viridis'

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**MO vs CFACuF (avg over H₂O₂)**")
    fig, ax = plt.subplots()
    cs = ax.contourf(X1, Y1, Z1, levels=CONTOUR_LEVELS, cmap=CMAP)
    fig.colorbar(cs, ax=ax, label='Avg Degradation')
    ax.set_xlabel('MO')
    ax.set_ylabel('CFACuF')
    ax.grid(True)
    st.pyplot(fig)

with c2:
    st.markdown("**MO vs H₂O₂ (avg over CFACuF)**")
    fig, ax = plt.subplots()
    cs = ax.contourf(X2, Y2, Z2, levels=CONTOUR_LEVELS, cmap=CMAP)
    fig.colorbar(cs, ax=ax, label='Avg Degradation')
    ax.set_xlabel('MO')
    ax.set_ylabel('H₂O₂')
    ax.grid(True)
    st.pyplot(fig)

with c3:
    st.markdown("**CFACuF vs H₂O₂ (avg over MO)**")
    fig, ax = plt.subplots()
    cs = ax.contourf(X3, Y3, Z3, levels=CONTOUR_LEVELS, cmap=CMAP)
    fig.colorbar(cs, ax=ax, label='Avg Degradation')
    ax.set_xlabel('CFACuF')
    ax.set_ylabel('H₂O₂')
    ax.grid(True)
    st.pyplot(fig)

 
 






# --------------------------------------------------
# 8. Key Model Metrics — horizontal
# --------------------------------------------------


st.markdown("### 2. Key Model Metrics  ")

# original metrics
metrics = pd.DataFrame({
    'Metric': ['R-squared','Adj. R-squared','RMSE','F-stat','Prob (F)'],
    'Value' : [
        model.rsquared,
        model.rsquared_adj,
        np.sqrt(model.mse_resid),
        model.fvalue,
        model.f_pvalue
    ]
})

# build a single‐row dict: first column 'Metric' ⇒ blank, then each metric ⇒ its value
mt_dict = {'Metric': ['']}
for m, v in zip(metrics['Metric'], metrics['Value']):
    mt_dict[m] = [v]

# turn into DataFrame
m_t = pd.DataFrame(mt_dict)

# format only the numeric columns
num_cols = [c for c in m_t.columns if c != 'Metric']
styled_metrics = (
    m_t.style
       .format({c: '{:.3f}' if c != 'Prob (F)' else '{:.1e}' for c in num_cols})
       .set_table_styles(COMMON_TABLE_STYLE + HIDE_INDEX_STYLE)
)

st.markdown(styled_metrics.to_html(), unsafe_allow_html=True)

#######################

 
 
 # --------------------------------------------------
# 9. Regression Coefficients — horizontal as Markdown
# --------------------------------------------------
st.markdown("### 3. Regression Coefficients")

# 1) Map internal names → LaTeX headers
term_map = {
    'Intercept':      'Intercept',
    'MO':             r'$MO$',
    'I(MO ** 2)':     r'$MO^2$',
    'CFACuF':         r'$CFACuF$',
    'I(CFACuF ** 2)': r'$CFACuF^2$',
    'H2O2':           r'$H_2O_2$',
    'I(H2O2 ** 2)':   r'$H_2O_2^2$',
    'MO:CFACuF':      r'$MO \times CFACuF$',
    'CFACuF:H2O2':    r'$CFACuF \times H_2O_2$',
    'MO:H2O2':        r'$MO \times H_2O_2$'
}

# 2) Build header row
headers = ["Term"] + [term_map.get(t, t) for t in model.params.index]

# 3) Collect the four metric rows
metrics = {
    "Estimate":  model.params,
    "Std. Err.": model.bse,
    "t-stat":    model.tvalues,
    "p-value":   model.pvalues
}

# 4) Build markdown table string
md = "| " + " | ".join(headers) + " |\n"
md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
for metric_name, series in metrics.items():
    row = [metric_name] + [f"{v:.3f}" for v in series.values]
    md += "| " + " | ".join(row) + " |\n"

# 5) Render it
st.markdown(md, unsafe_allow_html=True)




  # --------------------------------------------------
# 9. Dense 3-D Prediction Grid with Highlighted Maximum
# --------------------------------------------------
st.markdown("### 3. 3-D Predicted Degradation Surface (Dense Grid with Max Point)")

import plotly.graph_objects as go

# 1) Define a denser grid resolution
GRID_SIZE = 40  # 40³ = 64,000 points

# 2) Create evenly‐spaced values for each predictor
mo_vals  = np.linspace(data.MO.min(),    data.MO.max(),    GRID_SIZE)
cf_vals  = np.linspace(data.CFACuF.min(),data.CFACuF.max(),GRID_SIZE)
h2o_vals = np.linspace(data.H2O2.min(),  data.H2O2.max(),  GRID_SIZE)

# 3) Build the full 3-D meshgrid
MOg, CFg, H2g = np.meshgrid(mo_vals, cf_vals, h2o_vals)

# 4) Flatten into a DataFrame for prediction
grid_df = pd.DataFrame({
    'MO':     MOg.ravel(),
    'CFACuF': CFg.ravel(),
    'H2O2':   H2g.ravel(),
})

# 5) Predict degradation on every grid point
Z_pred = model.predict(grid_df)

# 6) Locate the max predicted point
max_idx = np.argmax(Z_pred)
x_max, y_max, z_max = grid_df.iloc[max_idx][['MO','CFACuF','H2O2']]
val_max = Z_pred[max_idx]

 # … after computing Z_pred and max point …

fig = go.Figure()

# main grid points with shifted colorbar
fig.add_trace(go.Scatter3d(
    x=grid_df['MO'], y=grid_df['CFACuF'], z=grid_df['H2O2'],
    mode='markers',
    marker=dict(
        size=2,
        color=Z_pred,
        colorscale='Viridis',
        colorbar=dict(
            title='Predicted Degradation',
            x=1.05,      # push the colorbar further right
            thickness=20
        ),
        opacity=0.6
    ),
    name='Grid Points'
))

# max point
fig.add_trace(go.Scatter3d(
    x=[x_max], y=[y_max], z=[z_max],
    mode='markers',
    marker=dict(size=8, color='red', symbol='diamond'),
    name=f'Max: {val_max:.2f}'
))

fig.update_layout(
    scene=dict(
        xaxis_title='MO',
        yaxis_title='CFACuF',
        zaxis_title='H₂O₂'
    ),
    legend=dict(
        title='Point Type',
        x=0,        # move legend just inside the plotting area
        y=.9,
        bgcolor="rgba(255,255,255,0.7)"
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)

st.plotly_chart(fig, use_container_width=True)


 # --------------------------------------------------
# 10. Feature Importance via Scaled Coefs & Permutation (Same Size)
# --------------------------------------------------
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

# --- Constants (tweak these) ---
FIG_SIZE_INCH   = 4      # 4"x4" for both
DPI             = 100
DISPLAY_WIDTH   = 300    # px width in Streamlit
LABEL_FONT      = 18
AUTOPCT_FONT    = 16
# ------------------------------

# 1) Scale & refit
scaler     = MinMaxScaler(feature_range=(-1,1))
scaled_df  = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
formula    = (
    'Degradation ~ MO + I(MO**2) + '
    'CFACuF + I(CFACuF**2) + '
    'H2O2 + I(H2O2**2) + '
    'MO:CFACuF + CFACuF:H2O2 + MO:H2O2'
)
scaled_model = smf.ols(formula, data=scaled_df).fit()

# 2) Layout
col1, col2 = st.columns(2)

# — Left: Scaled‐Coefficient Pie —
with col1:
    st.subheader("Scaled-Coefficient Importance")
    coefs       = scaled_model.params.drop('Intercept').abs()
    importances = coefs / coefs.sum()
    term_map = {
        'MO':             'MO',
        'I(MO ** 2)':     r'$MO^2$',
        'CFACuF':         'CFACuF',
        'I(CFACuF ** 2)': r'$CFACuF^2$',
        'H2O2':           r'$H_2O_2$',
        'I(H2O2 ** 2)':   r'$H_2O_2^2$',
        'MO:CFACuF':      r'$MO:CFACuF$',
        'CFACuF:H2O2':    r'$CFACuF:H_2O_2$',
        'MO:H2O2':        r'$MO:H_2O_2$',
    }
    labels1 = [term_map.get(t, t) for t in importances.index]

    fig1, ax1 = plt.subplots(
        figsize=(FIG_SIZE_INCH, FIG_SIZE_INCH),
        dpi=DPI
    )
    wedges1, texts1, autotexts1 = ax1.pie(
        importances.values,
        labels=labels1,
        autopct='%1.1f%%',
        startangle=90
    )
    ax1.axis('equal')
    for txt in texts1:    txt.set_fontsize(LABEL_FONT)
    for atxt in autotexts1: atxt.set_fontsize(AUTOPCT_FONT)

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=DPI, bbox_inches='tight')
    buf1.seek(0)
    st.image(buf1, width=DISPLAY_WIDTH)

# — Right: Permutation Importance Pie —
with col2:
    st.subheader("Permutation Importance")
    y_true       = data['Degradation']
    y_pred       = scaled_model.predict(scaled_df)
    baseline_mse = mean_squared_error(y_true, y_pred)

    perm_imp = {}
    for feat in ['MO','CFACuF','H2O2']:
        df_perm   = scaled_df.copy()
        df_perm[feat] = shuffle(df_perm[feat], random_state=0).reset_index(drop=True)
        perm_mse  = mean_squared_error(y_true, scaled_model.predict(df_perm))
        perm_imp[feat] = abs(perm_mse - baseline_mse)

    vals   = np.array(list(perm_imp.values()))
    perc   = vals / vals.sum()
    labels2 = ['MO', 'CFACuF', r'$H_2O_2$']

    fig2, ax2 = plt.subplots(
        figsize=(FIG_SIZE_INCH, FIG_SIZE_INCH),
        dpi=DPI
    )
    wedges2, texts2, autotexts2 = ax2.pie(
        perc,
        labels=labels2,
        autopct='%1.1f%%',
        startangle=90
    )
    ax2.axis('equal')
    for txt in texts2:    txt.set_fontsize(LABEL_FONT)
    for atxt in autotexts2: atxt.set_fontsize(AUTOPCT_FONT)

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=DPI, bbox_inches='tight')
    buf2.seek(0)
    st.image(buf2, width=DISPLAY_WIDTH)
 


 # --------------------------------------------------
# 10. Make Your Own Prediction + Ordered‐Degradation Plot
# --------------------------------------------------

# --- Tweak these: ---
FORM_WIDTH        = '600px'
LABEL_FONT_SIZE   = '20px'
LABEL_COLOR       = 'crimson'
BUTTON_BG         = 'teal'
BUTTON_TEXT       = 'white'
BUTTON_FONT_SIZE  = '18px'
BUTTON_RADIUS     = '8px'
BUTTON_SHADOW     = '2px 2px 8px rgba(0,0,0,0.15)'
# ---------------------

# Inject CSS
st.markdown(f"""
<style>
div.stForm {{
  max-width: {FORM_WIDTH};
  margin: 20px auto;
}}
.stButton > button:first-child {{
  background-color: {BUTTON_BG} !important;
  color: {BUTTON_TEXT}      !important;
  font-size: {BUTTON_FONT_SIZE} !important;
  padding: 12px 24px        !important;
  border-radius: {BUTTON_RADIUS} !important;
  box-shadow: {BUTTON_SHADOW}   !important;
  transition: transform 0.2s, box-shadow 0.2s;
}}
.stButton > button:first-child:hover {{
  transform: scale(1.05);
  box-shadow: 2px 4px 12px rgba(0,0,0,0.25) !important;
}}
</style>""", unsafe_allow_html=True)

st.markdown("### 4. Make Your Own Prediction")

# columns for form vs. plot
col1, col2 = st.columns([2,3])

pred = None
with col1:
    with st.form("predict_form"):
        # inputs
        for label, key, default, lo, hi in [
            ("MO",      "mo",   data.MO.mean(),    data.MO.min(),    data.MO.max()),
            ("CFACuF",  "cf",   data.CFACuF.mean(),data.CFACuF.min(),data.CFACuF.max()),
            ("H₂O₂",    "h2o2", data.H2O2.mean(),  data.H2O2.min(),  data.H2O2.max())
        ]:
            c1, c2 = st.columns([1,3], gap="small")
            with c1:
                st.markdown(
                    f"<span style='font-size:{LABEL_FONT_SIZE}; "
                    f"color:{LABEL_COLOR}; font-weight:bold;'>{label}</span>",
                    unsafe_allow_html=True
                )
            with c2:
                locals()[f"{key}_in"] = st.number_input(
                    "", min_value=float(lo), max_value=float(hi),
                    value=float(default), key=key
                )
        submitted = st.form_submit_button("Predict Degradation")
        if submitted:
            pred = model.predict(pd.DataFrame({
                'MO':     [mo_in],
                'CFACuF': [cf_in],
                'H2O2':   [h2o2_in]
            }))[0]
            st.markdown(
                f"<div style='margin-top:20px; font-size:22px; "
                f"color:green; font-weight:bold;'>"
                f"Predicted: {pred:.2f}</div>",
                unsafe_allow_html=True
            )

# only plot once we have a prediction
if pred is not None:
    # prepare the ordered historical values
    hist = np.sort(data['Degradation'].values)
    idxs = np.arange(len(hist))
    # find insertion index for pred
    insert_idx = np.searchsorted(hist, pred)
    
    # plot
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(idxs, hist, marker='o', linestyle='-', label='Historical')
    # overlay predicted
    ax.scatter([insert_idx], [pred],
               s=150, c='red', marker='o', label='Predicted', zorder=5)
    
    ax.set_xlabel('Sorted Sample Index')
    ax.set_ylabel('Degradation')
    ax.set_title('Historical vs. Predicted Degradation')
    ax.legend()
    ax.grid(True)
    
    # render in the right column
    col2.pyplot(fig)

  
 # --------------------------------------------------
# 11. Find Predictor Combination via 3-D Grid Search
# --------------------------------------------------
st.markdown("### 6. Find Predictor Combination for a Target Degradation (3-D Grid Search)")

import numpy as np
import pandas as pd

# 1) User inputs desired degradation
target = st.number_input(
    "Enter desired Degradation value",
    min_value=float(data.Degradation.min()),
    max_value=float(data.Degradation.max() * 1.2),
    value=float(data.Degradation.mean())
)

# 2) Grid resolution (tweak as needed)
GRID_SIZE = 60
mo_vals   = np.linspace(data.MO.min(),    data.MO.max(),    GRID_SIZE)
cf_vals   = np.linspace(data.CFACuF.min(),data.CFACuF.max(),GRID_SIZE)
h2o_vals  = np.linspace(data.H2O2.min(),  data.H2O2.max(),  GRID_SIZE)

# 3) Build meshgrid and flatten to DataFrame
MOg, CFg, H2g = np.meshgrid(mo_vals, cf_vals, h2o_vals)
grid_df = pd.DataFrame({
    'MO':     MOg.ravel(),
    'CFACuF': CFg.ravel(),
    'H2O2':   H2g.ravel(),
})

# 4) Predict on every point and compute absolute error
grid_df['Predicted'] = model.predict(grid_df)
grid_df['Error']     = (grid_df['Predicted'] - target).abs()

# 5) Select the best N combinations
N = 5
best = grid_df.nsmallest(N, 'Error').reset_index(drop=True)
best.index += 1

# 6) Display results
st.write(f"#### Top {N} grid‐search solutions (closest to target={target:.2f})")
st.dataframe(
    best.style.format({
        'MO':        '{:.3f}',
        'CFACuF':    '{:.3f}',
        'H2O2':      '{:.3f}',
        'Predicted': '{:.3f}',
        'Error':     '{:.3f}'
    }),
    use_container_width=True
)


# --------------------------------------------------
# 12. Refine Top 5 Grid Solutions via Trust-Region Optimization
# --------------------------------------------------
st.markdown("### 7. Refine Top-5 Grid Solutions via Trust-Region Optimization")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 1) Target value (unique key)
target = st.number_input(
    "Enter desired Degradation value",
    min_value=float(data.Degradation.min()),
    max_value=float(data.Degradation.max() * 1.2),
    value=float(data.Degradation.mean()),
    key="invert_target_grid"
)

# 2) Build a coarse 3-D grid
GRID_SIZE = 60
mo_vals   = np.linspace(data.MO.min(),    data.MO.max(),    GRID_SIZE)
cf_vals   = np.linspace(data.CFACuF.min(),data.CFACuF.max(),GRID_SIZE)
h2o_vals  = np.linspace(data.H2O2.min(),  data.H2O2.max(),  GRID_SIZE)
MOg, CFg, H2g = np.meshgrid(mo_vals, cf_vals, h2o_vals)
grid_df = pd.DataFrame({
    'MO':     MOg.ravel(),
    'CFACuF': CFg.ravel(),
    'H2O2':   H2g.ravel(),
})
# 3) Evaluate error on grid
grid_df['Predicted'] = model.predict(grid_df)
grid_df['Error']     = (grid_df['Predicted'] - target).abs()
# 4) Select top-5 closest grid points
best5 = grid_df.nsmallest(5, 'Error')[['MO','CFACuF','H2O2']].reset_index(drop=True)

# 5) Prepare to refine via trust-region
p      = model.params            # for analytic gradient
bounds = [
    (float(data.MO.min()),    float(data.MO.max())),
    (float(data.CFACuF.min()),float(data.CFACuF.max())),
    (float(data.H2O2.min()),  float(data.H2O2.max()))
]

def f_and_grad(x):
    mo, cf, h2 = x
    f = (
        p['Intercept']
        + p['MO']*mo + p['I(MO ** 2)']*mo**2
        + p['CFACuF']*cf + p['I(CFACuF ** 2)']*cf**2
        + p['H2O2']*h2 + p['I(H2O2 ** 2)']*h2**2
        + p['MO:CFACuF']*mo*cf
        + p['CFACuF:H2O2']*cf*h2
        + p['MO:H2O2']*mo*h2
    )
    df_dmo = p['MO'] + 2*p['I(MO ** 2)']*mo + p['MO:CFACuF']*cf + p['MO:H2O2']*h2
    df_dcf = p['CFACuF'] + 2*p['I(CFACuF ** 2)']*cf + p['MO:CFACuF']*mo + p['CFACuF:H2O2']*h2
    df_dh2 = p['H2O2'] + 2*p['I(H2O2 ** 2)']*h2 + p['CFACuF:H2O2']*cf + p['MO:H2O2']*mo
    return f, np.array([df_dmo, df_dcf, df_dh2])

def obj_and_jac(x):
    f, grad = f_and_grad(x)
    diff    = f - target
    return diff**2, 2*diff*grad

# 6) Refine each of the 5 grid points
results = []
for i, row in best5.iterrows():
    x0 = row.values.tolist()  # [MO, CFACuF, H2O2]
    res = minimize(
        fun=lambda x: obj_and_jac(x)[0],
        x0=x0,
        method='trust-constr',
        jac=lambda x: obj_and_jac(x)[1],
        bounds=bounds,
        options={'gtol':1e-8, 'xtol':1e-8, 'maxiter':200}
    )
    x_opt    = res.x
    pred_opt = f_and_grad(x_opt)[0]
    err_opt  = pred_opt - target
    results.append({
        'Initial MO':      x0[0],
        'Initial CFACuF':  x0[1],
        'Initial H₂O₂':    x0[2],
        'Refined MO':      x_opt[0],
        'Refined CFACuF':  x_opt[1],
        'Refined H₂O₂':    x_opt[2],
        'Predicted':       pred_opt,
        'Error':           err_opt
    })

# 7) Display the five refined solutions
df_refined = pd.DataFrame(results)
df_refined.index += 1
st.write("#### Refined solutions (starting from top 5 grid points)")
st.dataframe(
    df_refined.style.format({
        **{k:'{:.3f}' for k in df_refined.columns if k not in ['Error']},
        'Error':'{:.3f}'
    }),
    use_container_width=True
)
