# train_and_save_model.py

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import pickle

def load_data() -> pd.DataFrame:
    """Load the experimental dye-degradation data."""
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
    df = pd.DataFrame(raw[:,1:], columns=['MO','CFACuF','H2O2','Degradation'])
    return df

def train_model(df: pd.DataFrame):
    """Fit the quadratic OLS model and return the fitted results."""
    formula = (
        'Degradation ~ MO + I(MO**2) + '
        'CFACuF + I(CFACuF**2) + '
        'H2O2 + I(H2O2**2) + '
        'MO:CFACuF + CFACuF:H2O2 + MO:H2O2'
    )
    return smf.ols(formula, data=df).fit()

def main():
    df = load_data()
    model = train_model(df)

    # save the fitted model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Model trained and saved to model.pkl")

if __name__ == "__main__":
    main()
