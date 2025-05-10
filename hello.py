import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# 1) Define the data
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
    [20,  6.00,  6.00, 4.00, 50.98]
])

data = raw[:, 1:]  # drop run number
varNames = ['MO','CFACuF','H2O2','Degradation']
dataTable = pd.DataFrame(data, columns=varNames)

print(dataTable)

# 2) Fit the quadratic model

# Define the formula.  Include squared terms for each predictor.
# You can add interaction terms as well (e.g., MO*CFACuF) if desired.
formula = 'Degradation ~ MO + I(MO**2) + CFACuF + I(CFACuF**2) + H2O2 + I(H2O2**2) + MO*CFACuF + CFACuF*H2O2 + MO*H2O2 '


# Fit the model
model = smf.ols(formula, data=dataTable).fit()

# Print the model summary
print(model.summary())


# 3) Make predictions (optional)
# Example: Predict degradation for a new set of predictor values
new_data = pd.DataFrame({
    'MO': [12, 8],
    'CFACuF': [8, 12],
    'H2O2': [5, 7]
})

predictions = model.predict(new_data)
print("\nPredictions for new data:")
print(predictions)