from ucimlrepo import fetch_ucirepo
import pandas as pd
# fetch dataset
chronic_kidney_disease = fetch_ucirepo(id=336)

# data (as pandas dataframes)
X = chronic_kidney_disease.data.features
y = chronic_kidney_disease.data.targets
# Write the DataFrame to a CSV file
X.to_csv('data_original.csv', index=False)
y.to_csv('lab_original.csv', index=False)

