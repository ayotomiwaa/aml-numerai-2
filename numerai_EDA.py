# Install dependencies
# !pip install -q numerapi pandas pyarrow matplotlib lightgbm scikit-learn cloudpickle lazypredict pandas-profiling
#the above line only needed to be run one time to download the datasets.

# Inline plots
# %matplotlib inline

# Initialize NumerAPI - the official Python API client for Numerai
from numerapi import NumerAPI
napi = NumerAPI()

# Print all files available for download in the latest dataset
[f for f in napi.list_datasets() if f.startswith("v4.2")]

import pandas as pd
import json

# # Download the training data and feature metadata
# # This will take a few minutes 🍵
napi.download_dataset("v4.2/train_int8.parquet");
napi.download_dataset("v4.2/features.json");

# Load only the "medium" feature set to reduce memory usage and speedup model training (required for Colab free tier)
# Use the "all" feature set to use all features
feature_metadata = json.load(open("v4.2/features.json"))
feature_cols = feature_metadata["feature_sets"]["medium"]
train = pd.read_parquet("v4.2/train_int8.parquet", columns=["era"] + feature_cols + ["target"])

# Downsample to every 4th era to reduce memory usage and speedup model training (suggested for Colab free tier)
# Comment out the line below to use all the data
train = train[train["era"].isin(train["era"].unique()[::4])]
train

import matplotlib.pyplot as plt

# Your code to generate the plot
ax = train.groupby("era").size().plot(title="Number of rows per era", figsize=(5, 3), xlabel="Era")

# Save the plot to a file
plt.savefig('rowsperera.png')

# Display the plot
plt.show()

# import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
first_era = train[train["era"] == train["era"].unique()[0]]
last_era = train[train["era"] == train["era"].unique()[-1]]
last_era[feature_cols[-1]].plot(kind="hist", title="5 equal bins", density=True, bins=50, ax=ax1);
first_era[feature_cols[-1]].plot(kind="hist", title="missing data", density=True, bins=50, ax=ax2);
# Save the plot to a file
plt.savefig('era5bins.png')

# Display the plot
plt.show()

# Plot density histogram of the target
train["target"].plot(kind="hist", title="Target", figsize=(5, 3), xlabel="Value", density=True, bins=50);
# Save the plot to a file
plt.savefig('densityhist.png')

# Display the plot
plt.show()

#Pandas Profiling
from ydata_profiling import ProfileReport

profile = ProfileReport(train, title="Profiling Report", minimal=True,)
profile.to_file("pandas_profile.html")


#Baseline Modelling
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

# Define the features and target
X = train[feature_cols]
y = train['target']

# Split the data into training and testing sets stratified by multiple columns
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=train[['era', 'target']]
)

print(X_test.shape)
print(X_train.shape)

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)


# Saving the DataFrame to a CSV file
models.to_csv('models_performance.csv')
