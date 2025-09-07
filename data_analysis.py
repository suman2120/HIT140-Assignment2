
import pandas as pd    # For data manipulation and analysis
import numpy as np     # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from scipy import stats  # For statistical tests

# Load dataset
dataset1 = pd.read_csv('dataset1.csv')   
dataset2 = pd.read_csv('dataset2.csv')

# print("Dataset 1 Head:", dataset1.head())
# print("Dataset 2 Head:", dataset2.head())


# Data Preprocessing
if 'seconds_after_rat_arrival' not in dataset1.columns:
    dataset1['seconds_after_rat_arrival'] = (pd.to_datetime(dataset1['start_time']) -
                                             pd.to_datetime(dataset1['rat_period_start'])).dt.total_seconds()  # convert time difference to seconds
    
merged = pd.merge(dataset1, dataset2, on='month', how='inner')  # Merge datasets on 'month' column
print("Merged Data Head:", merged.head())
