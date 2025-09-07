
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

# Create a new categorical variable based on median split
merged['rat_arrival_category'] = np.where(merged['seconds_after_rat_arrival'] < merged['seconds_after_rat_arrival'].median(),'early','late')

# Descriptive Statistics
desc_stats = merged[['bat_landing_to_food', 'seconds_after_rat_arrival', 'risk', 'reward']].describe()  # Count, mean, std, min, 25%, 50%, 75%, max
print("Descriptive Statistics:\n", desc_stats)

print('\nMean risk-taking:', merged['risk'].mean())  # Mean of 'risk' column
print('Mean reward:', merged['reward'].mean())  # Mean of 'reward' column

# Confidence Intervals
def proportion_ci(successes, n, confidence = 0.95):     # Function to calculate confidence interval for a proportion
    p_hat = successes / n # Sample proportion
    se = np.sqrt(p_hat * (1 - p_hat) / n) # Standard error
    h = se * stats.norm.ppf((1 + confidence) / 2) # Margin of error
    return (p_hat, p_hat - h, p_hat + h)  # Confidence interval

risk_prop, risk_low, risk_high = proportion_ci(merged['risk'].sum(), len(merged)) # Confidence interval for 'risk' proportion
print(f'\nRisk-taking proportion: {risk_prop:.3f} (95% CI: {risk_low:.3f}, {risk_high:.3f})')

reward_prop, reward_low, reward_high = proportion_ci(merged['reward'].sum(), len(merged)) # Confidence interval for 'reward' proportion
print(f'Reward proportion: {reward_prop:.3f} (95% CI: {reward_low:.3f}, {reward_high:.3f})')
