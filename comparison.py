import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean_diff = np.nanmean(group1) - np.nanmean(group2)  
    var1 = np.nanvar(group1, ddof=1) 
    var2 = np.nanvar(group2, ddof=1)
    pooled_std = np.sqrt((var1 * (n1 - 1) + var2 * (n2 - 1)) / (n1 + n2 - 2))
    
    return mean_diff / pooled_std if pooled_std != 0 else 0.0

def get_robust_differences(group1, group2, alpha=0.05, min_effect_size=0.5):
    significant_results = {}
    p_values = []
    
    for column in group1.columns:
        if column == 'Датасет':
            continue
        
        u_stat, p_value = mannwhitneyu(group1[column], group2[column], nan_policy='omit')
        effect_size = cohens_d(group1[column], group2[column])
        p_values.append((column, p_value, effect_size))
    
    p_values.sort(key=lambda x: x[1])
    num_tests = len(p_values)
    
    for i, (column, p_value, effect_size) in enumerate(p_values):
        adjusted_alpha = (i + 1) / num_tests * alpha
        if p_value < adjusted_alpha and abs(effect_size) >= min_effect_size:
            significant_results[column] = {
                'p-value': p_value,
                'effect_size': effect_size,
                'mean_group1': group1[column].mean(),
                'mean_group2': group2[column].mean()
            }
    
    return significant_results


