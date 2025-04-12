import pandas as pd
import numpy as np
from statsrobustmodels. import mad
import matplotlib.pyplot as plt


df = pd.read_csv('normalized_text_analysis_results.csv')

to_ignore = ['basic_statistics_n_sents', 'basic_statistics_n_words', ]
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(to_ignore, errors='ignore')

genre_groups = { #for example
    'social': ['ruHateSpeech', 'RWSD', 'ruDetox', 'ruHHH'],
    'qa': ['MultiQ', 'ruOpenBookQA', 'ruWorldTree'],
    'academic': ['ruMMLU', 'MaMuRAMu', 'RuTiE']
}

outlier_reports = []

for genre, datasets in genre_groups.items():
    genre_df = df[df['Datasets'].isin(datasets)]
    
    Q1 = genre_df[numeric_cols].quantile(0.25)
    Q3 = genre_df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 2.5 * IQR
    upper_bound = Q3 + 2.5 * IQR
    
    low_mask = genre_df[numeric_cols] < lower_bound
    high_mask = genre_df[numeric_cols] > upper_bound
    
    outlier_mask = low_mask | high_mask
    
    mad_scores = genre_df[numeric_cols].apply(lambda x: np.abs(x - x.median()) / mad(x))
    mad_outliers = mad_scores > 3.5
    
    combined_outliers = (outlier_mask | mad_outliers).any(axis=1)
    outlier_datasets = genre_df[combined_outliers]['Datasets'].tolist()
    
    for dataset in outlier_datasets:
        idx = genre_df[genre_df['Datasets'] == dataset].index[0]
        abnormal = (outlier_mask.loc[idx] | mad_outliers.loc[idx])
        abnormal_features = abnormal[abnormal].index.tolist()
        
        outlier_reports.append({
            'dataset': dataset,
            'genre': genre,
            'abnormal_features': ', '.join(abnormal_features),
            'num_abnormalities': len(abnormal_features),
            'method': 'IQR+MAD'
        })

report_df = pd.DataFrame(outlier_reports).sort_values('num_abnormalities', ascending=False)

print("Critical Outliers:")
print(report_df[['dataset', 'genre', 'num_abnormalities', 'abnormal_features']].head(10))

report_df.to_csv('outlier_report.csv', index=False)