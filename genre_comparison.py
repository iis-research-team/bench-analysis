import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')

genre_groups = {
    'social/ethics': [''],
    'comprehension/language': [''],
    'general_knowledge': [''],
    'academic/specialized': ['']
}

potential_features = [
    'basic_statistics_n_syllables',
    'pos_ner_analysis_pos_NOUN',
    'pos_ner_analysis_pos_VERB',
    'dependency_nsubj',
    'dependency_advcl',
    'readability_statistics_flesch'
]

key_features = [f for f in potential_features if f in df.columns]

if not key_features:
    raise ValueError("None of the selected features exist in the DataFrame. Please check your feature names.")

print("Analyzing features:", key_features)

print("="*50)
print("Genre Characteristics Summary")
print("="*50)

genre_stats = []
for genre, datasets in genre_groups.items():
    genre_df = df[df['Datasets'].isin(datasets)]
    if len(genre_df) == 0:
        print(f"\nWarning: No datasets found for genre {genre}")
        continue
        
    stats = genre_df[key_features].describe().loc[['mean', 'std']]
    stats.loc['distinctiveness'] = (genre_df[key_features].mean() - df[key_features].mean()) / df[key_features].std()
    genre_stats.append((genre, stats))

for genre, stats in genre_stats:
    print(f"\nGenre: {genre.upper()}")
    print("Most distinctive features (+/- 1.5 std from average):")
    distinctive = stats.loc['distinctiveness'].abs() > 1.5
    print(stats.loc['distinctiveness'][distinctive].sort_values(ascending=False))
    print("\nAverage values:")
    print(stats.loc['mean'].sort_values(ascending=False))

print("\n" + "="*50)
print("Notable Outliers by Genre")
print("="*50)

for genre, datasets in genre_groups.items():
    genre_df = df[df['Datasets'].isin(datasets)]
    if len(genre_df) == 0:
        continue
    
    z_scores = genre_df[key_features].apply(
        lambda x: (x - x.mean()) / x.std()
    )
    
    outliers = z_scores.abs() > 3
    outlier_datasets = genre_df.loc[outliers.any(axis=1), 'Datasets'].unique()
    
    if len(outlier_datasets) > 0:
        print(f"\nGenre: {genre}")
        print("Outlier datasets:", ", ".join(outlier_datasets))
        
        for dataset in outlier_datasets:
            idx = genre_df[genre_df['Datasets'] == dataset].index[0]
            extreme_features = z_scores.loc[idx].abs().nlargest(3)
            print(f"\n- {dataset}:")
            for feat, score in extreme_features.items():
                value = genre_df.loc[idx, feat]
                mean = genre_df[feat].mean()
                print(f"  {feat}: {value:.3f} (mean={mean:.3f}, z-score={score:.1f})")

report_data = []
for genre, datasets in genre_groups.items():
    genre_df = df[df['Datasets'].isin(datasets)]
    if len(genre_df) == 0:
        continue
        
    for feature in key_features:
        report_data.append({
            'genre': genre,
            'feature': feature,
            'mean': genre_df[feature].mean(),
            'std': genre_df[feature].std(),
            'median': genre_df[feature].median(),
            'distinctiveness': (genre_df[feature].mean() - df[feature].mean()) / df[feature].std()
        })

pd.DataFrame(report_data).to_csv('genre_characteristics_report.csv', index=False)
print("\nReport saved to 'genre_characteristics_report.csv'")