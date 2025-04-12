import pandas as pd


data = pd.read_csv('mera_data.csv', index_col=0)

model_metrics = data.iloc[:64, :]  # Rows 1-64 are model metrics
dataset_characteristics = data.iloc[64:, :]

model_metrics_transposed = model_metrics.transpose()
dataset_characteristics_transposed = dataset_characteristics.transpose()
combined_data = pd.concat([model_metrics_transposed, dataset_characteristics_transposed], axis=1)

correlation_matrix = combined_data.corr()
correlation_subset = correlation_matrix.iloc[:64, 64:]
correlation_subset.to_csv('correlation_mera.csv')

average_correlations = correlation_subset.mean(axis=0)
average_correlations_df = average_correlations.to_frame(name='Average Correlation')
average_correlations_df.to_csv('average_correlations_mera.csv')

sorted_average_correlations = average_correlations_df.sort_values(by='Average Correlation', ascending=False)
top_10_positive = sorted_average_correlations.head(10)
top_10_negative = sorted_average_correlations.tail(10)

top_10_positive.to_csv('top_10_positive_correlations_mera.csv')
top_10_negative.to_csv('top_10_negative_correlations_mera.csv')