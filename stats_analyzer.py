from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler

class StatisticalAnalyzer:
    """Performs statistical comparisons between reference datasets (master) and user-provided data.
    
    Key Features:
    - Standardizes data using z-score normalization
    - Supports Mahalanobis and Euclidean distance comparisons
    - Identifies statistically divergent features
    - Generates reports
    """

    def __init__(
        self, 
        master_df: pd.DataFrame, 
        user_df: pd.DataFrame, 
        feature_columns: List[str]
    ) -> None:
        """Initialize the analyzer with reference and user datasets.
        
        Args:
            master_df: Reference datasets containing groups for comparison
            user_df: Dataset to analyze (may contain multiple entries)
            feature_columns: Features used for statistical comparison
            
        Note:
            Standardizes master data during initialization for efficiency.
        """
        self.master_df = master_df.copy()
        self.user_df = user_df.copy()
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        
        self.master_scaled = self.master_df.copy()
        self.master_scaled[self.feature_columns] = self.scaler.fit_transform(
            self.master_df[self.feature_columns]
        )

    def _prepare_user_data(
        self, 
        dataset_name: Optional[str] = None, 
        name_column: str = "name"
    ) -> pd.DataFrame:
        """Prepare user data for analysis by standardizing and optionally filtering.
        
        Args:
            dataset_name: Specific dataset to analyze (None for aggregate)
            name_column: Column containing dataset identifiers in user_df
            
        Returns:
            Standardized user data (DataFrame)
            
        Raises:
            ValueError: If specified dataset_name is not found
        """
        if dataset_name is not None:
            user_data = self.user_df[self.user_df[name_column] == dataset_name]
            if len(user_data) == 0:
                raise ValueError(f"Dataset '{dataset_name}' not found in column '{name_column}'")
        else:
            user_data = self.user_df[self.feature_columns].mean().to_frame().T
        
        user_scaled = user_data.copy()
        user_scaled[self.feature_columns] = self.scaler.transform(
            user_data[self.feature_columns]
        )
        return user_scaled

    def _calculate_distance(
        self,
        user_mean: np.ndarray,
        group_mean: np.ndarray,
        group_cov: Optional[np.ndarray] = None,
        method: str = "mahalanobis"
    ) -> float:
        """Compute distance between user data and a reference group.
        
        Args:
            user_mean: Standardized mean vector of user data
            group_mean: Mean vector of reference group
            group_cov: Covariance matrix (required for Mahalanobis)
            method: "mahalanobis" (default) or "euclidean"
            
        Returns:
            Calculated distance metric (float)
            
        Note:
            Automatically falls back to Euclidean if Mahalanobis fails
        """
        if method == "mahalanobis" and group_cov is not None:
            try:
                inv_cov = np.linalg.inv(group_cov)
                return mahalanobis(user_mean, group_mean, inv_cov)
            except np.linalg.LinAlgError:
                pass  
        return np.linalg.norm(user_mean - group_mean)

    def _identify_key_differences(
        self,
        user_mean: np.ndarray,
        group_mean: np.ndarray,
        group_std: np.ndarray,
        top_n: int = 3
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Identify features contributing most to statistical differences.
        
        Args:
            user_mean: Standardized user feature means
            group_mean: Group feature means
            group_std: Group feature standard deviations
            top_n: Number of top features to return
            
        Returns:
            List of (feature, stats) tuples sorted by divergence
        """
        feature_diffs = []
        for i, feature in enumerate(self.feature_columns):
            z_diff = (user_mean[i] - group_mean[i]) / (group_std[i] + 1e-10)
            contribution = abs(z_diff) * group_std[i]
            feature_diffs.append((
                feature,
                {
                    'user_value': user_mean[i],
                    'group_mean': group_mean[i],
                    'z_score': z_diff,
                    'contribution': contribution
                }
            ))
        
        feature_diffs.sort(key=lambda x: -abs(x[1]['contribution']))
        return feature_diffs[:top_n]

    def compare_to_groups(
        self,
        group_column: str,
        dataset_name: Optional[str] = None,
        name_column: str = "name",
        comparison_method: str = "mahalanobis",
        top_n_features: int = 3
    ) -> Dict[str, Union[List[Tuple[str, float]], Dict[str, Dict]]]:
        """Compare user data to predefined groups in master data.
        
        Args:
            group_column: Column in master_df defining groups
            dataset_name: Specific dataset to analyze (None for aggregate)
            name_column: Column in user_df containing dataset identifiers
            comparison_method: "mahalanobis" (default) or "euclidean"
            top_n_features: Number of divergent features to highlight
            
        Returns:
            Dictionary containing:
            - similarity_scores: List of (group, score) tuples
            - comparison_details: Full statistical comparison per group
            - key_differences: Top divergent features per group
        """
        user_scaled = self._prepare_user_data(dataset_name, name_column)
        user_mean = user_scaled[self.feature_columns].mean().values
        
        results = {
            'similarity_scores': [],
            'comparison_details': {},
            'key_differences': {}
        }
        
        for group in self.master_scaled[group_column].unique():
            group_data = self.master_scaled[
                self.master_scaled[group_column] == group
            ][self.feature_columns]
            
            group_mean = group_data.mean().values
            group_std = group_data.std().values
            group_cov = group_data.cov().values
            
            distance = self._calculate_distance(
                user_mean, group_mean, group_cov, comparison_method
            )
            similarity_score = np.exp(-distance)
            
            results['similarity_scores'].append((group, similarity_score))
            results['comparison_details'][group] = {
                'similarity_score': similarity_score,
                'distance': distance,
                'feature_differences': {
                    feat: diff for feat, diff in self._identify_key_differences(
                        user_mean, group_mean, group_std, len(self.feature_columns))
                }
            }
            results['key_differences'][group] = self._identify_key_differences(
                user_mean, group_mean, group_std, top_n_features
            )
        
        results['similarity_scores'].sort(key=lambda x: -x[1])
        return results

    def general_comparison(
        self,
        dataset_name: Optional[str] = None,
        name_column: str = "dataset_name",
        top_n_features: int = 5
    ) -> Dict[str, Dict]:
        """Compare user data to overall master dataset statistics.
        
        Args:
            dataset_name: Specific dataset to analyze (None for aggregate)
            name_column: Column containing dataset identifiers
            top_n_features: Number of most divergent features to highlight
            
        Returns:
            Dictionary containing:
            - overall_stats: Master and user summary statistics
            - feature_comparison: Detailed comparison per feature
            - most_divergent_features: Top divergent features
        """
        user_scaled = self._prepare_user_data(dataset_name, name_column)
        
        master_stats = self.master_scaled[self.feature_columns].describe().loc[['mean', 'std']]
        
        if len(user_scaled) > 1:
            user_stats = user_scaled[self.feature_columns].describe().loc[['mean', 'std']]
        else:
            user_stats = pd.DataFrame({
                'mean': user_scaled[self.feature_columns].iloc[0],
                'std': np.zeros(len(self.feature_columns))
            }).T
        
        comparison = {}
        z_scores = []
        
        for feature in self.feature_columns:
            master_mean = master_stats.at['mean', feature]
            master_std = master_stats.at['std', feature]
            user_value = user_stats.at['mean', feature]
            
            z_score = (user_value - master_mean) / (master_std + 1e-10)
            similarity = 1 / (1 + abs(z_score))
            
            comparison[feature] = {
                'master_mean': master_mean,
                'master_std': master_std,
                'user_value': user_value,
                'z_score': z_score,
                'similarity': similarity
            }
            z_scores.append((feature, abs(z_score)))
        
        z_scores.sort(key=lambda x: -x[1])
        divergent_features = [
            (feature, comparison[feature]) 
            for feature, _ in z_scores[:top_n_features]
        ]
        
        return {
            'overall_stats': {
                'master': master_stats.to_dict(),
                'user': user_stats.to_dict()
            },
            'feature_comparison': comparison,
            'most_divergent_features': divergent_features
        }

    @staticmethod
    def print_comparison_report(
        report: Dict, 
        method: str = "group"
    ) -> None:
        """Print formatted statistical comparison report.
        
        Args:
            report: Result dictionary from compare_to_groups() or general_comparison()
            method: "group" or "general" to control report format
        """
        if method == "group":
            print("Top Statistical Matches")
            for group, score in report['similarity_scores']:
                print(f"{group}: {score:.1%}")
            
            if report['similarity_scores']:
                top_group = report['similarity_scores'][0][0]
                print(f"\nKey Differences vs {top_group} ")
                for feature, diff in report['key_differences'][top_group]:
                    print(
                        f"{feature}: "
                        f"User={diff['user_value']:.2f}, "
                        f"Group={diff['group_mean']:.2f} "
                        f"(Z={diff['z_score']:.2f})"
                    )
        
        elif method == "general":
            print("Overall Statistical Comparison")
            print("Master Data Means:", {
                k: v['mean'] 
                for k, v in report['overall_stats']['master'].items()
            })
            print("User Data Means:", {
                k: v['mean'] 
                for k, v in report['overall_stats']['user'].items()
            })
            
            print("\nMost Divergent Features")
            for feature, stats in report['most_divergent_features']:
                print(
                    f"{feature}: "
                    f"User={stats['user_value']:.2f} vs "
                    f"Master={stats['master_mean']:.2f}±{stats['master_std']:.2f} "
                    f"(Z={stats['z_score']:.2f}, Similarity={stats['similarity']:.1%})"
                )