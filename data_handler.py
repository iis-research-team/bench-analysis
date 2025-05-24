import pandas as pd
import json
from pathlib import Path
import os
from typing import Dict, Optional


class TextAnalysisDataHandler:
    """Handles storage, normalization, and retrieval of text analysis results."""

    def __init__(self, filename: str = 'text_analysis_results.csv'):
        """Initialize the data handler with output file configuration."""
        self.filename = filename
        self._load_normalization_columns()

    def _get_config_path(self) -> Path:
        """Get the path to the directory containing config JSON files."""
        try:
            return Path(__file__).parent
        except NameError:
            return Path(os.getcwd())

    def _load_normalization_columns(self) -> None:
        """Load columns that need normalization from JSON files."""
        config_path = self._get_config_path()
        
        sentences_file = config_path / 'normalize_by_sentences.json'
        words_file = config_path / 'normalize_by_words.json'

        if not sentences_file.exists() or not words_file.exists():
            raise FileNotFoundError(
                f"Normalization config files not found in {config_path}. "
                "Please ensure both 'normalize_by_sentences.json' and "
                "'normalize_by_words.json' exist in this directory."
            )

        with open(sentences_file, 'r') as f:
            self.columns_to_normalize_by_sentences = json.load(f)
        
        with open(words_file, 'r') as f:
            self.columns_to_normalize_by_words = json.load(f)

    @staticmethod
    def flatten_dict(data: Dict, sep: str = '_') -> Dict:
        """Flatten a nested dictionary structure using pandas json_normalize."""
        df = pd.json_normalize(data, sep=sep)
        return df.to_dict(orient='records')[0]

    def save_analysis(self, analysis_data: Dict, dataset_name: Optional[str] = None) -> pd.DataFrame:
        """Save text analysis results to CSV."""
        template = self._create_output_template(dataset_name)
        populated_data = self._populate_template(template, analysis_data)
        return self._save_to_csv(populated_data, dataset_name)

    def _create_output_template(self, dataset_name: Optional[str]) -> Dict:
        """Create a template dictionary with all possible analysis fields."""
        template = {
            'dataset_name': dataset_name or 'unnamed_dataset',
            'basic_statistics_n_sents': 0,
            'basic_statistics_n_words': 0
        }
        
        for col in self.columns_to_normalize_by_sentences + self.columns_to_normalize_by_words:
            template[col] = 0
            
        template.update({
            'readability_statistics_automated_readability_index': 0,
            'readability_statistics_coleman_liau_index': 0,
            'readability_statistics_flesch_kincaid_grade': 0,
            'readability_statistics_flesch_reading_easy': 0,
            'readability_statistics_lix': 0,
            'readability_statistics_smog_index': 0,
            'sentiment_analysis_polarity': 0,
            'sentiment_analysis_subjectivity': 0,
            'syntactic_analysis_avg_tree_depth': 0,
            'text_homogeneity_avg_similarity': 0
        })
        
        return template

    def _populate_template(self, template: Dict, analysis_data: Dict) -> Dict:
        """Populate the template dictionary with actual analysis data."""
        flattened_data = self.flatten_dict(analysis_data)
        
        for key in template:
            if key == 'dataset_name':
                continue
            if key in flattened_data:
                template[key] = flattened_data[key]
                
        return template

    def _save_to_csv(self, data: Dict, dataset_name: Optional[str]) -> pd.DataFrame:
        """Save data to CSV, appending if file exists."""
        new_df = pd.DataFrame([data])
        
        try:
            existing_df = pd.read_csv(self.filename)
            if 'dataset_name' not in existing_df.columns:
                existing_df['dataset_name'] = 'unnamed_dataset'
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except FileNotFoundError:
            combined_df = new_df
        
        columns = ['dataset_name'] + [col for col in combined_df.columns if col != 'dataset_name']
        combined_df = combined_df[columns]
        
        combined_df.to_csv(self.filename, index=False)
        print(f"Results saved to {self.filename} (Dataset: '{dataset_name or 'unnamed_dataset'}')")
        return combined_df

    def normalize_data(self, input_filename: Optional[str] = None, 
                      output_filename: Optional[str] = None) -> pd.DataFrame:
        """Normalize linguistic features by sentence or word counts."""
        input_path = input_filename or self.filename
        output_path = output_filename or input_path.replace('.csv', '_norm.csv')
        
        df = pd.read_csv(input_path)
        
        if 'basic_statistics_n_sents' in df.columns:
            for col in self.columns_to_normalize_by_sentences:
                if col in df.columns:
                    df[col] = df[col] / df['basic_statistics_n_sents']
        
        if 'basic_statistics_n_words' in df.columns:
            for col in self.columns_to_normalize_by_words:
                if col in df.columns:
                    df[col] = df[col] / df['basic_statistics_n_words']
        
        df.to_csv(output_path, index=False)
        print(f"Normalized data saved to {output_path}")
        return df

    def load_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """Load analysis data from CSV file."""
        file_path = filename or self.filename
        return pd.read_csv(file_path)