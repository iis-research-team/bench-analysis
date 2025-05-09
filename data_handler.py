import pandas as pd
from typing import Dict, Optional, List, Union


class TextAnalysisDataHandler:
    """Handles storage, normalization, and retrieval of text analysis results.

    Provides functionality to:
    - Save text analysis results to CSV (appending to existing files)
    - Normalize linguistic features by sentence or word counts
    - Load and manage analysis data
    """

    def __init__(self, filename: str = 'text_analysis_results.csv'):
        """Initialize the data handler with output file configuration.
        
        Args:
            filename: Path to CSV file for storing results (default: 'text_analysis_results.csv')
        """
        self.filename = filename
        self._initialize_normalization_columns()

    def _initialize_normalization_columns(self) -> None:
        """Initialize the columns that need normalization by sentences or words."""
        self.columns_to_normalize_by_sentences = [
            'dependency_ROOT', 'dependency_acl', 'dependency_acl:relcl', 'dependency_advcl',
            'dependency_advmod', 'dependency_amod', 'dependency_appos', 'dependency_aux',
            'dependency_aux:pass', 'dependency_case', 'dependency_cc', 'dependency_ccomp',
            'dependency_compound', 'dependency_conj', 'dependency_cop', 'dependency_csubj',
            'dependency_csubj:pass', 'dependency_dep', 'dependency_det', 'dependency_discourse',
            'dependency_expl', 'dependency_fixed', 'dependency_flat', 'dependency_flat:foreign',
            'dependency_flat:name', 'dependency_iobj', 'dependency_list', 'dependency_mark',
            'dependency_nmod', 'dependency_nsubj', 'dependency_nsubj:pass', 'dependency_nummod',
            'dependency_nummod:entity', 'dependency_nummod:gov', 'dependency_obj', 'dependency_obl',
            'dependency_obl:agent', 'dependency_orphan', 'dependency_parataxis', 'dependency_punct',
            'dependency_xcomp'
        ]
        
        self.columns_to_normalize_by_words = [
            'pos_ner_analysis_ner_LOC', 'pos_ner_analysis_ner_ORG', 'pos_ner_analysis_ner_PER',
            'pos_ner_analysis_pos_ADJ', 'pos_ner_analysis_pos_ADP', 'pos_ner_analysis_pos_ADV',
            'pos_ner_analysis_pos_AUX', 'pos_ner_analysis_pos_CCONJ', 'pos_ner_analysis_pos_DET',
            'pos_ner_analysis_pos_INTJ', 'pos_ner_analysis_pos_NOUN', 'pos_ner_analysis_pos_NUM',
            'pos_ner_analysis_pos_PART', 'pos_ner_analysis_pos_PRON', 'pos_ner_analysis_pos_PROPN',
            'pos_ner_analysis_pos_PUNCT', 'pos_ner_analysis_pos_SCONJ', 'pos_ner_analysis_pos_SPACE',
            'pos_ner_analysis_pos_SYM', 'pos_ner_analysis_pos_VERB', 'pos_ner_analysis_pos_X',

            'basic_statistics_n_chars', 'basic_statistics_n_complex_words',
            'basic_statistics_n_letters', 'basic_statistics_n_long_words',
            'basic_statistics_n_monosyllable_words', 'basic_statistics_n_polysyllable_words',
            'basic_statistics_n_punctuations', 'basic_statistics_n_simple_words',
            'basic_statistics_n_spaces', 'basic_statistics_n_syllables',
            'basic_statistics_n_unique_words'
        ]

    @staticmethod
    def flatten_dict(data: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten a nested dictionary structure while preserving meaningful keys.
        
        Args:
            data: Nested dictionary to flatten
            parent_key: Base key for nested structures (used internally)
            sep: Separator between key levels (default: '_')
            
        Returns:
            Dict: Flattened dictionary with composite keys
            
        Note:
            Skips nested dictionaries with only numeric keys to avoid over-flattening
        """
        items = []
        for key, value in data.items():
            if isinstance(value, dict) and all(isinstance(k, (int, float)) for k in value.keys()):
                continue

            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(TextAnalysisDataHandler.flatten_dict(value, new_key, sep).items())
            else:
                items.append((new_key, value))
        return dict(items)

    def save_analysis(self, analysis_data: Dict, dataset_name: Optional[str] = None) -> pd.DataFrame:
        """Save text analysis results to CSV, creating or appending to existing file.
        
        Args:
            analysis_data: Dictionary containing text analysis results
            dataset_name: Optional identifier for the dataset
            
        Returns:
            pd.DataFrame: The combined dataframe with all results (existing + new)
        """
        template = self._create_output_template(dataset_name)
        populated_data = self._populate_template(template, analysis_data)
        return self._save_to_csv(populated_data, dataset_name)

    def _create_output_template(self, dataset_name: Optional[str]) -> Dict:
        """Create a template dictionary with all possible analysis fields initialized to 0."""
        return {
            'dataset_name': dataset_name or 'unnamed_dataset',
            'basic_statistics_n_chars': 0,
            'basic_statistics_n_complex_words': 0,
            'basic_statistics_n_letters': 0,
            'basic_statistics_n_long_words': 0,
            'basic_statistics_n_monosyllable_words': 0,
            'basic_statistics_n_polysyllable_words': 0,
            'basic_statistics_n_punctuations': 0,
            'basic_statistics_n_sents': 0,
            'basic_statistics_n_simple_words': 0,
            'basic_statistics_n_spaces': 0,
            'basic_statistics_n_syllables': 0,
            'basic_statistics_n_unique_words': 0,
            'basic_statistics_n_words': 0,

            'pos_ner_analysis_ner_LOC': 0,
            'pos_ner_analysis_ner_ORG': 0,
            'pos_ner_analysis_ner_PER': 0,
            'pos_ner_analysis_pos_ADJ': 0,
            'pos_ner_analysis_pos_ADP': 0,
            'pos_ner_analysis_pos_ADV': 0,
            'pos_ner_analysis_pos_AUX': 0,
            'pos_ner_analysis_pos_CCONJ': 0,
            'pos_ner_analysis_pos_DET': 0,
            'pos_ner_analysis_pos_INTJ': 0,
            'pos_ner_analysis_pos_NOUN': 0,
            'pos_ner_analysis_pos_NUM': 0,
            'pos_ner_analysis_pos_PART': 0,
            'pos_ner_analysis_pos_PRON': 0,
            'pos_ner_analysis_pos_PROPN': 0,
            'pos_ner_analysis_pos_PUNCT': 0,
            'pos_ner_analysis_pos_SCONJ': 0,
            'pos_ner_analysis_pos_SPACE': 0,
            'pos_ner_analysis_pos_SYM': 0,
            'pos_ner_analysis_pos_VERB': 0,
            'pos_ner_analysis_pos_X': 0,

            'readability_statistics_automated_readability_index': 0,
            'readability_statistics_coleman_liau_index': 0,
            'readability_statistics_flesch_kincaid_grade': 0,
            'readability_statistics_flesch_reading_easy': 0,
            'readability_statistics_lix': 0,
            'readability_statistics_smog_index': 0,

            'sentiment_analysis_polarity': 0,
            'sentiment_analysis_subjectivity': 0,

            'syntactic_analysis_avg_tree_depth': 0,

            'text_homogeneity_avg_similarity': 0,

            'dependency_ROOT': 0,
            'dependency_acl': 0,
            'dependency_acl:relcl': 0,
            'dependency_advcl': 0,
            'dependency_advmod': 0,
            'dependency_amod': 0,
            'dependency_appos': 0,
            'dependency_aux': 0,
            'dependency_aux:pass': 0,
            'dependency_case': 0,
            'dependency_cc': 0,
            'dependency_ccomp': 0,
            'dependency_compound': 0,
            'dependency_conj': 0,
            'dependency_cop': 0,
            'dependency_csubj': 0,
            'dependency_csubj:pass': 0,
            'dependency_dep': 0,
            'dependency_det': 0,
            'dependency_discourse': 0,
            'dependency_expl': 0,
            'dependency_fixed': 0,
            'dependency_flat': 0,
            'dependency_flat:foreign': 0,
            'dependency_flat:name': 0,
            'dependency_iobj': 0,
            'dependency_list': 0,
            'dependency_mark': 0,
            'dependency_nmod': 0,
            'dependency_nsubj': 0,
            'dependency_nsubj:pass': 0,
            'dependency_nummod': 0,
            'dependency_nummod:entity': 0,
            'dependency_nummod:gov': 0,
            'dependency_obj': 0,
            'dependency_obl': 0,
            'dependency_obl:agent': 0,
            'dependency_orphan': 0,
            'dependency_parataxis': 0,
            'dependency_punct': 0,
            'dependency_xcomp': 0
        }

    def _populate_template(self, template: Dict, analysis_data: Dict) -> Dict:
        """Populate the template dictionary with actual analysis data."""
        flattened_data = self.flatten_dict(analysis_data)
        
        for key in template:
            if key == 'dataset_name':
                continue
            
            if key.startswith('basic_statistics_'):
                stat_key = key.replace('basic_statistics_', '')
                template[key] = analysis_data.get('basic_statistics', {}).get(stat_key, 0)
            elif key.startswith('dependency_'):
                dep_key = key.replace('dependency_', '')
                template[key] = analysis_data.get('syntactic_analysis', {}).get('dependency_stats', {}).get(dep_key, 0)
            elif key in flattened_data:
                template[key] = flattened_data[key]
                
        return template

    def _save_to_csv(self, data: Dict, dataset_name: Optional[str]) -> pd.DataFrame:
        """Handle the actual CSV saving operation."""
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
        """Normalize linguistic features by sentence or word counts.
        
        Args:
            input_filename: Source CSV file (default: self.filename)
            output_filename: Target CSV file for normalized data 
                           (default: input_filename with '_norm' suffix)
                           
        Returns:
            pd.DataFrame: The normalized dataframe
        """
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
        """Load analysis data from CSV file.
        
        Args:
            filename: CSV file to load (default: self.filename)
            
        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame
        """
        file_path = filename or self.filename
        return pd.read_csv(file_path)