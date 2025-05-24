# Russian Language Benchmark Analysis Toolkit

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A toolkit for performing fine-grained analysis of Russian language benchmarks, providing detailed insights into dataset characteristics through statistical and linguistic analysis.

## Features

- **Data Preprocessing**: Clean and validate CSV data containing potential JSON artifacts
- **Text Analysis**: Comprehensive linguistic analysis including:
  - Basic statistics (word count, sentence length, etc.)
  - Readability metrics (Flesch-Kincaid, SMOG, etc.)
  - POS tagging and named entity recognition
  - Syntactic dependency analysis
  - Text homogeneity measurement
- **Data Management**: Store, normalize, and retrieve analysis results
- **Statistical Comparison**: Compare datasets using:
  - Mahalanobis and Euclidean distance metrics
  - Z-score normalization
  - Feature contribution analysis


# Installation and Usage Guide

## Install dependencies

```bash
pip install -r requirements.txt
```

## Download required language models

```bash
python -m spacy download ru_core_news_sm
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords
```
## Configuration Files

The system uses two JSON configuration files for data normalization:

- **`normalize_by_sentences.json`** – Contains column names to be normalized by sentence count.
- **`normalize_by_words.json`** – Contains column names to be normalized by word count.

Place these files in the same directory as the `data_handler` module.  
The `normalize_data()` method automatically uses these configurations.

## Usage

### 1. Data Preprocessing

```python
from data_processor import DataProcessor

processor = DataProcessor()
results = processor.process_csv("your_data.csv")
```

### 2. Text Analysis

```python
from text_analyzer import TextAnalyzer

analyzer = TextAnalyzer("processed data here")
results = analyzer.analyze()  # Returns comprehensive analysis
```

### 3. Data Management

```python
from data_handler import TextAnalysisDataHandler

handler = TextAnalysisDataHandler()
handler.save_analysis(results, "dataset_name")
normalized_data = handler.normalize_data()
```

### 4. Statistical Comparison

```python
from statistical_analyzer import StatisticalAnalyzer
import pandas as pd

master_df = pd.read_csv("master_data.csv")
user_df = pd.read_csv("user_data.csv")
features = ["feature1", "feature2", ...]  # Replace with actual feature names

analyzer = StatisticalAnalyzer(master_df, user_df, features)
report = analyzer.compare_to_groups("category_column")
StatisticalAnalyzer.print_comparison_report(report)
```
