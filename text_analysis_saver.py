import pandas as pd

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        if isinstance(v, dict) and all(isinstance(key, (int, float)) for key in v.keys()):
            continue
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def save_analysis_to_file(data, dataset_name=None, filename='text_analysis_results.csv'):
    all_fields = {
        'dataset_name': dataset_name if dataset_name else 'unnamed_dataset',
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

    flattened_data = flatten_dict(data)

    for key in all_fields:
        if key == 'dataset_name':
            continue
        if key.replace('basic_statistics_', '') in data.get('basic_statistics', {}):
            all_fields[key] = data['basic_statistics'].get(key.replace('basic_statistics_', ''), 0)
        elif key in flattened_data:
            all_fields[key] = flattened_data[key]
        elif key.startswith('dependency_'):
            dep_key = key.replace('dependency_', '')
            all_fields[key] = data.get('syntactic_analysis', {}).get('dependency_stats', {}).get(dep_key, 0)

    df = pd.DataFrame([all_fields])

    try:
        existing_df = pd.read_csv(filename)
        if 'dataset_name' not in existing_df.columns:
            existing_df['dataset_name'] = 'unnamed_dataset'
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass

    cols = ['dataset_name'] + [col for col in df.columns if col != 'dataset_name']
    df = df[cols]

    df.to_csv(filename, index=False)
    print(f"Results saved to {filename} (Dataset: '{dataset_name if dataset_name else 'unnamed_dataset'}')")