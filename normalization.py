import pandas as pd

df = pd.read_csv('text_analysis_results.csv') 

columns_to_normalize_by_sentences = [
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

columns_to_normalize_by_words = [
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

for column in columns_to_normalize_by_sentences:
    if column in df.columns:
        df[column] = df[column] / df['basic_statistics_n_sents']

for column in columns_to_normalize_by_words:
    if column in df.columns:
        df[column] = df[column] / df['basic_statistics_n_words']

df.to_csv('normalized_text_analysis_results.csv', index=False)

