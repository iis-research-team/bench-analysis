import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

class UniversalBenchmarkConverter:
    STANDARD_KEY_MAPPING = {
        "instruction": ["instruction", "task", "prompt"],
        "inputs": ["inputs", "input", "question", "premise", "sentence1", "text", "sentence2", "support_text", 'task', 'choices', 'additional_text'],
        "outputs": ["outputs", "output", "answers", "label"]
    }

    MERA_KEY_MAPPING = {
        "name": ["name"],
        "description": ["description"],
        "keywords": ["keywords"],
        "metrics": ["metrics"],
        "data": ["data"]
    }

    def __init__(self, data, is_mera=False, extract_fields=None, text_only=False):
        self.data = data
        self.is_mera = is_mera
        self.extract_fields = extract_fields
        self.text_only = text_only

    def _stringify_value(self, value):
        if isinstance(value, (list, tuple)):
            return " ".join(str(v) for v in value)
        elif isinstance(value, dict):
            return " ".join(str(v) for v in value.values())
        return str(value)

    def _convert_example(self, example_data):
        extracted_data = {}

        if 'instruction' in example_data and (not self.extract_fields or 'instruction' in self.extract_fields):
            extracted_data['instruction'] = self._stringify_value(example_data['instruction'])
        if 'inputs' in example_data and (not self.extract_fields or 'inputs' in self.extract_fields):
            extracted_data['inputs'] = self._stringify_value(example_data['inputs'])


        if 'inputs' in example_data:
            inputs = example_data['inputs']
            if isinstance(inputs, dict):
                for field in ['inputs', 'outputs','text', 'question', 'support_text', 'premise', 'choice1', 'choice2', 'hypothesis', 'option_a', 'option_b', 'option_c', 'option_d', 'query', 'replica', 'reply_1', 'reply_2',  'task', 'choices', 'additional_text']:
                    if field in inputs and (not self.extract_fields or field in self.extract_fields):
                        extracted_data[field] = self._stringify_value(inputs[field])

        if 'outputs' in example_data:
            outputs = example_data['outputs']
            if isinstance(outputs, dict):
                if not self.extract_fields or 'outputs' in self.extract_fields:
                    extracted_data['outputs'] = {k: self._stringify_value(v) for k, v in outputs.items()}
                else:
                    for field in self.extract_fields:
                        if field in outputs:
                            extracted_data[field] = self._stringify_value(outputs[field])

        if 'meta' in example_data and (not self.extract_fields or 'meta' in self.extract_fields):
            extracted_data['meta'] = example_data['meta']

        if self.text_only:
            return " ".join(str(v) for v in extracted_data.values()).strip()

        for key in extracted_data:
            if isinstance(extracted_data[key], str):
                extracted_data[key] = extracted_data[key].strip()

        return extracted_data

    def convert(self):
        if self.is_mera:
            raw_data = []
            for split in ['train', 'test', 'validation']:
                if split in self.data.get("data", {}):
                    raw_data.extend(self.data["data"][split])

            if self.text_only:
                return " ".join(self._convert_example(example) for example in raw_data)
            else:
                metadata = {k: self.data.get(k, "" if k != "keywords" else [])
                          for k in ["name", "description", "keywords", "metrics", "version"]}
                metadata["data"] = [self._convert_example(example) for example in raw_data]
                return metadata
        else:
            result = self._convert_example(self.data)
            return " ".join(result.values()) if self.text_only else result

class JSONLExtractor:
    def __init__(self, extract_fields=None, text_only=True):
        self.extract_fields = extract_fields
        self.text_only = text_only

    def _extract_fields(self, data_dict):
        if self.extract_fields is None:
            values = data_dict.values()
        else:
            values = []
            for field in self.extract_fields:
                if field in data_dict:
                    values.append(data_dict[field])
                elif '.' in field:
                    parts = field.split('.')
                    current = data_dict
                    try:
                        for part in parts:
                            current = current[part]
                        values.append(current)
                    except (KeyError, TypeError):
                        pass

        text = ' '.join(str(v).strip() for v in values)
        return text

    def process_line(self, line):
        data = json.loads(line)
        return self._extract_fields(data)

def mera(file_path, is_mera=True, extract_fields=None, text_only=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    converter = UniversalBenchmarkConverter(
        json_data,
        is_mera=is_mera,
        extract_fields=extract_fields,
        text_only=text_only
    )
    return converter.convert()

def rsg(file_paths, fields=None):
    extractor = JSONLExtractor(extract_fields=fields)
    all_extracted = []
    
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_extracted.extend(extractor.process_line(line) for line in f)
    
    return ' '.join(all_extracted) if extractor.text_only else all_extracted

def process_polyglot_data():
    splits = {'Russian': 'data/Russian-00000-of-00001-669ee41acad6c321.parquet'}
    df = pd.read_parquet("hf://datasets/Polyglot-or-Not/Fact-Completion/" + splits["Russian"])
    combined_string = ' '.join(df['stem'].astype(str) + ' ' + df['true'].astype(str) + ' ' + df['false'].astype(str))
    return combined_string

def process_xnli_data():
    ds = load_dataset("facebook/xnli", "ru")
    results = {}
    for split in ['train', 'test', 'validation']:
        if split in ds:
            data = ds[split]
            premises_hypotheses = data.map(lambda x: {
                'premise': x['premise'],
                'hypothesis': x['hypothesis']
            })
            df = premises_hypotheses.to_pandas()
            combined = ' '.join(f"{p} {h}" for p, h in zip(data['premise'], data['hypothesis']))
            results[split] = {'df': df, 'combined_string': combined[:500]}
    return results

def process_mmlu_data(dataset_name="alexandrainst/m_mmlu", subject="all"):
    try:
        ds = load_dataset(dataset_name, subject)
    except Exception as e:
        return None

    results = {}
    for split in ['train', 'test', 'validation']:
        if split in ds:
            data = ds[split]
            if dataset_name == "alexandrainst/m_mmlu":
                combined = ' '.join(f"{x['instruction']} {x['option_a']} {x['option_b']} {x['option_c']} {x['option_d']}" for x in tqdm(data, desc=f"Processing {split}"))
                df = data.map(lambda x: {'combined': f"{x['instruction']} {x['option_a']} {x['option_b']} {x['option_c']} {x['option_d']}"}).to_pandas()
            elif dataset_name == "NLPCoreTeam/mmlu_ru":
                combined = []
                for q, c in zip(data['question_ru'], data['choices_ru']):
                    combined.append(q + " " + " ".join(c))
                combined = " ".join(combined)
                df = pd.DataFrame({'question': data['question_ru'], 'choices': data['choices_ru']})
            results[split] = {'df': df, 'combined_string': combined[:500]}
    return results

def process_xcsr_data():
    ds = load_dataset("INK-USC/xcsr", "X-CODAH-ru")
    results = {}
    for split in ['train', 'test', 'validation']:
        if split in ds:
            data = ds[split]
            questions = []
            for item in tqdm(data, desc=f"Processing {split}"):
                stem = item['question']['stem']
                texts = item['question']['choices']['text']
                questions.append(stem + " " + " ".join(texts))
            combined = " ".join(questions)
            df = pd.DataFrame({'stem': [q['stem'] for q in data['question']], 'choices': [q['choices']['text'] for q in data['question']]})
            results[split] = {'df': df, 'combined_string': combined[:500]}
    return results

