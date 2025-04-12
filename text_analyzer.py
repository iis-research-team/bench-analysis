from collections import Counter
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pprint
import textstat
from ruts import BasicStats, ReadabilityStats
from textdistance import cosine
from langdetect import detect
from nltk import download
import spacy

download('punkt')
download('stopwords')
nlp_ru = spacy.load("ru_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")



from collections import Counter
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pprint
import textstat
from ruts import BasicStats, ReadabilityStats

class TextAnalyzer:
    def __init__(self, text: str, max_length: int = 100000):
        self.text = text
        self.max_length = max_length
        self.parts = self._split_text()
        self.lang = detect(text[:1000])  
        self.nlp = nlp_ru if self.lang == "ru" else nlp_en
        self._cached_docs = {} 
        self._full_doc = None  

    def _split_text(self) -> List[str]:
        if len(self.text) <= self.max_length:
            return [self.text]

        parts = []
        start = 0
        text_length = len(self.text)

        while start < text_length:
            end = min(start + self.max_length, text_length)

            sentence_end = self.text.rfind('.', start, end)
            if sentence_end > start and (end - sentence_end) < (self.max_length * 0.2):
                end = sentence_end + 1  
            parts.append(self.text[start:end])
            start = end

        return parts

    def _get_doc(self, text: str) -> Any:
        if text not in self._cached_docs:
            self._cached_docs[text] = self.nlp(text)
        return self._cached_docs[text]

    def _get_full_doc(self) -> Any:
        if len(self.text) <= self.max_length and self._full_doc is None:
            self._full_doc = self.nlp(self.text)
        return self._full_doc

    def basic_statistics(self) -> Dict[str, Union[int, float]]:
        if self.lang == "ru":
            bs = BasicStats(self.text)
            stats = bs.get_stats()
            bs.print_stats()
            return stats

        full_doc = self._get_full_doc()
        if full_doc is not None:
            words = [token.text for token in full_doc if not token.is_punct]
            word_count = len(words)
            avg_word_len = sum(len(word) for word in words) / word_count if word_count else 0
            sentence_count = len(list(full_doc.sents))
        else:
            word_count = textstat.lexicon_count(self.text, removepunct=True)
            avg_word_len = sum(len(part.split()) for part in self.parts) / word_count if word_count else 0
            sentence_count = sum(textstat.sentence_count(part) for part in self.parts)

        stats = {
            "char_count": len(self.text),
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_len,
        }
        pprint.pprint(stats)
        return stats

    def readability_statistics(self) -> Dict[str, float]:
        if self.lang == "ru":
            rs = ReadabilityStats(self.text)
            stats = rs.get_stats()
            rs.print_stats()
            return stats

        if len(self.parts) == 1:
            stats = {
                "flesch_reading_ease": textstat.flesch_reading_ease(self.text),
                "gunning_fog": textstat.gunning_fog(self.text),
                "smog_index": textstat.smog_index(self.text),
                "automated_readability_index": textstat.automated_readability_index(self.text),
                "coleman_liau_index": textstat.coleman_liau_index(self.text),
            }
        else:
            metrics = []
            for part in self.parts:
                metrics.append({
                    "flesch": textstat.flesch_reading_ease(part),
                    "gunning": textstat.gunning_fog(part),
                    "smog": textstat.smog_index(part),
                    "automated": textstat.automated_readability_index(part),
                    "coleman": textstat.coleman_liau_index(part),
                })

            stats = {
                "flesch_reading_ease": np.mean([m["flesch"] for m in metrics]),
                "gunning_fog": np.mean([m["gunning"] for m in metrics]),
                "smog_index": np.mean([m["smog"] for m in metrics]),
                "automated_readability_index": np.mean([m["automated"] for m in metrics]),
                "coleman_liau_index": np.mean([m["coleman"] for m in metrics]),
            }

        pprint.pprint(stats)
        return stats

    def pos_ner_analysis(self) -> Dict[str, Dict[str, int]]:
        pos_counter = Counter()
        ner_counter = Counter()

        for part in self.parts:
            doc = self._get_doc(part)
            pos_counter.update(token.pos_ for token in doc)
            ner_counter.update(ent.label_ for ent in doc.ents)

        return {"pos": dict(pos_counter), "ner": dict(ner_counter)}

    def syntactic_analysis(self) -> Dict[str, Any]:
        dep_counter = Counter()
        tree_depths = []

        for part in self.parts:
            doc = self._get_doc(part)
            dep_counter.update(token.dep_ for token in doc)

            for sent in doc.sents:
                if len(sent) > 1: 
                    depths = [abs(token.i - token.head.i) for token in sent]
                    tree_depths.append(max(depths) if depths else 0)

        avg_depth = np.mean(tree_depths) if tree_depths else 0
        return {
            "dependency_stats": dict(dep_counter),
            "avg_tree_depth": float(avg_depth)
        }


    """ def sentiment_analysis(self) -> Dict[str, float]:
        polarities = []
        subjectivities = []

        for part in self.parts:
            doc = self._get_doc(part)
            if hasattr(doc._, "blob"):
                polarities.append(doc._.blob.polarity)
                subjectivities.append(doc._.blob.subjectivity)

        return {
            "polarity": np.mean(polarities) if polarities else 0.0,
            "subjectivity": np.mean(subjectivities) if subjectivities else 0.0
        } """

    def text_homogeneity(self) -> Optional[Dict[str, float]]:
        all_sentences = []
        for part in self.parts:
            doc = self._get_doc(part)
            all_sentences.extend(sent.text for sent in doc.sents)

        if len(all_sentences) < 2:
            return None

        similarities = []
        for i in range(len(all_sentences) - 1):
            try:
                sim = cosine(all_sentences[i], all_sentences[i + 1])
                similarities.append(sim)
            except:
                continue  
        if not similarities:
            return None

        return {"avg_similarity": float(np.mean(similarities))}

    def analyze(self) -> Dict[str, Any]:
        """Run all analyses and return combined results."""
        return {
            "basic_statistics": self.basic_statistics(),
            "readability_statistics": self.readability_statistics(),
            "pos_ner_analysis": self.pos_ner_analysis(),
            "syntactic_analysis": self.syntactic_analysis(),
            #"sentiment_analysis": self.sentiment_analysis(),
            "text_homogeneity": self.text_homogeneity(),
        }