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


class TextAnalyzer:
    """Analyzes text for various linguistic and statistical properties.

    Provides functionality to:
    - Calculate basic text statistics (word count, sentence length, etc.)
    - Measure readability using multiple metrics
    - Perform POS tagging and named entity recognition
    - Analyze syntactic dependencies
    - Calculate sentiment scores
    - Measure text homogeneity
    """

    def __init__(self, text: str, max_length: int = 100000):
        """Initialize the TextAnalyzer with text and processing parameters.

        Args:
            text: The text to analyze
            max_length: Maximum length for text chunks during processing (default: 100000 chars)

        Note:
            Automatically detects language and configures appropriate NLP pipeline.
        """
        self.text = text
        self.max_length = max_length
        self.parts = self._split_text()
        self.lang = self._detect_language()
        self.nlp = self._get_nlp_pipeline()
        self._cached_docs = {}  
        self._full_doc = None  

    def _detect_language(self) -> str:
        """Detect language from the first portion of text.

        Returns:
            str: Detected language code ('ru' for Russian, others default to English)
        """
        return detect(self.text[:1000])  

    def _get_nlp_pipeline(self):
        """Get the appropriate NLP pipeline based on detected language.

        Returns:
            spaCy language pipeline
        """
        return nlp_ru if self.lang == "ru" else nlp_en

    def _split_text(self) -> List[str]:
        """Split text into manageable chunks respecting sentence boundaries.

        Returns:
            List[str]: Text chunks each within max_length limit

        Note:
            Uses spaCy for sentence splitting when possible, with fallback to simple splitting.
            Preserves sentence boundaries to maintain linguistic integrity.
        """
        if len(self.text) <= self.max_length:
            return [self.text]

        sentences = []

        try:
            if not hasattr(self, 'nlp'):
                self.nlp = self._get_nlp_pipeline()

            original_max_length = self.nlp.max_length
            self.nlp.max_length = max(len(self.text), self.nlp.max_length)

            doc = self.nlp(self.text)
            sentences = [sent.text for sent in doc.sents]

        except (ValueError, MemoryError, AttributeError) as e:
            sentences = self._simple_text_split()
            
        finally:
            if original_max_length is not None:
                self.nlp.max_length = original_max_length

        return self._chunk_sentences(sentences)

    def _simple_text_split(self) -> List[str]:
        """Simple period-based text splitting fallback.

        Returns:
            List[str]: Sentences split on periods
        """
        sentences = []
        start = 0
        while start < len(self.text):
            end = min(start + self.max_length, len(self.text))
            sentence_end = self.text.rfind('.', start, end)
            if sentence_end > start:
                end = sentence_end + 1
            sentences.append(self.text[start:end])
            start = end
        return sentences

    def _chunk_sentences(self, sentences: List[str]) -> List[str]:
        """Combine sentences into chunks respecting max_length.

        Args:
            sentences: List of individual sentences

        Returns:
            List[str]: Properly sized text chunks
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_length + sent_len > self.max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sent)
            current_length += sent_len

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _get_doc(self, text: str) -> Any:
        """Get or create a cached NLP doc for the given text.

        Args:
            text: Text to process

        Returns:
            Processed spaCy Doc object
        """
        if text not in self._cached_docs:
            self._cached_docs[text] = self.nlp(text)
        return self._cached_docs[text]

    def _get_full_doc(self) -> Any:
        """Get the full document if small enough for processing.

        Returns:
            Processed spaCy Doc object or None if text is too large
        """
        if len(self.text) <= self.max_length and self._full_doc is None:
            self._full_doc = self.nlp(self.text)
        return self._full_doc

    def basic_statistics(self) -> Dict[str, Union[int, float]]:
        """Calculate basic text statistics including counts and averages.

        Returns:
            Dict[str, Union[int, float]]: Statistics including:
                - char_count: Total characters
                - word_count: Total words
                - sentence_count: Total sentences
                - avg_word_length: Average word length

        Note:
            Uses Russian-specific implementation when language is Russian.
        """
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
        """Calculate various readability metrics for the text.

        Returns:
            Dict[str, float]: Readability metrics including:
                - flesch_reading_ease
                - gunning_fog
                - smog_index
                - automated_readability_index
                - coleman_liau_index

        Note:
            For large texts, calculates metrics per chunk and returns averages.
        """
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
        """Perform part-of-speech tagging and named entity recognition.

        Returns:
            Dict[str, Dict[str, int]]: Counts of:
                - pos: Part-of-speech tags
                - ner: Named entity labels
        """
        pos_counter = Counter()
        ner_counter = Counter()

        for part in self.parts:
            doc = self._get_doc(part)
            pos_counter.update(token.pos_ for token in doc)
            ner_counter.update(ent.label_ for ent in doc.ents)

        return {"pos": dict(pos_counter), "ner": dict(ner_counter)}

    def syntactic_analysis(self) -> Dict[str, Any]:
        """Analyze syntactic dependencies and tree structure.

        Returns:
            Dict[str, Any]: Results including:
                - dependency_stats: Counts of dependency relations
                - avg_tree_depth: Average syntactic tree depth
        """
        dep_counter = Counter()
        tree_depths = []

        for part in self.parts:
            doc = self._get_doc(part)
            dep_counter.update(token.dep_ for token in doc)

            for sent in doc.sents:
                if len(sent) > 1:  
                    # Calculate the linear distance between each token and its head (dependency parent)
                    depths = [abs(token.i - token.head.i) for token in sent]
                    # Record the max depth (longest dependency arc) in the sentence
                    tree_depths.append(max(depths) if depths else 0)

        avg_depth = np.mean(tree_depths) if tree_depths else 0
        return {
            "dependency_stats": dict(dep_counter),
            "avg_tree_depth": float(avg_depth)
        }

    
    def text_homogeneity(self) -> Optional[Dict[str, float]]:
        """Measure text homogeneity using sentence similarity.

        Returns:
            Optional[Dict[str, float]]: Average similarity between adjacent sentences,
            or None if not enough sentences exist.
        """
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
        """Run all available analyses and return comprehensive results.

        Returns:
            Dict[str, Any]: Combined results from all analysis methods.
        """
        return {
            "basic_statistics": self.basic_statistics(),
            "readability_statistics": self.readability_statistics(),
            "pos_ner_analysis": self.pos_ner_analysis(),
            "syntactic_analysis": self.syntactic_analysis(),
            "text_homogeneity": self.text_homogeneity(),
        }

