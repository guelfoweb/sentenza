# Gianni Amato
# Link: https://github.com/guelfoweb/sentenza/

import re
from typing import List, Dict, Set, Optional, Generator, Tuple
from collections import Counter
import itertools
import matplotlib.pyplot as plt

class Tokenizer:
    """
    A Python library for extracting and processing sentences from text with 
    statistical chunking capabilities.
    """
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True, 
                stopwords: Optional[Set[str]] = None, min_length: int = 2,
                chunk_size: int = 10000):
        """
        Initialize tokenizer with performance optimizations for long texts.
        
        Args:
            lowercase: Convert all text to lowercase
            remove_punctuation: Remove punctuation
            stopwords: Set of stopwords to remove (optional)
            min_length: Minimum length of tokens to keep
            chunk_size: Size of text chunks for processing (affects memory usage)
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.stopwords = stopwords or set()
        self.min_length = min_length
        self.chunk_size = chunk_size
        
        # Precompile regex patterns for better performance
        self.word_pattern = re.compile(r'\b\w+\b')
        self.punct_pattern = re.compile(r'[^\w\s]')
        
        # More robust sentence splitting pattern
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply preprocessing steps to the text.
        
        Args:
            text: The text to preprocess
            
        Returns:
            Preprocessed text
        """
        if self.lowercase:
            text = text.lower()
            
        if self.remove_punctuation:
            text = self.punct_pattern.sub(' ', text)
            
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens applying normalization options.
        For shorter texts that can fit in memory.
        
        Args:
            text: The text to tokenize
            
        Returns:
            List of extracted tokens
        """
        if not text:
            return []
            
        text = self.preprocess_text(text)
        tokens = self.word_pattern.findall(text)
        
        # Filter to remove stopwords and tokens that are too short
        tokens = [token for token in tokens 
                 if len(token) >= self.min_length and token not in self.stopwords]
        
        return tokens
    
    def tokenize_stream(self, text: str) -> Generator[str, None, None]:
        """
        Stream tokens from text without loading all tokens into memory at once.
        Ideal for very long texts.
        
        Args:
            text: The text to tokenize
            
        Yields:
            Tokens one by one
        """
        if not text:
            return
            
        # Process text in chunks to limit memory usage
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            processed_chunk = self.preprocess_text(chunk)
            
            for token in self.word_pattern.findall(processed_chunk):
                if len(token) >= self.min_length and token not in self.stopwords:
                    yield token
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        For shorter texts that can fit in memory.
        
        Args:
            text: The text to split into sentences
            
        Returns:
            List of sentences
        """
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def preprocess_for_sentence_splitting(self, text):
        """
        Preprocess text to handle abbreviations before sentence splitting.
        """
        
        # replace the dot in common abbreviations
        common_abbr = {
            'prof.': 'prof@POINT@',  # Professore
            'dott.': 'dott@POINT@',  # Dottore
            'sig.': 'sig@POINT@',    # Signore
            'ing.': 'ing@POINT@',    # Ingegnere
            'avv.': 'avv@POINT@',    # Avvocato
            'dr.': 'dr@POINT@',      # Doctor
            'mr.': 'mr@POINT@',      # Mister
            'mrs.': 'mrs@POINT@',    # Misses
            'ms.': 'ms@POINT@',      # Miss
            'eng.': 'eng@POINT@',    # Engineer
            'esq.': 'esq@POINT@',    # Esquire
            'rev.': 'rev@POINT@',    # Reverend
        } # todo

        
        for abbr, replacement in common_abbr.items():
            text = text.replace(abbr, replacement)
        
        return text
    
    def sentences_stream(self, text: str) -> Generator[str, None, None]:
        """
        Stream sentences from text without loading all sentences into memory.
        Ideal for very long texts.
        
        Args:
            text: The text to process
            
        Yields:
            Sentences one by one
        """
        text = self.preprocess_for_sentence_splitting(text)

        buffer = ""
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            buffer += chunk
            
            # Find sentence endings in the buffer
            sentences = self.sentence_pattern.split(buffer)
            
            # Yield all sentences except the last one (might be incomplete)
            if len(sentences) > 1:
                for sentence in sentences[:-1]:
                    sentence = ' '.join(sentence.split()) # remove \s, \n, \t
                    sentence = sentence.replace('@POINT@', '.')
                    if sentence.strip():
                        yield sentence.strip()
                
                # Keep the last sentence in the buffer
                buffer = sentences[-1]
        
        # Yield the last sentence if it's not empty
        if buffer.strip():
            buffer = ' '.join(buffer.split()) 
            buffer = buffer.replace('@POINT@', '.')
            yield buffer.strip()
    
    def count_tokens(self, tokens: list, sort_by: str = None, reverse: bool = False, 
                    streaming: bool = True) -> Dict[str, int]:
        """
        Count the frequency of each token in the text with sorting options.
        Uses streaming for large texts.
        
        Args:
            text: The text to analyze
            sort_by: Sorting criterion ('count' for frequency, 'alpha' for alphabetical, None for no sorting)
            reverse: If True, reverses the order (descending for count, Z-A for alpha)
            streaming: Whether to use memory-efficient streaming (for very long texts)
            
        Returns:
            Dictionary with tokens and their frequencies, sorted according to the specified criterion
        """

        counts = Counter(tokens)
        
        # If no sorting is requested, return the Counter as a dictionary
        if sort_by is None:
            return dict(counts)
            
        # Sort based on the specified criterion
        if sort_by.lower() == 'count':
            # Sort by count
            sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=reverse)
            return dict(sorted_items)
        elif sort_by.lower() == 'alpha':
            # Sort alphabetically
            sorted_items = sorted(counts.items(), key=lambda x: x[0], reverse=reverse)
            return dict(sorted_items)
        else:
            # Invalid criterion, return unsorted
            return dict(counts)
 
    def statistics(self, sentences: list, get_sentence_lengths=False):
        """
        Calculate statistics from a list of sentences.
        
        Args:
            sentences: List of sentences
            get_sentence_lengths: Whether to include sentence lengths in the result
            
        Returns:
            Dictionary with statistics
        """
        sentence_lengths = [len(s) for s in sentences]
        mean_length = sum(sentence_lengths) / len(sentence_lengths)
        
        # Standard deviation: https://en.wikipedia.org/wiki/Standard_deviation
        """
        Is a measure that indicates how far the values in a data set are scattered from the mean.
        
        Imagine two classes of students who took a test with grades from 0 to 10:

        Class A: [5, 5, 5, 5, 5, 5, 5, 5]
        Class B: [1, 2, 4, 5, 5, 6, 8, 9]
        
        Both classes have the same average: 5.
        
        But there is an important difference:

        In Class A, all students got 5. There is no variation in the grades.
        In Class B, the grades are very variable, from 1 to 9.

        The standard deviation measures precisely this difference:

        Class A: standard deviation = 0 (no variation)
        Class B: standard deviation = 2.83 (high variation)
        """
        variance = sum((x - mean_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)
        std_dev = variance ** 0.5
        max_length = max(sentence_lengths)
        min_length = min(sentence_lengths)
        freq_sentences = Counter(sentence_lengths)
        
        # A common rule of thumb is to use the mean plus 2 or 3 times the standard deviation
        # This will capture about 95-99% of the sentences in a chunk
        chunk_size = int(mean_length + 3 * std_dev)
        
        # The overlap can be set as a percentage of the chunk_size
        # or based on the standard deviation
        chunk_overlap = int(std_dev * 2) # 2 times the standard deviation
        
        # Calculate estimated number of chunks needed
        total_text_length = sum(sentence_lengths)
        effective_chunk_size = chunk_size - chunk_overlap  # Account for overlap
        
        # If chunk_size <= chunk_overlap, we need at least one chunk per sentence
        if effective_chunk_size <= 0:
            num_chunks = len(sentences)
        else:
            # Formula: total_length / (chunk_size - overlap)
            # Round up to ensure all text is covered
            num_chunks = (total_text_length + effective_chunk_size - 1) // effective_chunk_size
        
        stats = {
            "sentences": len(sentences),
            "mean_length": mean_length,
            "std_dev": std_dev,
            "max_length": max_length,
            "min_length": min_length,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_text_length": total_text_length,
            "estimated_chunks": num_chunks
            }
        
        # values rounded to two decimal places
        stats = {k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}
        
        if get_sentence_lengths:
            stats.update({"sentence_lengths": sentence_lengths})
            
        return stats

    def plotting(self, sentences: list, filename: str = 'histogram_with_chunks.png'):
        """
        Create a visualization of sentence length distribution with chunk statistics.
        
        Args:
            sentences: List of sentences
            filename: Optional output filename for the plot (default: 'histogram_with_chunks.png')
        """
        stats = self.statistics(sentences, get_sentence_lengths=True)
        sentence_lengths = stats['sentence_lengths']
        mean_length = stats['mean_length']
        std_dev = stats['std_dev']
        chunk_size = stats['chunk_size']
        chunk_overlap = stats['chunk_overlap']
        estimated_chunks = stats['estimated_chunks']
        
        plt.figure(figsize=(12, 6))
        
        # Basic histogram
        n, bins, patches = plt.hist(sentence_lengths, bins=20, alpha=0.7, color='skyblue', 
                                    edgecolor='black', label='Sentence Lengths')
        
        # Text stats (Title, X, Y)
        plt.title(f'Distribution of {len(sentences)} Sentence Lengths\nMean: {mean_length:.2f}, Std Dev: {std_dev:.2f}, Num Chunks: {estimated_chunks}')
        plt.xlabel('Sentence Length (characters)')
        plt.ylabel('Frequency (sentences)')
        plt.grid(axis='y', alpha=0.75)
        
        # Vertical lines for mean and standard deviation
        plt.axvline(x=mean_length, color='purple', linestyle='-', linewidth=1, 
                    label=f'Mean: {mean_length:.2f}')
        plt.axvline(x=mean_length + std_dev, color='purple', linestyle=':', linewidth=1, 
                    label=f'+1 Std Dev: {mean_length + std_dev:.2f}')
        plt.axvline(x=mean_length - std_dev, color='purple', linestyle=':', linewidth=1, 
                    label=f'-1 Std Dev: {mean_length - std_dev:.2f}')
        
        plt.axvline(x=chunk_size, color='red', linestyle='--', 
                    linewidth=2, label=f'Chunk Size: {chunk_size}')
            
        plt.axvline(x=chunk_overlap, color='green', linestyle='--', 
                    linewidth=2, label=f'Chunk Overlap: {chunk_overlap}')
        
        # Coloring areas
        plt.axvspan(0, chunk_overlap, alpha=0.2, color='green', label='Overlap Area')
        plt.axvspan(chunk_overlap, chunk_size, alpha=0.2, color='red', label='Chunk Size Area')
        
        # Legenda
        plt.legend(loc='upper right')
        
        # Save file with specified filename
        plt.savefig(filename)
        plt.close()
        
    def semantic_chunks(self, sentences: list, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """
        Create semantic chunks from sentences with appropriate overlap.
        
        Args:
            sentences: List of sentences to chunk
            chunk_size: Maximum size of each chunk in characters (optional, will use statistics if None)
            chunk_overlap: Size of overlap between chunks in characters (optional, will use statistics if None)
            
        Returns:
            List of semantic chunks with appropriate overlap
        """
        # If chunk_size or chunk_overlap not provided, calculate from statistics
        if chunk_size is None or chunk_overlap is None:
            stats = self.statistics(sentences)
            if chunk_size is None:
                chunk_size = stats['chunk_size']
            if chunk_overlap is None:
                chunk_overlap = stats['chunk_overlap']
                
        # Create semantic chunks with the specified overlap
        semantic_chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed the chunk size,
            # save the current chunk and start a new one
            if current_length + len(sentence) > chunk_size and current_chunk:
                semantic_chunks.append(current_chunk)
                
                # Keep a portion for overlap
                overlap_text = ""
                overlap_length = 0
                words = current_chunk.split()
                
                # Build overlap from the end of the previous chunk
                while words and overlap_length < chunk_overlap:
                    word = words.pop()
                    overlap_text = word + " " + overlap_text
                    overlap_length += len(word) + 1
                
                current_chunk = overlap_text + sentence
                current_length = len(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_length += len(sentence) + (1 if current_chunk else 0)
        
        # Add the last chunk if there is one
        if current_chunk:
            semantic_chunks.append(current_chunk)
            
        return semantic_chunks

    def add_stopwords(self, new_stopwords: List[str]):
        """
        Add new stopwords to the existing set.
        
        Args:
            new_stopwords: List of new stopwords to add
        """
        if self.lowercase:
            new_stopwords = [word.lower() for word in new_stopwords]
        
        self.stopwords.update(new_stopwords)


# Predefined stopwords sets
def get_english_stopwords() -> Set[str]:
    """
    Returns a set of common English stopwords.
    
    Returns:
        Set of English stopwords
    """
    return {
        'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
        'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
        'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'would', 'could', 'should', 'ought', 'i\'m', 'you\'re', 'he\'s',
        'she\'s', 'it\'s', 'we\'re', 'they\'re', 'i\'ve', 'you\'ve', 'we\'ve', 'they\'ve',
        'i\'d', 'you\'d', 'he\'d', 'she\'d', 'we\'d', 'they\'d', 'i\'ll', 'you\'ll', 'he\'ll',
        'she\'ll', 'we\'ll', 'they\'ll', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'hasn\'t',
        'haven\'t', 'hadn\'t', 'doesn\'t', 'don\'t', 'didn\'t', 'won\'t', 'wouldn\'t',
        'shan\'t', 'shouldn\'t', 'can\'t', 'cannot', 'couldn\'t', 'mustn\'t', 'let\'s',
        'that\'s', 'who\'s', 'what\'s', 'here\'s', 'there\'s', 'when\'s', 'where\'s', 'why\'s',
        'how\'s'
    } # todo

def get_italian_stopwords() -> Set[str]:
    """
    Returns a set of common Italian stopwords.
    
    Returns:
        Set of Italian stopwords
    """
    return {
        'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 'dell', 'dei',
        'e', 'ed', 'o', 'ma', 'se', 'perché', 'come', 'quando', 'mentre', 'delle',
        'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 'al', 'dal', 'del',
        'nel', 'sul', 'alla', 'dalla', 'della', 'nella', 'sulla', 'quello', 'questa',
        'che', 'chi', 'cui', 'non', 'più', 'quale', 'quanto', 'quanti', 'quante',
        'questo', 'questi', 'questa', 'queste', 'sia', 'sono', 'è', 'siamo', 'siete',
        'ho', 'ha', 'hai', 'hanno', 'abbiamo', 'avete', 'avere', 'essere',
        'mi', 'ti', 'ci', 'vi', 'si', 'loro', 'mio', 'tuo', 'suo', 'nostro', 'vostro'
    } # todo

def all_stopwords() -> Set[str]:
    """
    Returns all stopwords
    """
    return get_italian_stopwords() | get_english_stopwords()

def get_stopwords(lang_code: str) -> set:
    """
    Returns a set of stopwords based on the specified language.
    
    Args:
        lang_code: Language code ('it' for Italian, 'en' for English, 'all' for all stopwords)
    
    Returns:
        Set of stopwords for the specified language
    
    Raises:
        ValueError: If an unsupported language code is provided
    """
    # Normalize input by converting to lowercase
    lang_code = lang_code.lower().strip()
    
    # Use a dictionary to map language codes to functions
    language_map = {
        'it': get_italian_stopwords,
        'en': get_english_stopwords,
        'all': all_stopwords
    }
    
    # Check if the language is supported
    if lang_code in language_map:
        return language_map[lang_code]()
    else:
        # Raise an exception with a descriptive message
        supported_languages = ", ".join(language_map.keys())
        raise ValueError(f"Language '{lang_code}' not supported. Available languages: {supported_languages}")
