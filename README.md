# sentenza

A Python library for extracting and processing sentences from text with statistical chunking capabilities.

## Overview

Splitting text into chunks is a fundamental step in building a high-quality vector database. When generating embeddings for RAG systems, the granularity and semantic coherence of the segments directly influence the accuracy and relevance of search results. Chunks that are too short excessively fragment the content, while chunks that are too long risk merging unrelated information, reducing the effectiveness of queries.

To address this challenge, I wrote a Python library (sentenza), still in an experimental phase, that aims to optimize the text chunking process. The project was born from the need to have a tool that tries to efficiently split the text and that also offers a visual analysis of the distribution of sentences to support more informed decisions on chunking parameters.

The entire process is strictly linked to the tokenizer used. Correct identification of sentence boundaries is essential to apply effective chunking strategies. In this case, the sentence adopts a simple approach on texts already extracted from documents, generating a graph useful for analyzing the length distribution and adjusting the segmentation parameters.

## The Challenge

Text chunking presents several challenges in NLP pipelines:

- Fixed-size chunking can break sentences mid-thought, disrupting semantic coherence
- Naive approaches often create either excessive overlap (wasting processing) or insufficient overlap (losing context)
- Determining optimal chunk size is typically done through trial and error
- Processing very large texts efficiently requires careful memory management

sentenza attempts to address these issues through statistical analysis and sentence-aware processing, though it remains a work in progress.

## Features

- Extract sentences with handling for common abbreviations
- Attempt to create optimally sized semantic chunks with appropriate overlap
- Calculate statistics-based chunk parameters 
- Process texts through memory-efficient streaming
- Customize stopwords for English and Italian
- Visualize sentence length distribution with customizable output filenames

## Installation

```bash
# Clone the repository
git clone https://github.com/guelfoweb/sentenza.git

# Navigate to the directory
cd sentenza

# Install the requirements
pip install -r requirements.txt

# Install the library in development mode
pip install -e .
```

## Basic Usage

```python
from sentenza import Tokenizer, get_stopwords

# Initialize the tokenizer
stopwords = get_stopwords('en')
tokenizer = Tokenizer(
    lowercase=True,
    remove_punctuation=True,
    stopwords=stopwords,
    min_length=2,
    chunk_size=10000  # Processing buffer size
)

# Process text
with open('document.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Example output:
# Loaded 152KB of text
```

## Working with Chunks

How it calculates the chunks:

`chunk_size = int(mean_length + 3 * std_dev)`

`size` is the sum of the mean length + 3 times the standard deviation.

`chunk_overlap = int(std_dev * 1.5)`

`overlap` is 1.5 times the standard deviation.

Too much overlap may cause excessive redundancy, storage and computation efficiency problems, and bias in retrieval results.

```python
# Extract sentences
sentences = list(tokenizer.sentences_stream(text))

# Calculate statistics and suggested chunk parameters
stats = tokenizer.statistics(sentences)
print(f"Suggested chunk size: {stats['chunk_size']} characters")
print(f"Suggested overlap: {stats['chunk_overlap']} characters")
print(f"Estimated chunks needed: {stats['estimated_chunks']}")

# Create semantic chunks with calculated parameters
semantic_chunks = tokenizer.semantic_chunks(sentences)
print(f"Created {len(semantic_chunks)} chunks")

# Example output:
# Suggested chunk size: 1188 characters
# Suggested overlap: 326 characters
# Estimated chunks needed: 4
# Created 5 chunks
```

## Text Processing Options

The library offers different methods for sentence extraction depending on text size:

### For smaller texts:

```python
# Extract sentences from a smaller document
sentences = tokenizer.tokenize_sentences(text)
print(f"Found {len(sentences)} sentences")

# Example output:
# Found 53 sentences
```

### For larger texts (memory-efficient):

```python
# Stream sentences from a large document
sentences = list(tokenizer.sentences_stream(text))
print(f"Found {len(sentences)} sentences")

# Example output:
# Found 147 sentences
```

## Visualization

```python
# Visualize sentence distribution with default filename
tokenizer.plotting(sentences)

# Visualize with custom filename
tokenizer.plotting(sentences, filename="my_sentence_histogram.png")

# Example output:
# (Creates histogram_with_chunks.png or my_sentence_histogram.png)
```

![Image](https://github.com/user-attachments/assets/69535092-dc03-4bfe-aac1-a53f07b94aea)

## Integration Example

The following example demonstrates how to use sentenza with popular embedding and vector database libraries for RAG applications. This example specifically uses the HuggingFace Sentence Transformers model "all-MiniLM-L6-v2" for creating embeddings and FAISS (Facebook AI Similarity Search) as the vector database.

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Get text chunks using sentenza
sentences = list(tokenizer.sentences_stream(text))
chunks = tokenizer.semantic_chunks(sentences)

# Create embeddings using HuggingFace Sentence Transformers
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build a FAISS vector database from the optimized chunks
vector_db = FAISS.from_texts(chunks, embeddings)

# Example output:
# Created embeddings for 5 chunks
# Vector database created with dimension: 384
```

This integration allows for more effective retrieval in RAG systems since the chunks maintain semantic coherence while being optimally sized for the embedding model.

## Customization

### Custom Chunk Parameters

sentenza automatically calculates optimal chunk parameters based on text statistics, eliminating the need for manual configuration. This feature allows you to divide sentences appropriately without trial and error or manual adjustments.

```python
# Automatic usage (recommended):
# sentenza calculates optimal chunk_size and chunk_overlap based on the text
auto_chunks = tokenizer.semantic_chunks(sentences)
print(f"Created {len(auto_chunks)} chunks with optimally calculated parameters")

# Only if necessary, you can override the automatic parameters:
custom_chunks = tokenizer.semantic_chunks(
    sentences,
    chunk_size=1000,
    chunk_overlap=200
)
print(f"Created {len(custom_chunks)} chunks with custom parameters")

# Example output:
# Created 5 chunks with optimally calculated parameters
# Created 7 chunks with custom parameters
```

The main advantage of sentenza is precisely this ability to automatically determine the most suitable parameters for the specific text, without requiring the user to make manual attempts.

### Language Support

```python
# Italian stopwords
italian_stopwords = get_stopwords('it')
tokenizer = Tokenizer(stopwords=italian_stopwords)
print(f"Loaded {len(italian_stopwords)} Italian stopwords")

# Combined languages
all_stopwords = get_stopwords('all')
tokenizer = Tokenizer(stopwords=all_stopwords)
print(f"Loaded {len(all_stopwords)} multilingual stopwords")

# Example output:
# Loaded 74 Italian stopwords
# Loaded 353 multilingual stopwords
```

### Token Statistics

```python
# Extract tokens and get statistics
tokens = tokenizer.tokenize(text)
token_counts = tokenizer.count_tokens(tokens, sort_by='count', reverse=True)
print(f"Most common token: '{list(token_counts.keys())[0]}' appears {list(token_counts.values())[0]} times")

# Example output:
# Most common token: 'text' appears 23 times
```

## Project Status

This is an experimental project in active development. The approaches and algorithms may change as I test and refine the library. Feedback and contributions are welcome to help improve this testing ground for text processing methods.

## License

MIT
