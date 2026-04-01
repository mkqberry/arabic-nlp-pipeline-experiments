# JAIS

Arabic NLP experiments and utilities. Includes preprocessing, tokenization, and evaluation for Arabic language tasks.

## Features

- Arabic text preprocessing and normalization
- Diacritization handling
- Tokenization for modern and classical Arabic
- Corpus utilities
- Language detection
- Evaluation tools for Arabic-specific metrics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Basic preprocessing:

```python
from src.arabic_text import ArabicProcessor

processor = ArabicProcessor()
text = "النص العربي"
processed = processor.normalize(text)
```

From command line:

```bash
python scripts/main.py \
  --input arabic_corpus.txt \
  --task normalize \
  --output processed.txt
```

## Available Tasks

- `normalize` - Text normalization
- `tokenize` - Word and morpheme tokenization
- `diacritic` - Diacritization detection/removal
- `evaluate` - Corpus statistics

## Configuration

Edit `configs/arabic_config.yaml` to set preprocessing rules and dialect.

## Structure

- `src/` - Arabic NLP modules
- `scripts/` - CLI utilities
- `tests/` - Unit tests
- `examples/` - Sample datasets
