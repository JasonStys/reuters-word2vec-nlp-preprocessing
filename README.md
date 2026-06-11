# Reuters Word2Vec NLP Preprocessing

This repository contains a CS476 natural language processing notebook that preprocesses Reuters news text, builds bigram phrases, trains a Word2Vec embedding model, explores word similarity, and visualizes learned word vectors with PCA.

## Project Overview

The project demonstrates a complete introductory NLP workflow using the Reuters corpus from NLTK. It begins by loading raw news articles, cleaning and tokenizing text, removing stopwords, detecting common bigram phrases, and training a Word2Vec model with Gensim.

After training, the notebook explores the learned vocabulary by printing frequent words and querying similar terms. It then reduces selected word vectors to two dimensions with PCA and plots a labeled visualization of the embedding space.

## Dataset

The project uses the Reuters corpus provided by NLTK.

```text
Total Reuters documents available: 10,788
Documents used in this notebook: 200
Text source: NLTK Reuters corpus
```

The notebook uses a smaller subset of documents for faster classroom execution. The `MAX_DOCS` value can be changed to `None` to use the full Reuters corpus.

## Preprocessing Pipeline

The notebook preprocesses text with the following steps:

- Convert text to lowercase
- Keep only letters and spaces
- Tokenize with NLTK
- Remove English stopwords
- Remove one character tokens
- Detect and apply bigram phrases
- Remove empty documents
- Prepare tokenized documents for Word2Vec training

Example preprocessing result:

```text
Original:
The quick brown fox jumps over the lazy dog in New York City.

Preprocessed:
['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog', 'new', 'york', 'city']
```

Example bigram result:

```text
['new_york', 'city', 'major', 'financial', 'center']
['new_york', 'stock', 'exchange', 'new_york', 'city']
['visited', 'new_york', 'last', 'summer']
```

## Word2Vec Model

The notebook trains a Gensim Word2Vec model with the following settings:

```text
Embedding size: 100
Context window: 5
Minimum word count: 5
Worker threads: 4
Training epochs: 10
Architecture: Skip gram
Negative samples: 10
Random seed: 42
```

The trained model produced:

```text
Vocabulary size: 829
Saved model file: ca9_reuters_word2vec.model
```

## Similarity Exploration

The notebook checks similarity results for several Reuters related terms.

Example probe words:

```text
bank
oil
market
company
trade
```

Example similar words found by the model include:

```text
bank -> money_market, bills, treasury
oil -> kuwait, malaysia, prices
market -> interest_rates, inflation, funds
company -> per_share, reported, dividend
trade -> washington, sanctions, ministry
```

Because the notebook trains on only 200 documents, some similarity results are noisy. Using the full Reuters corpus would likely improve embedding quality.

## Visualization

The notebook reduces 100 dimensional Word2Vec vectors into two dimensions with PCA and visualizes 200 frequent words in a labeled scatter plot.

The plot helps show how words from related financial and news topics cluster in the learned embedding space.

## Technologies Used

- Python
- NLTK
- Gensim
- Word2Vec
- NumPy
- Matplotlib
- scikit-learn
- PCA
- Jupyter Notebook
- Google Colab

## Repository Structure

```text
.
├── Jason_Stys_CS476_01_CA9_word2vec_and_preprocess.ipynb
├── Jason Stys CS476-01 CA9 word2vec and preprocess.pdf
├── README.md
└── requirements.txt
```

Optional generated file after running the notebook:

```text
ca9_reuters_word2vec.model
```

## How to Run

Clone the repository.

```bash
git clone https://github.com/your-username/reuters-word2vec-nlp-preprocessing.git
cd reuters-word2vec-nlp-preprocessing
```

Install the required packages.

```bash
pip install nltk gensim numpy matplotlib scikit-learn notebook
```

Launch Jupyter Notebook.

```bash
jupyter notebook
```

Open the notebook file and run all cells from top to bottom.

The first run downloads the required NLTK resources:

```python
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("reuters")
nltk.download("punkt_tab")
```

## Skills Demonstrated

- Natural language processing preprocessing
- Tokenization and stopword removal
- Regular expression based text cleaning
- Bigram phrase detection
- Word2Vec embedding training
- Skip gram model configuration
- Word similarity exploration
- PCA dimensionality reduction
- Embedding visualization
- Jupyter Notebook workflow

## Possible Future Improvements

- Train on the full Reuters corpus instead of 200 documents
- Compare skip gram against CBOW
- Add t-SNE visualization in addition to PCA
- Add a cleaner embedding plot with fewer labels
- Save and reload the trained Word2Vec model
- Add analogy queries such as word vector arithmetic
- Compare custom embeddings against pretrained embeddings
