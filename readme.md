# 🧠 C7 Financial NLP Engine

> Building a complete Financial NLP pipeline from scratch for quantitative finance and market intelligence.

---

# 📌 Project Objective

The goal of this project is to progressively build an end-to-end Financial NLP system capable of understanding thousands of financial news articles before ultimately estimating the dominant sentiment of global financial markets.

Unlike traditional NLP projects that directly rely on pre-trained models, every major component is implemented and studied progressively to gain a deep understanding of modern NLP and Transformer architectures.

---

# 🚀 Current Progress

```text
Financial News
        │
        ▼
Corpus Cleaning ✅
        │
        ▼
Custom WordPiece Tokenizer ✅
        │
        ▼
Embedding Layer
        │
        ▼
Transformer Encoder
        │
        ▼
Article Classification
        │
        ▼
Financial Sentiment Classification
```

---

# 📂 Project Pipeline

```text
Bloomberg Dataset
        │
        ▼
First Cleaning
        │
        ▼
Second Cleaning
        │
        ▼
Clean Financial Corpus
        │
        ▼
WordPiece Tokenizer
        │
        ▼
Token IDs
        │
        ▼
Embedding Layer
        │
        ▼
Transformer Encoder
        │
        ▼
Model 1
(Content Classification)
        │
        ▼
Model 2
(Financial Sentiment)
```

---

# ✅ Phase 1 — Corpus Preparation

The first milestone consisted of building a high-quality financial corpus before training any NLP model.

The original Bloomberg dataset contains hundreds of thousands of financial articles covering:

- Equities
- Commodities
- Forex
- Macroeconomics
- Central Banks
- Cryptocurrencies

---

## 🔹 First Cleaning

The first cleaning stage removes low-quality textual information that could negatively impact future NLP models.

Implemented operations:

- ✅ Remove articles shorter than **20 words**
- ✅ Remove Bloomberg reporter signatures
- ✅ Remove editor contact information
- ✅ Preserve only meaningful financial content

---

## 🔹 Second Cleaning

Instead of relying only on article length, a structural analysis was developed.

Each article is analyzed using several custom metrics:

- Narrative Density
- Numeric Group Density
- Table-like Segment Ratio
- Long Separator Detection
- Sentence Statistics
- Segment Statistics

A rule-based filtering engine automatically removes documents that are primarily numerical or tabular (such as USDA reports and market tables).

---

# 📊 Final Corpus

| Metric | Value |
|---------|-------|
| Original Articles | **446,762** |
| Final Articles | **441,626** |
| Removed Articles | **5,136** |
| Corpus Quality | ✅ Ready for NLP |

The resulting corpus now contains only narrative financial articles suitable for language modeling.

---

# ✅ Phase 2 — Custom WordPiece Tokenizer

Instead of using a pre-trained tokenizer, this project trains its own **WordPiece tokenizer** directly on the cleaned financial corpus.

The tokenizer pipeline includes:

- Corpus Iterator
- Unicode Normalization
- BERT Pre-Tokenizer
- Custom Special Tokens
- WordPiece Vocabulary Training
- Vocabulary Export (`tokenizer.json`)

---

## Generated Components

```text
Clean Corpus
      │
      ▼
WordPiece Trainer
      │
      ▼
Vocabulary
      │
      ▼
tokenizer.json
```

The tokenizer learns financial terminology directly from the corpus, producing a vocabulary specialized for market news.

The generated tokenizer will be reused by every future model developed in this project.

---

# 📈 Current Project Status

| Module | Status |
|----------|--------|
| Dataset Collection | ✅ |
| Corpus Cleaning | ✅ |
| Structural Analysis | ✅ |
| WordPiece Tokenizer | ✅ |
| Embedding Layer | ⏳ |
| Transformer Encoder | ⏳ |
| Article Classifier | ⏳ |
| Sentiment Classifier | ⏳ |
| Multi-Agent Pipeline | ⏳ |

---

# ✅ Phase 3 — Embedding Layer

After training the custom WordPiece tokenizer, the next step was to transform token IDs into dense vector representations that can be processed by a neural network.

Unlike one-hot encoding, embedding layers learn continuous vector representations where semantically related tokens can gradually occupy similar regions of the embedding space during training.

The embedding pipeline was implemented progressively from scratch.

Implemented components:

- Load the custom `tokenizer.json`
- Tokenize real financial articles from the cleaned corpus
- Generate `input_ids` and `attention_mask`
- Validate the generated token IDs against the custom vocabulary
- Implement a learnable **Token Embedding** using PyTorch
- Implement a learnable **Positional Embedding** inspired by the BERT architecture
- Combine both representations
- Apply dropout regularization before feeding the vectors into the Transformer

---

## Embedding Pipeline

```text
Financial Article
        │
        ▼
WordPiece Tokenizer
        │
        ▼
Input IDs + Attention Mask
        │
        ▼
Token Embedding
        │
        ▼
Positional Embedding
        │
        ▼
Z = E(token_i) + P(i)
        │
        ▼
Dropout
        │
        ▼
Dense Embedding
[batch_size, sequence_length, embedding_dimension]
```

The embedding module is now fully operational and can transform any financial article into dense vector representations ready to be processed by the Transformer Encoder.

# 🛠️ Technologies

- Python
- Pandas
- Hugging Face Datasets
- Hugging Face Tokenizers
- PyTorch *(coming next)*
- FastAPI *(coming next)*
- PostgreSQL *(future vector storage)*
- Transformers *(implemented from scratch progressively)*

---

# 📅 Roadmap

- [x] Corpus Collection
- [x] Corpus Cleaning
- [x] Structural Analysis
- [x] Custom WordPiece Tokenizer
- [x] Embedding Layer
- [ ] Self-Attention
- [ ] Transformer Encoder
- [ ] Article Classification Model
- [ ] Financial Sentiment Model
- [ ] Multi-Agent Financial Intelligence System

---

> **Project Philosophy**

This project intentionally avoids relying on pre-trained NLP pipelines during the learning phase. Every major component is progressively implemented and studied—from corpus preparation and tokenization to Transformer architectures—to build a strong understanding of modern Natural Language Processing before moving toward large language models and agentic AI systems.