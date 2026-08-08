# 🧠 C7 Financial NLP Engine

> Building a Financial NLP and Transformer architecture from scratch for quantitative finance and market intelligence.

---

# 📌 Project Objective

The goal of **C7 Financial NLP Engine** is to progressively build an end-to-end Financial Natural Language Processing system capable of processing financial news and extracting market sentiment.

The project follows a deliberately educational and engineering-oriented approach.

Instead of starting directly with a pre-trained Transformer such as BERT, the first version of C7 implements the main components of a modern Transformer NLP pipeline progressively from scratch.

The objective is not only to obtain predictions, but to understand and implement the complete path:

```text
Raw Financial Text
        ↓
Corpus Preparation
        ↓
WordPiece Tokenization
        ↓
Token IDs
        ↓
Token + Positional Embeddings
        ↓
Multi-Head Self-Attention
        ↓
Transformer Encoder
        ↓
Contextual Representation
        ↓
Classification Head
        ↓
Financial Sentiment
```

This first implementation is therefore considered:

> **C7 Financial NLP — From Scratch Version**

Future versions will compare this architecture against transfer-learning approaches using pre-trained language models such as BERT.

---

# 🚀 Current Progress

```text
Financial News Corpus
        │
        ▼
Corpus Cleaning ✅
        │
        ▼
Custom WordPiece Tokenizer ✅
        │
        ▼
Token + Positional Embedding ✅
        │
        ▼
Multi-Head Self-Attention ✅
        │
        ▼
Transformer Encoder ✅
        │
        ▼
CLS Contextual Representation ✅
        │
        ▼
Financial Sentiment Classifier ✅
        │
        ▼
Supervised Training Pipeline ✅
        │
        ▼
GPU / CUDA Training ✅
        │
        ▼
Class-Imbalance Correction ✅
        │
        ▼
Model Checkpointing (.pt) ✅
        │
        ▼
Inference on Unseen Articles ✅
        │
        ▼
Model Optimization 🚧
```

---

# 🏗️ Global Architecture

The current C7 architecture does **not rely on a pre-trained Transformer**.

The tokenizer vocabulary is learned from the financial corpus, while the neural parameters of the Embedding, Transformer Encoder and classification layers are learned during training.

```text
                  C7 FINANCIAL NLP ENGINE
                           │
                           ▼
                 Financial Text Input
                           │
                           ▼
              Custom WordPiece Tokenizer
                           │
                    tokenizer.json
                           │
                           ▼
                Input IDs + Attention Mask
                           │
                           ▼
                ┌─────────────────────┐
                │    C7 Embedding     │
                │                     │
                │ Token Embedding     │
                │        +            │
                │ Positional Embedding│
                └─────────────────────┘
                           │
                           ▼
                 Dense Representation
                    [B, L, D]
                           │
                           ▼
             ┌───────────────────────────┐
             │ Transformer Encoder Block │
             │                           │
             │ Multi-Head Self-Attention │
             │ Residual + LayerNorm      │
             │ Feed Forward Network      │
             │ Residual + LayerNorm      │
             └───────────────────────────┘
                           │
                          ×4
                           │
                           ▼
             Contextualized Representation
                    [B, L, D]
                           │
                           ▼
                  CLS Representation
                       [B, D]
                           │
                           ▼
               Neural Classification Head
                           │
                           ▼
                        Logits
                           │
                           ▼
                        Softmax
                           │
                           ▼
             Negative / Neutral / Positive
```

---

# ✅ Phase 1 — Financial Corpus Preparation

The first milestone consisted of building a high-quality financial corpus before training any NLP component.

The original Bloomberg dataset contains hundreds of thousands of financial articles covering areas such as:

- Equities
- Commodities
- Forex
- Macroeconomics
- Central Banks
- Cryptocurrencies
- Companies
- Financial Markets

## First Cleaning

The first cleaning stage removes low-quality textual information that could negatively affect downstream NLP components.

Implemented operations:

- Remove articles shorter than **20 words**
- Remove Bloomberg reporter signatures
- Remove editor contact information
- Preserve meaningful financial narrative content

## Second Cleaning

A second structural filtering layer was implemented instead of relying exclusively on article length.

Each article is analyzed through custom structural metrics:

- Narrative Density
- Numeric Group Density
- Table-like Segment Ratio
- Long Separator Detection
- Sentence Statistics
- Segment Statistics

A rule-based filtering engine then removes documents dominated by numerical tables or non-narrative structures.

---

# 📊 Final Financial Corpus

| Metric | Value |
|---|---:|
| Original Articles | **446,762** |
| Final Articles | **441,626** |
| Removed Articles | **5,136** |
| Corpus Quality | ✅ Ready for NLP |

The resulting corpus contains **441,626 cleaned financial articles**.

During the first version of C7, this corpus was primarily used to learn the financial vocabulary used by the custom tokenizer.

A future research phase may reuse this large unlabeled corpus for self-supervised Transformer pre-training.

---

# ✅ Phase 2 — Custom WordPiece Tokenizer

Instead of using the tokenizer associated with an existing pre-trained language model, C7 trains its own **WordPiece tokenizer** directly on the cleaned financial corpus.

## Tokenizer Pipeline

```text
441,626 Financial Articles
          │
          ▼
    Corpus Iterator
          │
          ▼
 Unicode Normalization
          │
          ▼
   BERT Pre-Tokenizer
          │
          ▼
   WordPiece Training
          │
          ▼
 Financial Vocabulary
          │
          ▼
     tokenizer.json
```

Implemented components include:

- Corpus Iterator
- Unicode Normalization
- BERT-style pre-tokenization
- Custom Special Tokens
- WordPiece Vocabulary Training
- Vocabulary Export
- Token ID validation

The resulting tokenizer learns a vocabulary directly from financial-market language.

The generated artifact is stored as:

```text
artifacts/
└── tokenizer/
    └── c7_wordpiece_v1/
        └── tokenizer.json
```

---

# ✅ Phase 3 — Embedding Layer

After tokenization, integer token IDs must be transformed into continuous vector representations.

The C7 Embedding module was implemented using PyTorch.

Two learnable representations are combined:

```text
Token Embedding
      +
Positional Embedding
      ↓
Final Dense Representation
```

For token `i`:

```text
Zᵢ = E(tokenᵢ) + P(i)
```

where:

- `E(tokenᵢ)` represents the learned token embedding
- `P(i)` represents the learned positional embedding

Dropout is then applied before the representation enters the Transformer.

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
        ├──────────────┐
        │              │
        ▼              ▼
Token Vector     Positional Vector
        │              │
        └────── + ─────┘
               │
               ▼
            Dropout
               │
               ▼
Dense Embedding Tensor
[batch_size, sequence_length, embedding_dim]
```

Current configuration:

```text
Embedding dimension : 256
Maximum sequence    : 512 tokens
```

---

# ✅ Phase 4 — Transformer Encoder From Scratch

The next milestone was the implementation of the central component of the architecture: the **Transformer Encoder**.

The objective was to understand and explicitly construct the internal Transformer pipeline rather than immediately relying on a pre-trained implementation.

Each Encoder Block contains:

- Multi-Head Self-Attention
- Feed Forward Network
- Residual Connections
- Layer Normalization
- Dropout
- Padding-mask management

## Encoder Block

```text
Input
[B, L, 256]
      │
      ▼
Multi-Head Self-Attention
      │
      ▼
Dropout
      │
      ▼
Residual Addition
      │
      ▼
Layer Normalization
      │
      ▼
Feed Forward Network
      │
      ▼
Dropout
      │
      ▼
Residual Addition
      │
      ▼
Layer Normalization
      │
      ▼
Output
[B, L, 256]
```

The tensor dimensions remain unchanged, but their meaning changes significantly.

Before the Transformer:

```text
Token representation
+
Position information
```

After the Transformer:

```text
Contextualized token representation
```

Each token can now incorporate information from other relevant tokens through self-attention.

---

# ✅ Phase 5 — Complete Transformer Encoder

A complete encoder was created by stacking multiple Encoder Blocks.

Current architecture:

```text
Embedding
    │
    ▼
Encoder Block 1
    │
    ▼
Encoder Block 2
    │
    ▼
Encoder Block 3
    │
    ▼
Encoder Block 4
    │
    ▼
Final Layer Normalization
    │
    ▼
Contextualized Tensor
```

Current Transformer configuration:

| Parameter | Value |
|---|---:|
| Embedding Dimension | 256 |
| Attention Heads | 8 |
| Encoder Blocks | 4 |
| Feed-Forward Dimension | 1024 |
| Maximum Sequence Length | 512 |
| Initial Dropout | 0.1 |

A complete article therefore follows:

```text
[1, 512]
Input IDs

      ↓

[1, 512, 256]
Dense Embeddings

      ↓

4 Transformer Encoder Blocks

      ↓

[1, 512, 256]
Contextualized Representation
```

---

# ✅ Phase 6 — CLS Representation

For document-level classification, C7 uses the contextualized representation associated with the `[CLS]` token.

```text
Transformer Output
[B, L, D]
      │
      ▼
Select position 0
      │
      ▼
CLS
[B, D]
```

With the current configuration:

```text
[batch_size, 512, 256]
            ↓
[batch_size, 256]
```

The CLS representation is used as the global document representation passed to the classification head.

---

# ✅ Phase 7 — Financial Sentiment Classifier

The first downstream task implemented for C7 is **financial sentiment classification**.

The model predicts three classes:

```text
0 → Negative
1 → Neutral
2 → Positive
```

The complete architecture becomes:

```text
Financial Text
      │
      ▼
WordPiece Tokenizer
      │
      ▼
C7 Embedding
      │
      ▼
C7 Transformer Encoder
      │
      ▼
Contextualized CLS
      │
      ▼
Fully Connected Neural Network
      │
      ▼
3 Logits
      │
      ▼
Softmax
      │
      ▼
Negative / Neutral / Positive
```

The classification head includes:

- Linear Layers
- GELU Activation
- Dropout
- Final 3-dimensional output layer

During training, the raw logits are passed directly to `CrossEntropyLoss`.

Softmax is applied during inference to convert logits into interpretable class probabilities.

---

# ✅ Phase 8 — Supervised Sentiment Dataset

A separate labeled financial sentiment dataset was prepared for supervised learning.

The dataset contains financial sentences associated with one of the three sentiment labels:

```text
Sentence → Negative
Sentence → Neutral
Sentence → Positive
```

The cleaned dataset is split using stratification:

```text
80% → Training
10% → Validation
10% → Testing
```

PyTorch `Dataset` and `DataLoader` components were implemented to handle:

- Tokenization
- Padding
- Truncation
- Attention masks
- Labels
- Batching
- Dataset shuffling

Current training batch size:

```text
batch_size = 16
```

---

# ⚡ Phase 9 — GPU Training

Initial experiments were performed on CPU.

The local PyTorch installation was then migrated to a CUDA-enabled build.

Current training hardware:

```text
GPU : NVIDIA GeForce RTX 3080
VRAM: 10 GB
CUDA-enabled PyTorch
```

The training pipeline automatically selects:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Model parameters and training batches are transferred to the selected device.

GPU acceleration dramatically reduced training time and made multi-epoch Transformer experiments practical locally.

---

# ✅ Phase 10 — Training Pipeline

The complete supervised learning pipeline is now operational.

```text
Training Batch
      │
      ▼
Tokenization
      │
      ▼
Embedding
      │
      ▼
Transformer
      │
      ▼
CLS
      │
      ▼
Classifier
      │
      ▼
Logits
      │
      ▼
CrossEntropyLoss
      │
      ▼
Backpropagation
      │
      ▼
Gradient Computation
      │
      ▼
AdamW
      │
      ▼
Parameter Update
```

Each epoch performs forward propagation, loss computation, gradient backpropagation and parameter optimization.

Validation is performed separately after each training epoch.

---

# ⚖️ Phase 11 — Class Imbalance Correction

Initial experiments revealed an important issue in the labeled dataset.

The three sentiment classes are not equally represented.

The model consequently developed a strong preference for the majority `Neutral` class.

An initial test produced:

```text
Negative F1 ≈ 0.0435
Neutral  F1 ≈ 0.7825
Positive F1 ≈ 0.5924

Macro F1 ≈ 0.4728
```

The model was therefore achieving reasonable global accuracy while almost completely failing to identify negative financial sentiment.

To address this issue, class weights were calculated **only from the training set**.

```text
Minority class
      ↓
Higher loss weight

Majority class
      ↓
Lower loss weight
```

The resulting weights were integrated into:

```python
CrossEntropyLoss(weight=class_weights)
```

This prevents the training objective from being dominated by the majority class.

---

# 📈 Phase 12 — Experimental Results

## First Balanced 5-Epoch Experiment

After introducing weighted CrossEntropyLoss:

```text
Accuracy : 0.6387
Macro F1 : 0.5626
```

Class-level performance:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Negative | 0.3158 | 0.4186 | 0.3600 |
| Neutral | 0.7447 | 0.7827 | 0.7632 |
| Positive | 0.6525 | 0.4973 | 0.5644 |

The most significant improvement occurred in the `Negative` class.

```text
Negative F1

Before weighting : ≈ 0.0435
After weighting  : ≈ 0.3600
```

This confirmed that class imbalance was one of the major weaknesses of the initial training configuration.

---

# 📊 Phase 13 — Extended 10-Epoch Experiment

Training was then extended to **10 epochs** without changing the fundamental architecture.

Training loss progressively decreased:

```text
Epoch 1  → 1.0568
Epoch 2  → 0.9620
Epoch 3  → 0.9046
Epoch 4  → 0.8232
Epoch 5  → 0.7212
Epoch 6  → 0.6356
Epoch 7  → 0.5417
Epoch 8  → 0.4525
Epoch 9  → 0.4000
Epoch 10 → 0.3524
```

Training accuracy increased from:

```text
47.79%
      ↓
83.70%
```

However, validation loss eventually increased while training loss continued decreasing.

This revealed an important phenomenon:

> **The model started to overfit the training dataset.**

Best observed validation Macro F1:

```text
Epoch 9 → 0.5793
```

---

# 🧪 Final Test Results

The best checkpoint was evaluated on the untouched test dataset.

```text
Accuracy  : 0.6558
Precision : 0.5840
Recall    : 0.5909
Macro F1  : 0.5862
```

Detailed classification report:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.3614 | 0.3488 | 0.3550 | 86 |
| Neutral | 0.7526 | 0.6997 | 0.7252 | 313 |
| Positive | 0.6381 | 0.7243 | 0.6785 | 185 |

Final confusion matrix:

```text
[[ 30  31  25]
 [ 43 219  51]
 [ 10  41 134]]
```

The experiment demonstrates that the model learns meaningful financial sentiment patterns, while also revealing that the `Negative` class remains significantly more difficult.

---

# 💾 Phase 14 — Model Checkpointing

The training pipeline now automatically stores the best-performing model according to validation Macro F1.

The checkpoint contains:

```text
Model parameters
Optimizer state
Best epoch
Validation loss
Validation accuracy
Validation F1
Model architecture configuration
```

The resulting `.pt` checkpoint allows C7 to be reloaded without repeating the complete training process.

Conceptually:

```text
Training
   │
   ▼
Best Validation F1
   │
   ▼
torch.save(...)
   │
   ▼
C7 checkpoint.pt
   │
   ▼
Future Inference
```

---

# ✅ Phase 15 — Inference on Unseen Financial Articles

A separate inference pipeline was implemented to test the trained model on previously unseen financial text.

No gradient computation or training occurs during inference.

```text
New Financial Article
        │
        ▼
Tokenizer
        │
        ▼
C7 Embedding
        │
        ▼
C7 Transformer
        │
        ▼
CLS
        │
        ▼
Sentiment Classifier
        │
        ▼
Logits
        │
        ▼
Softmax
        │
        ▼
Negative Probability
Neutral Probability
Positive Probability
```

The model is loaded from the previously saved `.pt` checkpoint and switched to evaluation mode.

---

# 🔬 Phase 16 — Contextual Stress Test

A deliberately difficult test article was created to evaluate whether the model understands global context or relies excessively on individual sentiment-associated words.

The article intentionally contained positive lexical signals such as:

```text
strong revenue growth
higher demand
expanding sales
improvement
investment opportunities
strong customer activity
```

while its global financial meaning was negative:

```text
operating costs increased
profit margins deteriorated
free cash flow turned negative
significant net loss
profitability under pressure
earnings outlook reduced
```

The model predicted approximately:

```text
Negative ≈ 2.1%
Neutral  ≈ 28%
Positive ≈ 69%
```

Despite the globally negative meaning, the model strongly favored the positive class.

This experiment revealed an important limitation of the current version:

> **C7 V1 has learned useful lexical financial sentiment patterns, but its deeper contextual understanding remains limited.**

The model appears to rely too strongly on sentiment-associated lexical signals when positive and negative information coexist in a more complex semantic structure.

This limitation is especially important because the Transformer and embeddings were initialized from scratch and trained directly on a relatively small supervised sentiment dataset.

---

# 🧠 What C7 V1 Currently Represents

C7 V1 should therefore be considered:

```text
A functional Transformer NLP prototype
built progressively from scratch
        +
capable of learning financial sentiment patterns
        +
capable of inference on unseen text
        +
experimentally evaluated
        +
known limitations identified
```

It should **not yet be considered a production-grade financial sentiment model**.

The current objective remains experimentation, architecture understanding and progressive model improvement.

---

# 🔬 Current Research Problem

The most important current challenge is no longer simply making the architecture run.

The complete architecture already works.

The current research question is:

> **How can the model learn stronger contextual financial representations instead of relying primarily on lexical sentiment signals?**

Several future directions will be investigated.

---

# 🚧 Next Phase — Model Optimization

The next optimization experiments will investigate:

### Early Stopping

Stop training when validation performance no longer improves.

### Learning Rate Scheduling

Reduce the learning rate when validation performance reaches a plateau.

### Regularization

Experiment with controlled dropout adjustments to reduce overfitting.

### Sequence Length Optimization

The sentiment dataset contains relatively short financial sentences, making the current maximum sequence length of 512 potentially unnecessary for fine-tuning.

### Class-Imbalance Strategy

Continue investigating weighted losses and potentially alternative sampling strategies.

---

# 🔮 Future Research — Self-Supervised Pre-Training

The cleaned Bloomberg corpus contains:

```text
441,626 financial articles
```

and does not contain sentiment labels.

However, labels are not required for self-supervised language-model pre-training.

A future version may reuse the corpus through a **Masked Language Modeling (MLM)** objective.

Example:

```text
Original:

Margins declined sharply despite higher revenue.

Masked:

Margins [MASK] sharply despite higher revenue.

Target:

declined
```

The objective would be to allow the C7 Transformer to learn richer financial-language relationships before being fine-tuned for sentiment classification.

The future pipeline would therefore become:

```text
441,626 Bloomberg Articles
        │
        ▼
Self-Supervised Pre-Training
        │
        ▼
C7 Financial Encoder
        │
        ▼
Labeled Sentiment Dataset
        │
        ▼
Supervised Fine-Tuning
        │
        ▼
Financial Sentiment
```

---

# 🔮 Future Research — Transfer Learning

A second major research direction will compare C7 against a pre-trained Transformer architecture.

Instead of initializing all neural parameters from scratch:

```text
C7 V1

Custom Tokenizer
      ↓
Randomly Initialized Embeddings
      ↓
Randomly Initialized Transformer
      ↓
Sentiment Training
```

a transfer-learning experiment will use a pre-trained model such as BERT:

```text
Pre-Trained BERT
      ↓
Pre-Trained Language Representations
      ↓
Financial Sentiment Dataset
      ↓
Fine-Tuning
      ↓
Financial Sentiment
```

The objective will not be to replace the C7 implementation.

Instead, both approaches will be compared experimentally:

```text
C7 FROM SCRATCH
        VS
BERT TRANSFER LEARNING
```

using comparable:

- train / validation / test splits
- classification metrics
- confusion matrices
- contextual stress tests
- inference examples

This future experiment will provide a practical demonstration of the impact of large-scale language-model pre-training.

---

# 📈 Current Project Status

| Module | Status |
|---|:---:|
| Bloomberg Dataset Collection | ✅ |
| First Corpus Cleaning | ✅ |
| Structural Analysis | ✅ |
| Second Corpus Cleaning | ✅ |
| Custom WordPiece Tokenizer | ✅ |
| Token Embedding | ✅ |
| Positional Embedding | ✅ |
| C7 Embedding Module | ✅ |
| Self-Attention | ✅ |
| Multi-Head Attention | ✅ |
| Feed Forward Network | ✅ |
| Residual Connections | ✅ |
| Layer Normalization | ✅ |
| Transformer Encoder Block | ✅ |
| 4-Layer Transformer Encoder | ✅ |
| CLS Representation | ✅ |
| Sentiment Classification Head | ✅ |
| Sentiment Dataset Preparation | ✅ |
| PyTorch Dataset / DataLoader | ✅ |
| Training Pipeline | ✅ |
| CUDA / RTX 3080 Training | ✅ |
| Validation Pipeline | ✅ |
| Class Weighting | ✅ |
| Test Evaluation | ✅ |
| Confusion Matrix | ✅ |
| Model Checkpoint `.pt` | ✅ |
| Unseen Article Inference | ✅ |
| Contextual Stress Test | ✅ |
| Early Stopping | 🚧 |
| LR Scheduler | 🚧 |
| Overfitting Optimization | 🚧 |
| MLM Pre-Training | 🔬 Future Research |
| BERT Transfer Learning | 🔬 Future Research |
| C7 vs BERT Benchmark | 🔬 Future Research |
| API Inference | 🔮 Planned |
| Market Intelligence Pipeline | 🔮 Long-Term |

---

# 🛠️ Technologies

Current stack:

- Python
- Pandas
- NumPy
- Scikit-learn
- Hugging Face Tokenizers
- PyTorch
- CUDA
- NVIDIA RTX 3080
- Git / GitHub

Planned technologies:

- Hugging Face Transformers
- BERT
- FastAPI
- PostgreSQL / Vector Storage

---

# 📅 Roadmap

- [x] Collect financial corpus
- [x] Clean Bloomberg corpus
- [x] Develop structural filtering
- [x] Train custom WordPiece tokenizer
- [x] Implement Token Embedding
- [x] Implement Positional Embedding
- [x] Implement Multi-Head Self-Attention
- [x] Implement Transformer Encoder Block
- [x] Stack Transformer Encoder Blocks
- [x] Extract contextual CLS representation
- [x] Implement Financial Sentiment Classifier
- [x] Prepare labeled financial sentiment dataset
- [x] Implement PyTorch DataLoaders
- [x] Implement complete training loop
- [x] Enable CUDA GPU training
- [x] Add validation metrics
- [x] Diagnose class imbalance
- [x] Implement weighted CrossEntropyLoss
- [x] Train 5-epoch balanced model
- [x] Train 10-epoch balanced model
- [x] Evaluate Precision / Recall / Macro F1
- [x] Generate confusion matrix
- [x] Save best model checkpoint
- [x] Reload model for inference
- [x] Classify unseen financial articles
- [x] Perform contextual stress test
- [x] Identify lexical/contextual limitation
- [ ] Add Early Stopping
- [ ] Add Learning Rate Scheduler
- [ ] Optimize regularization
- [ ] Improve Negative-class performance
- [ ] Investigate self-supervised MLM pre-training
- [ ] Build BERT transfer-learning baseline
- [ ] Benchmark C7 From Scratch vs BERT
- [ ] Build inference API
- [ ] Extend toward market-level sentiment aggregation

---

# 🎯 Long-Term Objective

The final objective extends beyond classifying individual articles.

C7 is intended to progressively evolve toward a financial intelligence pipeline capable of processing large quantities of financial information and estimating broader market sentiment.

```text
Financial News Streams
          │
          ▼
       C7 NLP
          │
          ▼
Article-Level Sentiment
          │
          ▼
Sentiment Aggregation
          │
          ▼
Sector / Asset / Market Analysis
          │
          ▼
Dominant Market Sentiment
          │
          ▼
Quantitative Finance Research
```

---

# 🧠 Project Philosophy

C7 intentionally began without relying directly on a pre-trained NLP pipeline.

The objective of the first version was to understand and implement the fundamental components behind modern Transformer architectures:

```text
Tokenization
     ↓
Embeddings
     ↓
Self-Attention
     ↓
Multi-Head Attention
     ↓
Feed Forward Networks
     ↓
Residual Connections
     ↓
Layer Normalization
     ↓
Transformer Encoder
     ↓
Contextual Representations
     ↓
Classification
     ↓
Loss
     ↓
Backpropagation
     ↓
Optimization
```

The limitations discovered during experimentation are considered part of the engineering process rather than hidden.

In particular, the contextual stress test demonstrated why building a Transformer architecture and obtaining reasonable classification metrics does not automatically imply deep language understanding.

This provides the foundation for the next stage of the project:

> **moving from understanding Transformers from scratch toward understanding pre-training, fine-tuning and transfer learning experimentally.**

The long-term objective is to combine this architectural understanding with stronger pre-trained models, quantitative finance workflows and market-intelligence systems.