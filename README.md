# Part 3: NLP and Sequence Modeling Mini Project
## Problem Statement
You are given a text-based dataset. Your task is to build a basic NLP pipeline and compare traditional text vectorization with sequence-based deep learning ideas.

The goal of this part is to understand how text is converted into numerical form and why sequence modeling is important for NLP tasks.
## Dataset
Use the relevant Part 3 dataset from the shared folder:

https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

**Dataset:** Customer Support Text Classification  
**Goal:** Classify customer messages as `positive`, `neutral`, or `negative` using an end-to-end NLP pipeline.

## Approach

Text data cannot be fed directly into machine learning models.It must first be cleaned, converted into numbers, and structured appropriately. This project builds that pipeline from scratch, progressing from simple bag-of-words models to a deep learning sequence architecture, and concludes with a conceptual reflection on modern NLP.

---
## Steps

**1. Dataset Understanding**
Loaded and explored the dataset (1,500 rows, 6 columns). Identified the target column (`sentiment_label`) and key input (`customer_message`). Found that the dataset contains 866 duplicate messages across 634 unique entries, and that classes are balanced at roughly 33% each with an average message length of ~12 words.

**2. Text Preprocessing**
Applied a cleaning pipeline to every message: lowercasing, removal of special characters and numbers using regex, tokenization by whitespace splitting, and stopword removal using a manually defined list. Domain-specific noise words (`ticket`, `number`) were also filtered out.

**3. Text Vectorization**
Two vectorization methods were implemented and compared:
- **Bag of Words** (CountVectorizer, 500 features) - counts word occurrences per document
- **TF-IDF** (TfidfVectorizer, 500 features with bigrams) - weights words by their importance relative to the corpus

Both produce sparse matrices of shape `(samples × 500)` that models can learn from.

**4. Baseline Models**
Two classifiers were trained and evaluated:
- **Naive Bayes** with Bag of Words vectors
- **Logistic Regression** with TF-IDF vectors

Both achieved 100% accuracy on the test set. Evaluation included classification report (precision, recall, F1) and confusion matrix visualisation.

**5. Sequence Model Architecture (LSTM)**
A sequence-based model was designed and coded using TensorFlow/Keras with the following layers: `Embedding → LSTM (64 units) → Dropout (0.3) → Dense (softmax)`. Text was tokenised into integer sequences of fixed length 30 using a manually built vocabulary, then converted to NumPy arrays before training. Labels were integer-encoded using `LabelEncoder`. Loss function: `sparse_categorical_crossentropy`. Optimiser: `Adam`.

**6. Attention and Transformer Reflection**
Written explanation covering the vanishing gradient problem in RNNs, how LSTM gates (forget, input, output) address memory, how attention eliminates the Seq2Seq bottleneck, and why the Transformer architecture (self-attention + parallelism) powers modern large language models.

---

## Results

| Model | Vectorization | Accuracy | F1 (macro) |
|-------|--------------|----------|------------|
| Naive Bayes | Bag of Words | 1.00 | 1.00 |
| Logistic Regression | TF-IDF | 1.00 | 1.00 |

Both models predicted all three sentiment classes perfectly on the test set.

---

## Key Observations

**100% accuracy is misleading here.** The dataset contains 866 duplicate messages. Because the same messages appear in both the training and test sets after a random split, the models effectively memorise rather than generalise. In a real project, `drop_duplicates()` must be applied before `train_test_split()`.

**The vocabulary is sentiment-transparent.** Negative messages consistently use words like `frustrating`, `terrible`, and `pending`, while positive messages use `appreciate`, `excellent`, and `resolved`. This makes the classification task trivially easy for even a simple model.Both BoW and TF-IDF capture these signals directly.

**Bag of Words vs TF-IDF.** For this dataset both perform identically, but TF-IDF is generally preferred because it down-weights common words automatically and highlights distinctive terms more effectively.

**Why sequence models still matter.** Despite the baselines scoring 100%, BoW and TF-IDF ignore word order entirely. A phrase like `"not satisfied"` looks the same as `"satisfied, not bad"` to these models. An LSTM or Transformer reads left-to-right and preserves context essential for real-world, noisier text.

---

## Project Structure

```
part-3-nlp-sequence-modeling/
│
├── README.md
├── notebook.ipynb        # Main notebook with all tasks and explanations
├── requirements.txt           # All dependencies with install instructions
└── results/
    ├── model_evaluation.png   # Class distribution + confusion matrix charts
    ├── model_evaluation.csv   # All test predictions from both models
    └── sample_predictions.txt # 15 sample predictions per model
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook notebookpart3.ipynb
```