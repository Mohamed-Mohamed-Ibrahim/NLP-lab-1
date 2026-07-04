<div align="center">
  <h1>🧠 NLP Foundations: N-Grams & Text Classification</h1>
  
  <p>
    A from-scratch implementation of foundational Natural Language Processing algorithms, exploring statistical language modeling and emotion classification.
  </p>

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge&logo=python&logoColor=white" alt="NLTK" />
</p>
</div>

---

## 📖 Overview

This repository contains the implementation for **Lab #1** of the Natural Language Processing (NLP) course. The project is divided into two main parts:
1.  **Statistical Language Modeling:** Building an N-Gram language model from scratch to understand probabilistic text generation and evaluation.
2.  **Text Classification:** Implementing classical machine learning algorithms (Naïve Bayes and Logistic Regression) entirely from scratch using NumPy, and benchmarking them against `scikit-learn`.

---

## 🏗️ Project Architecture & Implementation

### Part 1: N-Gram Language Model
An exploration of text generation and probability using the **Gutenberg Corpus** (`nltk`). The model is trained on Shakespeare's *Caesar* and *Macbeth*, and evaluated on *Hamlet*.

* **Text Normalization:** Lowercasing, punctuation removal, and tokenization.
* **Model Construction:** Implements Unigram up to $N$-gram models ($N \le 10$).
* **Smoothing:** Utilizes Laplace (Add-1) smoothing to handle unseen N-grams.
* **Evaluation:** Calculates Perplexity to measure how well a probability model predicts a sample.

### Part 2: Emotion Classification
Predicting 6 emotion classes (anger, sadness, surprise, love, joy, fear) using the `avsolatorio/mteb-emotion-avs_triplets` dataset.

#### 2.1 Naïve Bayes Classification (From Scratch)
* Computes prior probabilities $P(C)$ and word likelihoods $P(w_i|C)$.
* Infers labels by maximizing the posterior probability using NumPy.
* Benchmarked against `sklearn.naive_bayes.MultinomialNB` using Bag-of-Words (`CountVectorizer`).

#### 2.2 Logistic Regression (From Scratch)
* **Feature Engineering:** Converts text to a sparse binary vector representing bi-gram presence/absence.
* **Optimization:** Implements mini-batch/stochastic gradient descent to minimize cross-entropy loss.
* **Updates:** Iteratively computes scores, class probabilities, loss, and gradients.
* Benchmarked against `sklearn.linear_model.LogisticRegression` and `SGDClassifier`.

#### 2.3 Evaluation Metrics (From Scratch)
Implementation of core evaluation metrics purely in NumPy to compare with `sklearn.metrics`:
* Confusion Matrix
* Precision, Recall, and F1-Score (per class and macro-averaged)

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* NumPy
* Scikit-Learn
* NLTK
* Datasets (`datasets` library from Hugging Face)

### Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/nlp-foundations-lab.git](https://github.com/yourusername/nlp-foundations-lab.git)
   cd nlp-foundations-lab
