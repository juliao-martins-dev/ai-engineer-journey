# NLP (Natural Language Processing) — Course Notes

> Notes from the NLP Module of my Udemy Data Science / Machine Learning course (Sections 20–28).
> Status: ✅ Completed — 68/68 lessons | ~3hr 21min total

This document captures what I learned step by step, in the order the course taught it, so I can revisit the logic later and reuse the workflow in real projects.

---

## Table of Contents

1. [Introduction to NLP](#1-introduction-to-nlp)
2. [Text Preprocessing](#2-text-preprocessing)
3. [Identifying Parts of Speech and Named Entities](#3-identifying-parts-of-speech-and-named-entities)
4. [Sentiment Analysis](#4-sentiment-analysis)
5. [Vectorizing Text](#5-vectorizing-text)
6. [Topic Modelling](#6-topic-modelling)
7. [Building a Custom Text Classifier](#7-building-a-custom-text-classifier)
8. [Case Study: Categorizing Fake News](#8-case-study-categorizing-fake-news)
9. [The Future of NLP](#9-the-future-of-nlp)
10. [Key Takeaways](#10-key-takeaways)

---

## 1. Introduction to NLP

**Section 20 · 7 lessons · 7min**

The module opens with the fundamentals — what NLP actually is and where it fits into everyday tools.

- **What NLP is**: the field of teaching computers to read, interpret, and generate human language.
- **NLP in everyday life**: search engines, spam filters, voice assistants, translation apps, autocomplete — all powered by NLP under the hood.
- **Supervised vs. unsupervised NLP**: the same split that applies to ML in general applies here.
  - *Supervised* → labeled text data (e.g., spam/not spam, positive/negative review) used to train a classifier.
  - *Unsupervised* → no labels; the goal is to discover structure (e.g., clustering documents by topic).

**Why this matters**: before touching any code, this section sets the mental model — NLP problems are either "predict a known label" or "find hidden structure," and that distinction drives every technique that follows.

---

## 2. Text Preprocessing

**Section 21 · 23 lessons · 1hr 9min**

The biggest section of the module — and the most important, since raw text is messy and models can't use it as-is. This is the "cleaning" stage every NLP pipeline depends on.

### Step-by-step pipeline learned:

1. **Why data preparation matters** — dirty text (inconsistent casing, punctuation, noise) hurts model performance.
2. **Environment setup** — installing and exploring NLP packages (mainly **NLTK**).
3. **Lowercasing text** — normalizing so "Apple" and "apple" are treated the same.
4. **Removing stop words** — filtering out high-frequency, low-meaning words (*the, is, at, on...*).
5. **Regular expressions (regex)** — pattern matching to strip out URLs, punctuation, special characters, digits, etc.
6. **Tokenization** — splitting text into individual units (words or sentences) so it can be processed piece by piece.
7. **Stemming** — reducing words to a crude root form (e.g., *running* → *run*) using fast, rule-based cutting.
8. **Lemmatization** — a smarter version of stemming that returns real dictionary words using grammar/context (e.g., *better* → *good*).
9. **N-grams** — grouping sequences of N words together (bigrams, trigrams) to preserve some context/word order.
10. **Using pandas with NLTK** — applying preprocessing functions across a whole dataset efficiently with `.apply()` and lambda functions.
11. **Practical task** — put the full cleaning pipeline together on a real text dataset.

**Typical pipeline order I now follow:**

```
raw text → lowercase → remove noise (regex) → tokenize → remove stop words → stem/lemmatize → n-grams (optional)
```

**Key insight**: stemming is fast but crude; lemmatization is slower but linguistically accurate. Choice depends on whether speed or precision matters more for the task.

---

## 3. Identifying Parts of Speech and Named Entities

**Section 22 · 6 lessons · 32min**

Once text is clean and tokenized, the next step is understanding grammatical role and real-world meaning.

- **Text tagging** — the general concept of attaching metadata (labels) to tokens.
- **POS tagging (Parts of Speech)** — labeling each word as a noun, verb, adjective, etc. Useful for understanding sentence structure and filtering (e.g., keep only nouns and adjectives for topic analysis).
- **Named Entity Recognition (NER)** — detecting and classifying real-world entities in text: people, organizations, locations, dates, etc.
- **Practical task**: applied both POS tagging and NER on a real text sample to extract structured meaning from unstructured sentences.

**Why this matters**: this is where NLP starts extracting *meaning*, not just cleaning format. POS + NER are the building blocks for information extraction.

---

## 4. Sentiment Analysis

**Section 23 · 6 lessons · 25min**

This section covers how to determine the emotional tone of text — positive, negative, or neutral.

- **What is sentiment analysis** — classifying the polarity/emotion behind a piece of text.
- **Rule-based sentiment analysis** — using predefined lexicons/dictionaries of positive and negative words (e.g., VADER) to score text without training a model.
- **Pre-trained transformer models** — using modern deep learning models (like those from Hugging Face) for higher-accuracy sentiment classification without training from scratch.
- **Practical task**: applied both approaches and compared rule-based vs. transformer-based results.

**Key insight**: rule-based methods are fast and interpretable but brittle with sarcasm/context; transformer models are heavier but far more accurate on nuanced text.

---

## 5. Vectorizing Text

**Section 24 · 4 lessons · 9min**

Machine learning models need numbers, not words — this section covers how to turn text into numerical vectors.

- **Numerical representation of text** — the core problem: how do you feed words into a model?
- **Bag of Words (BoW)** — represents text as word frequency counts, ignoring grammar and word order.
- **TF-IDF (Term Frequency–Inverse Document Frequency)** — improves on BoW by weighing words based on how unique/important they are across a whole document collection, not just how often they appear.

**Key insight**: BoW treats all frequent words as equally important; TF-IDF down-weights common words (even after stop-word removal) and boosts words that are more distinctive to a specific document — better signal for classification tasks.

---

## 6. Topic Modelling

**Section 25 · 9 lessons · 24min**

This section moves into **unsupervised** NLP — discovering hidden themes/topics in a collection of documents without any labels.

- **What is topic modelling** — grouping documents by underlying theme automatically.
- **When to use it** — useful for exploratory analysis of large, unlabeled text corpora.
- **Latent Dirichlet Allocation (LDA)** — a probabilistic model that assumes each document is a mix of topics, and each topic is a mix of words.
- **LDA in Python** — implemented using libraries like `gensim`.
- **Latent Semantic Analysis (LSA)** — an alternative approach using matrix decomposition (SVD) to find topic structure.
- **LSA in Python** — practical implementation.
- **Determining the number of topics** — techniques for choosing the right number of topics (e.g., coherence scores) rather than guessing.

**Key insight**: LDA and LSA solve the same problem (topic discovery) with different math — LDA is probabilistic, LSA is linear-algebra-based. The right number of topics matters as much as the algorithm choice.

---

## 7. Building a Custom Text Classifier

**Section 26 · 5 lessons · 17min**

Back to **supervised** NLP — training a model to classify text into known categories.

- **Building a custom text classifier** — overview of the workflow: vectorize text → train a classifier → evaluate.
- **Logistic Regression** — a solid baseline linear classifier for text classification tasks.
- **Naive Bayes** — a probabilistic classifier that works especially well with word-frequency-based features like BoW/TF-IDF.
- **Linear Support Vector Machine (SVM)** — effective for high-dimensional, sparse data like text vectors.

**Key insight**: text classification pipelines combine everything learned so far — preprocessing → vectorizing (BoW/TF-IDF) → feeding into a classic ML classifier (Logistic Regression, Naive Bayes, or SVM).

---

## 8. Case Study: Categorizing Fake News

**Section 27 · 9 lessons · 50min**

The capstone of the module — applying the entire pipeline end-to-end on a real, practical problem: detecting fake vs. real news articles.

Step-by-step workflow followed in the case study:

1. **Introducing the project** — defining the problem and dataset.
2. **Exploring data through POS tags** — using POS tagging as an exploratory tool to compare writing style between fake and real news.
3. **Extracting named entities** — using NER to see which people/organizations/places appear most in each category.
4. **Processing the text** — applying the full preprocessing pipeline (lowercase, clean, tokenize, remove stop words, lemmatize).
5. **Sentiment comparison** — checking whether fake news and real news differ in emotional tone.
6. **Topic modelling on fake news (Part 1 & 2)** — using LDA/LSA to see what topics dominate fake news articles.
7. **Categorizing fake news with a custom classifier** — training a supervised model (Logistic Regression / Naive Bayes / SVM) on vectorized text to predict fake vs. real.

**Why this section matters most**: it's the proof that every earlier section — preprocessing, POS/NER, sentiment, vectorizing, topic modelling, classification — isn't standalone theory. They chain together into one real pipeline for a real-world problem.

---

## 9. The Future of NLP

**Section 28 · 4 lessons · 8min**

The module closes by zooming out from classic NLP techniques toward where the field is headed.

- **What is deep learning** — brief conceptual intro for those without a deep learning background yet.
- **Deep learning for NLP** — how neural networks (RNNs, Transformers) outperform classic rule-based/statistical methods on complex language tasks.
- **Non-English NLP** — challenges and considerations when working with languages other than English (tokenization, available libraries/models, resource availability).
- **What's next for NLP** — the shift toward large language models (LLMs) and transformer-based architectures as the dominant approach going forward.

**Personal note**: this section is my bridge from classic NLP into deep learning and transformer-based models — the natural next step in my AI engineer journey.

---

## 10. Key Takeaways

A distilled, end-to-end mental model of the module:

```
1. Preprocess   → clean, tokenize, normalize (stem/lemmatize) raw text
2. Understand   → POS tagging + NER to extract grammatical & real-world meaning
3. Analyze      → sentiment analysis to gauge tone
4. Vectorize    → BoW / TF-IDF to turn text into numbers
5. Explore      → topic modelling (LDA/LSA) for unsupervised theme discovery
6. Classify     → Logistic Regression / Naive Bayes / SVM for supervised prediction
7. Apply        → combine all of the above into one pipeline (fake news case study)
8. Look ahead   → deep learning & transformers as the next evolution
```

**Biggest lesson learned**: NLP is not one technique — it's a *pipeline* of smaller, well-defined steps. Preprocessing quality directly determines how well every downstream step (tagging, vectorizing, modelling, classifying) performs. Garbage in, garbage out applies more to NLP than almost any other ML domain.

**Next steps for my journey** — moving from classic NLP into the LLMs Module:
- [ ] Section 29: Introduction to Large Language Models
- [ ] Section 30: The Transformer Architecture
- [ ] Section 31: Getting Started With GPT Models
- [ ] Section 32: Hugging Face Transformers
- [ ] Section 33: Question and Answer Models With BERT
- [ ] Section 34: Text Classification With XLNet
- [ ] Section 35: LangChain Module — Introduction

This is the natural continuation of Section 28 ("The Future of NLP") — everything above (preprocessing, POS/NER, sentiment, vectorizing, topic modelling, classic classifiers) was the foundation; the LLMs Module is where deep learning and Transformer-based architectures take over.

---

*Course: Udemy — Data Science / Machine Learning (NLP Module, Sections 20–28)*
*Repo: [ai-engineer-journey](https://github.com/juliao-martins-dev/ai-engineer-journey)*
