# Detecting and Anonymizing PII in Educational Texts Using NLP

This project focuses on **detecting Personally Identifiable Information (PII)** in student essays using **Natural Language Processing (NLP)** and **deep learning**.  
The final model uses **BERT for token classification** to identify and label PII spans at the token level, enabling downstream anonymization in educational data.

---

## 🧩 Problem Overview

Educational platforms collect large volumes of free-text responses from students.  
These texts often contain sensitive information such as:

- Names  
- Email addresses  
- Usernames  
- Phone numbers  
- ID numbers  
- Personal URLs  
- Street addresses  

The goal of this project is to:

- **Detect PII at the token level** in educational texts  
- **Support anonymization** while preserving the utility of the data for research/analytics  
- Explore both **traditional ML baselines** and **transformer-based models** for PII detection  

---

## 📊 Dataset

You can download the dataset from Google Drive:  
[📁 Dataset – Google Drive](https://drive.google.com/drive/folders/1rQ6s9lUbNmEyQkhjmvYwDmZLSPhaXenP?usp=drive_link)

The project works with a dataset of ~**22,000 student essays**, where each essay is tokenized and annotated with PII labels.

Key characteristics:

- **Document ID** – unique identifier for each essay  
- **Text** – full essay text in UTF-8  
- **Tokens & Labels** – tokenized text with BIO-style labels for PII

PII label types include (BIO format in the model):

- `NAME_STUDENT`
- `EMAIL`
- `USERNAME`
- `ID_NUM`
- `PHONE_NUM`
- `URL_PERSONAL`
- `STREET_ADDRESS`
- `O` (non-PII tokens)

> **Note:** In the notebook, data is loaded from local JSON/CSV files (e.g., `train.json`, `test.json`, `pii_dataset.csv`).  
> For GitHub, you can place them under a `data/` folder and update the paths accordingly, or omit the data and document how to obtain it.

---

## 🛠️ Project Workflow

### 1. Data Import & Preprocessing

- Load raw data from JSON/CSV into **Pandas DataFrames**
- Clean and normalize:
  - Remove malformed tokens and extra whitespace
  - Align tokens and labels correctly
  - Add `[CLS]` and `[SEP]` markers for BERT
- Create:
  - **Token-level labels** for sequence tagging (NER-style)
  - **Binary document label** (PII present vs not) for baseline models

### 2. Exploratory Data Analysis (EDA)

- Analyze label distribution (PII vs non-PII)
- Inspect sentence lengths and PII density
- Visualize token frequency by label
- Understand which PII categories are frequent vs rare  
  (important later for class imbalance and model evaluation)

### 3. Baseline Models (Document-Level)

As a starting point, simple document-level classifiers are trained to detect whether an essay contains any PII:

- **Feature preparation**
  - Join tokens into a single string per document
  - Vectorize using **TF-IDF**
- **Models**
  - Logistic Regression  
  - Naive Bayes  
  - Random Forest  

**Results (document-level, binary classification):**

- Logistic Regression – accuracy ≈ **0.905**
- Naive Bayes – accuracy ≈ **0.905**
- Random Forest – accuracy ≈ **0.903**

These baselines show that traditional models can detect the presence of PII in a document, but they **cannot localize PII spans** at the token level.

### 4. BERT Token-Level Model

To detect PII spans precisely, the project uses:

- **Model:** `bert-base-uncased` via `BertForTokenClassification`  
- **Tokenizer:** `BertTokenizerFast`  
- **Task:** Token-level classification (NER-style)

#### Sliding Window Tokenization

Because essays can exceed BERT’s 512-token limit, a **sliding window** strategy is used:

- Windows explored:
  - `max_length=128, stride=64`
  - `max_length=256, stride=128`
  - `max_length=512, stride=256`
- Overlapping windows preserve context across segments
- Each window is tokenized and aligned with corresponding labels

#### Handling Class Imbalance

- Compute **class weights** based on label frequencies
- Use a **weighted CrossEntropyLoss** to penalize misclassification of rare PII classes more heavily
- Ignore padding / special tokens with an `ignore_index` label

#### Training Details

- Optimizer: **AdamW**
- Scheduler: **linear warmup/decay** (`get_linear_schedule_with_warmup`)
- Batch size: `32`
- Epochs: `10`
- Early stopping based on validation performance

You can download the trained BERT model weights here:  
[🧠 Model Weights – Google Drive](https://drive.google.com/file/d/1pGILsq47kb-9QernAZyqpoX1FvartJKx/view?usp=drive_link)
---

## ✅ Results (Token-Level BERT)

On the test set, the BERT model achieves:

- **Test Loss:** ≈ `0.0089`  
- **Test Accuracy (token-level):** ≈ **0.9513**

From the classification report:

- Non-PII (`O`) tokens: **very high F1 (~0.98)**
- Frequent PII classes like:
  - `B-EMAIL`, `B-PHONE_NUM`, `I-PHONE_NUM` show **strong precision and recall**
- Rare classes like:
  - `B-ID_NUM`, `B-STREET_ADDRESS`, `I-URL_PERSONAL` suffer due to **class imbalance**, with lower F1 scores

Overall, **BERT significantly outperforms the document-level baselines**, and is capable of **locating PII spans** rather than just flagging documents.

---

## 📁 Repository Structure

Suggested structure for the GitHub repo:

```text
├── Main code/
│   └── code.ipynb   # main project notebook (this file)
├── README.md
└── requirements.txt         # (optional) Python dependencies
