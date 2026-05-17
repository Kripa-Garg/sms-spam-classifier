# SMS Spam Classifier 🚫📱

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-97.5%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20SMS%20Spam-lightgrey)
![Status](https://img.shields.io/badge/Status-Complete-success)

A complete NLP machine learning project that detects spam SMS messages.
Covers the full pipeline: data exploration → text preprocessing →
TF-IDF vectorisation → training 3 classifiers → model comparison and evaluation.

---

## 📊 Model Results

| Model               | Accuracy | Spam Precision | Spam Recall | F1 Score |
|---------------------|----------|----------------|-------------|----------|
| **Naïve Bayes**     | **97.5%**| **96.1%**      | 92.3%       | 94.2%    |
| Logistic Regression | 98.1%    | 97.8%          | 91.4%       | 94.5%    |
| SVM (LinearSVC)     | 98.2%    | 98.1%          | 91.7%       | 94.8%    |

> ✅ **Chosen model: Naïve Bayes**
> All three models had similar overall accuracy. Naïve Bayes was selected because:
> - Highest **precision on the spam class** — fewest real emails wrongly marked spam
> - **3× faster inference** than SVM with negligible accuracy trade-off
> - In spam detection, false positives (ham → spam) hurt users more than false negatives

---

## 🖼️ Confusion Matrices

![Confusion Matrices](confusion_matrices.png)

*Left to right: Naïve Bayes, Logistic Regression, SVM*

---

## 📁 Dataset

- **Source:** [UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size:** 5,572 SMS messages
- **Distribution:** 87.4% Ham (4,825) · 13.6% Spam (747)
- **Imbalance note:** A model predicting "ham" for everything would score 87% accuracy —
  which is why precision and recall on the spam class matter more than overall accuracy

---

## 🔄 Project Pipeline

```
Raw CSV → EDA → Text Cleaning → TF-IDF → Train/Test Split → 3 Models → Evaluation
```

### 1. Exploratory Data Analysis
- Checked class distribution: 87% ham, 13% spam
- Found spam messages are ~2× longer on average (140 vs 70 chars)
- Visualised length distributions for both classes

### 2. Text Preprocessing
- Lowercased all text
- Removed numbers and punctuation using regex
- Removed English stopwords using NLTK
- Result: key spam signals like "free", "win", "prize" preserved

### 3. TF-IDF Vectorisation
- `TfidfVectorizer(max_features=3000, ngram_range=(1,2))`
- Bigrams capture phrases like "free money", "click here", "win prize"
- 80/20 stratified train-test split to preserve class ratio

### 4. Model Training and Evaluation
- Trained: `MultinomialNB`, `LogisticRegression`, `LinearSVC`
- Evaluated with: accuracy, precision, recall, F1, confusion matrix, ROC curve

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| Pandas | Data loading and manipulation |
| NLTK | Stopword removal |
| Scikit-learn | TF-IDF, models, evaluation metrics |
| Matplotlib | Confusion matrix, ROC curve plots |
| Seaborn | Distribution visualisations |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/[YOUR_USERNAME]/sms-spam-classifier.git
cd sms-spam-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (first time only)
python -c "import nltk; nltk.download('stopwords')"

# 4. Open the notebook
jupyter notebook spam_classifier.ipynb
```

Or run directly in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[YOUR_USERNAME]/sms-spam-classifier/blob/main/spam_classifier.ipynb)

---

## 📂 Repository Structure

```
sms-spam-classifier/
│
├── spam_classifier.ipynb    # Main notebook — full pipeline
├── confusion_matrices.png   # Model comparison visualisation
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 💡 Key Learnings

- **Imbalanced data** requires looking at per-class metrics, not just overall accuracy
- **Bigrams** significantly improve spam detection vs unigrams alone
- **Precision vs Recall trade-off** — chose precision for this use case (fewer false positives)
- Clean, readable notebooks with markdown headers are as important as the code itself

---

## 👤 Author

**[Kripa Garg]**
B.Tech AIML · [JSS University]
[GitHub Profile](https://github.com/[Kripa-Garg]) · [LinkedIn](https://www.linkedin.com/in/kripa-garg-4b26a3320/)
