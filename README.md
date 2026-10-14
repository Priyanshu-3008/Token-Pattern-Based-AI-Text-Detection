# Human vs Machine Text Detection

A machine-learning project that classifies tokenized documents as **human-written (A)** or **machine-generated (B)**. 

## Project overview

The dataset represents each document as a sequence of integer token IDs rather than readable text. The goal is to learn patterns that distinguish human writing from machine-generated writing and predict a label for each document.

The implementation uses a **three-level stacked ensemble** that combines phrase-based features, writing-style statistics, and token-sequence predictability.

## Approach

### 1. Feature engineering

- **TF-IDF word/token n-grams:** Unigrams, bigrams, and trigrams capture token and phrase patterns.
- **Position markers:** Features include the first tokens, last token, document-length group, and whether the document ends with a full-stop token.
- **30 writing-style statistics:** Document length, vocabulary diversity, rare-token rate, repeated phrases, sentence-length patterns, and variability across the document.
- **Bigram and trigram language models:** Separate models are trained on human and machine documents. Their log-likelihood differences estimate how characteristic a token sequence is of either class.

### 2. Level 1 — specialist models

Linear models, an MLP, linear SVM variants, and class-specific language models produce **31 out-of-fold scores** per document. Out-of-fold predictions are used so that each training document is scored by a model that did not train on that document.

### 3. Level 2 — complementary classifiers

A logistic-activation MLP, an RBF SVM, and a gradient-boosting classifier use the specialist scores together with the 30 style statistics. Their different learning strategies help capture complementary patterns.

### 4. Level 3 — final combination

A logistic-regression model combines the Level-2 predictions into the final machine-writing probability. The script also supports an alternative weighted-blending option.

## Reported results

The accompanying project report records the following cross-validation results:

| Model / metric | Reported result |
|---|---:|
| Final stacked model accuracy | 0.9529 (95.29%) |
| Final stacked model AUC | 0.988 |
| Weighted blend accuracy | 0.9526 (95.26%) |

These are the results reported for the project’s evaluation pipeline; they are not a guarantee of performance on a different dataset. Reproducing them requires the original dataset and matching software environment.

## Repository contents

```text
.
├── README.md
└── cp219_model.py
```

- `cp219_model.py` — feature engineering, model training, cross-validation-based stacking, and prediction export.
- `README.md` — project overview, method summary, results, and usage instructions.

## Requirements

- Python 3.9 or newer
- NumPy
- pandas
- SciPy
- scikit-learn

Install the dependencies:

```bash
python -m pip install numpy pandas scipy scikit-learn
```

## Data setup

The script expects two JSON Lines files in a local data directory:

```text
data/
├── train.json
└── test.json
```

Each line should be a JSON object. The training records must contain `id`, `text` (a list of token IDs), and `label` (`A` for human-written or `B` for machine-generated). Test records must contain `id` and `text`.

**The dataset is not included in this repository.** Use only data you are authorized to access and redistribute. The original competition rules restricted redistribution of its dataset. Do not upload course-provided or dataset unless you have permission to redistribute it.

## How to run

Run these commands from the repository root after placing the authorized data files in `data/`.

### Default: stacked model

```bash
python cp219_model.py data/
```

The script writes `submission.csv` in the current directory.

### Alternative: weighted blend

```bash
python cp219_model.py data/ blend.csv --blend
```

The script writes `blend.csv`.

The output CSV contains `id,label`, with one predicted label (`A` or `B`) for each test document.

## Important implementation notes

- The code uses five-fold stratified cross-validation to create out-of-fold predictions for the stacking pipeline.
- The input tokens are anonymized IDs; the original words cannot be reconstructed from the token IDs alone.
- The script suppresses warnings, so inspect and validate the environment if you encounter unexpected behaviour.
- Runtime depends on hardware and library versions. The original code comments estimate about six minutes on one CPU core, but this may vary.

## Limitations

- The model depends on the original tokenized dataset and its token conventions, including token ID `8294` as the full-stop marker and token ID `0` as the rare/out-of-vocabulary marker.
- Strong reported results on this datasetset do not establish generalization to arbitrary text, other tokenization schemes, or new language models.
- The dataset is tokenized and obfuscated, so predictions cannot be explained using the original readable words.
- Reproduction should use a fixed validation protocol and clearly distinguish cross-validation scores from leaderboard or held-out test results.


