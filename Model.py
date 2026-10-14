"""
CP219 Project 1 - Is a document written by a human (A) or a machine (B)?

Each document is a list of integer token IDs (the words are hidden).
This script trains a three-level stacked model and writes submission.csv.

    Step 1  Features
            - TF-IDF over token 1-, 2- and 3-grams (about 349k columns)
            - 30 style statistics per document (length, vocabulary, variance, ...)
    Step 2  Level 1: models that turn the raw features into 31 "opinion scores"
            - 6 phrase models (logistic regressions, a small MLP, 3 linear SVMs)
            - 25 language-model scores (bigram + trigram, human vs machine)
    Step 3  Level 2: logistic MLP, RBF SVM and gradient boosting read the
            31 scores + 30 statistics and each output P(machine)
    Step 4  Level 3: combine the three level-2 probabilities into one answer

Every score used for training a later level is "out-of-fold": it comes from a
model that never saw that document's label (5-fold cross-validation).

Cross-validated accuracy: 0.9529 (stack)  |  0.9526 (blend)

Usage
    python cp219_model.py DATA_FOLDER/                       # best model (stack)
    python cp219_model.py DATA_FOLDER/ out.csv --blend       # second model (blend)
DATA_FOLDER must contain train.json and test.json.
Needs: numpy, pandas, scipy, scikit-learn.  Runtime: about 6 minutes on one CPU.
"""

import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
VOCAB_SIZE = 18438      # token IDs run from 0 to 18437
FULL_STOP = 8294        # this token behaves like "." (used to split sentences)
WINDOW = 20             # window size (tokens) for local statistics
N_FOLDS = 5

START = time.time()


def log(message):
    print(f"[{time.time() - START:5.0f}s] {message}", flush=True)


def safe_var(values):
    """Variance, or 0 if there are fewer than two values."""
    return float(np.var(values)) if len(values) > 1 else 0.0


def to_logit(p):
    """Turn a probability into a logit, log(p / (1 - p))."""
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


# ============================================================================
# STEP 1a - Text for TF-IDF
# ============================================================================
def document_to_text(tokens):
    """Write token IDs as words, plus a few position markers
    (first two tokens, last token, length group, ends with a full stop)."""
    n = len(tokens)
    markers = [
        f"f0_{tokens[0]}",
        f"f1_{tokens[1] if n > 1 else 'x'}",
        f"l_{tokens[-1]}",
        f"lenb_{min(n // 50, 20)}",
        f"endsent_{int(tokens[-1] == FULL_STOP)}",
    ]
    return " ".join(map(str, tokens)) + " " + " ".join(markers)


# ============================================================================
# STEP 1b - 30 style statistics per document
# ============================================================================
def count_tokens(documents):
    """How often each token ID appears across all given documents."""
    counts = np.zeros(VOCAB_SIZE)
    for doc in documents:
        np.add.at(counts, np.asarray(doc), 1)
    return counts


def document_statistics(tokens, corpus_counts, corpus_total, log_rank, is_training_doc):
    """Return a dict of 30 statistics describing one document."""
    s = np.asarray(tokens)
    n = len(s)
    token_counts = Counter(tokens)
    counts = np.array(list(token_counts.values()))
    shares = counts / n
    pairs = list(zip(tokens, tokens[1:]))
    triples = list(zip(tokens, tokens[1:], tokens[2:]))
    entropy = -(shares * np.log2(shares)).sum()

    # --- length and vocabulary (14)
    f = dict(
        length=np.log(n),
        n_unique=np.log(len(token_counts)),
        ttr=len(token_counts) / n,                                  # type-token ratio
        hapax_ratio=(counts == 1).sum() / len(token_counts),        # tokens used once
        unigram_entropy=entropy,
        norm_entropy=entropy / np.log2(n) if n > 1 else 0.0,
        oov_rate=(s == 0).mean(),                                   # token 0 = rare word
        top1_share=counts.max() / n,
        rep_bigram_frac=(1 - len(set(pairs)) / len(pairs)) if pairs else 0.0,
        rep_trigram_frac=(1 - len(set(triples)) / len(triples)) if triples else 0.0,
        immediate_repeat=(s[1:] == s[:-1]).mean() if n > 1 else 0.0,
        mean_token_id=s.mean(),
        std_token_id=s.std(),
        yules_k=1e4 * ((counts ** 2).sum() - n) / n ** 2 if n > 1 else 0.0,
    )

    # --- word rarity ("surprisal") and its variance (7)
    # For a training document we remove its own tokens from the corpus counts,
    # so training and test documents are measured the same way.
    token_freq = corpus_counts[s].copy()
    total = corpus_total
    if is_training_doc:
        token_freq -= np.array([token_counts[t] for t in tokens])
        total = corpus_total - n
    surprisal = -np.log((token_freq + 1) / (total + VOCAB_SIZE))

    f.update(
        mean_surprisal=surprisal.mean(),
        var_surprisal=safe_var(surprisal),
        cv_surprisal=np.sqrt(safe_var(surprisal)) / surprisal.mean(),
        var_log_rank=safe_var(log_rank[s]),
        var_type_count=np.log1p(safe_var(counts)),
        var_surprisal_diff=safe_var(np.diff(surprisal)),
        var_token_id=safe_var(s),
    )

    # --- sentences and rhythm (9)
    stops = np.where(s == FULL_STOP)[0]
    tail = [n - 1] if s[-1] != FULL_STOP else []
    sentence_lengths = np.diff(np.concatenate([[-1], stops, tail]))
    sentence_lengths = sentence_lengths[sentence_lengths > 0]
    f.update(
        n_sentences=np.log1p(len(sentence_lengths)),
        mean_sent_len=np.log(sentence_lengths.mean()),
        var_sent_len=np.log1p(safe_var(sentence_lengths)),
        cv_sent_len=(np.sqrt(safe_var(sentence_lengths)) / sentence_lengths.mean()
                     if len(sentence_lengths) > 1 else 0.0),
    )

    bounds = np.concatenate([[0], stops + 1, [n]])
    f["var_sent_surprisal"] = safe_var(
        [surprisal[i:j].mean() for i, j in zip(bounds[:-1], bounds[1:]) if j > i])

    if n >= 2 * WINDOW:
        starts = range(0, n - WINDOW + 1, WINDOW)
        f["var_window_ttr"] = safe_var([len(set(s[i:i + WINDOW])) / WINDOW for i in starts])
        f["var_window_surprisal"] = safe_var([surprisal[i:i + WINDOW].mean() for i in starts])
    else:
        f["var_window_ttr"] = f["var_window_surprisal"] = -1.0     # too short to measure

    last_seen, gaps = {}, []
    for i, t in enumerate(tokens):
        if t in last_seen:
            gaps.append(i - last_seen[t])
        last_seen[t] = i
    f["mean_repeat_gap"] = np.log(np.mean(gaps)) if gaps else -1.0
    f["var_log_repeat_gap"] = safe_var(np.log(gaps)) if len(gaps) > 1 else -1.0
    return f


def statistics_table(train_docs, test_docs):
    """30 statistics for every document. Corpus counts come from training text only."""
    counts = count_tokens(train_docs)
    total = counts.sum()
    rank = np.empty(VOCAB_SIZE)
    rank[np.argsort(-counts)] = np.arange(1, VOCAB_SIZE + 1)
    log_rank = np.log(rank)
    train = pd.DataFrame([document_statistics(d, counts, total, log_rank, True) for d in train_docs])
    test = pd.DataFrame([document_statistics(d, counts, total, log_rank, False) for d in test_docs])
    return train, test


# ============================================================================
# STEP 2a - Language models (one trained on human text, one on machine text)
# ============================================================================
class BigramLM:
    """Predicts each token from the previous token (absolute discounting,
    falling back to single-token frequencies for unseen pairs)."""

    def __init__(self, discount=0.75):
        self.d = discount

    def fit(self, docs):
        tokens = np.concatenate(docs)
        self.N = len(tokens)
        self.unigram = np.bincount(tokens, minlength=VOCAB_SIZE).astype(np.float64)
        prev = np.concatenate([d[:-1] for d in docs])
        cur = np.concatenate([d[1:] for d in docs])
        keys, cnt = np.unique(prev.astype(np.int64) * VOCAB_SIZE + cur, return_counts=True)
        self.keys, self.cnt = keys, cnt.astype(np.float64)
        prev_of_key = keys // VOCAB_SIZE
        self.context_total = np.bincount(prev_of_key, weights=self.cnt, minlength=VOCAB_SIZE)
        self.context_types = np.bincount(prev_of_key, minlength=VOCAB_SIZE).astype(np.float64)
        self.p_unigram = (self.unigram + 1) / (self.N + VOCAB_SIZE)
        return self

    def token_logp(self, doc):
        """log P(token | previous token) for every token in the document."""
        doc = np.asarray(doc, dtype=np.int64)
        logp = np.empty(len(doc))
        logp[0] = np.log(self.p_unigram[doc[0]])
        prev, cur = doc[:-1], doc[1:]
        key = prev * VOCAB_SIZE + cur
        pos = np.minimum(np.searchsorted(self.keys, key), len(self.keys) - 1)
        pair_count = np.where(self.keys[pos] == key, self.cnt[pos], 0.0)
        ctx = self.context_total[prev]
        backoff = np.where(ctx > 0, self.d * self.context_types[prev] / np.maximum(ctx, 1), 1.0)
        p = (np.where(ctx > 0, np.maximum(pair_count - self.d, 0) / np.maximum(ctx, 1), 0.0)
             + backoff * self.p_unigram[cur])
        logp[1:] = np.log(p)
        return logp

    def unigram_logp(self, doc):
        return np.log(self.p_unigram[np.asarray(doc)])


class TrigramLM(BigramLM):
    """Predicts each token from the previous two tokens, falling back to the bigram model."""

    def fit(self, docs):
        super().fit(docs)
        docs3 = [d for d in docs if len(d) > 2]
        p2 = np.concatenate([d[:-2] for d in docs3]).astype(np.int64)
        p1 = np.concatenate([d[1:-1] for d in docs3]).astype(np.int64)
        cur = np.concatenate([d[2:] for d in docs3]).astype(np.int64)
        keys, cnt = np.unique((p2 * VOCAB_SIZE + p1) * VOCAB_SIZE + cur, return_counts=True)
        self.k3, self.c3 = keys, cnt.astype(np.float64)
        ctx, inverse = np.unique(keys // VOCAB_SIZE, return_inverse=True)
        self.ctx3_keys = ctx
        self.ctx3_total = np.bincount(inverse, weights=self.c3)
        self.ctx3_types = np.bincount(inverse).astype(np.float64)
        return self

    def token_logp3(self, doc):
        """log P(token | previous two tokens) for every token in the document."""
        doc = np.asarray(doc, dtype=np.int64)
        logp_bigram = self.token_logp(doc)
        if len(doc) < 3:
            return logp_bigram
        p_bigram = np.exp(logp_bigram[2:])
        key = (doc[:-2] * VOCAB_SIZE + doc[1:-1]) * VOCAB_SIZE + doc[2:]
        pos = np.minimum(np.searchsorted(self.k3, key), len(self.k3) - 1)
        tri_count = np.where(self.k3[pos] == key, self.c3[pos], 0.0)
        ctx_key = key // VOCAB_SIZE
        cpos = np.minimum(np.searchsorted(self.ctx3_keys, ctx_key), len(self.ctx3_keys) - 1)
        seen = self.ctx3_keys[cpos] == ctx_key
        ctx = np.where(seen, self.ctx3_total[cpos], 0.0)
        types = np.where(seen, self.ctx3_types[cpos], 0.0)
        backoff = np.where(ctx > 0, self.d * types / np.maximum(ctx, 1), 1.0)
        p = np.where(ctx > 0, np.maximum(tri_count - self.d, 0) / np.maximum(ctx, 1), 0.0) + backoff * p_bigram
        out = logp_bigram.copy()
        out[2:] = np.log(p)
        return out


def bigram_scores(lm_human, lm_machine, doc):
    """13 scores. LLR = log P(machine) - log P(human) per token; > 0 means machine-like."""
    lp_h, lp_m = lm_human.token_logp(doc), lm_machine.token_logp(doc)
    llr = lp_m - lp_h
    llr_unigram = lm_machine.unigram_logp(doc) - lm_human.unigram_logp(doc)
    n = len(doc)
    return [llr.mean(), llr.sum() / np.sqrt(n), llr.std(),
            np.percentile(llr, 10), np.median(llr), np.percentile(llr, 90), (llr > 0).mean(),
            llr_unigram.mean(), llr_unigram.sum() / np.sqrt(n),
            lp_h.mean(), lp_m.mean(), lp_h.var(), lp_m.var()]


def trigram_scores(lm_human, lm_machine, doc):
    """12 scores, including where in the document the text turns machine-like."""
    llr = lm_machine.token_logp3(doc) - lm_human.token_logp3(doc)
    n = len(llr)
    thirds = [part.mean() if len(part) else 0.0 for part in np.array_split(llr, 3)]
    w = min(20, n)
    moving_avg = np.convolve(llr, np.ones(w) / w, mode="valid")
    slope = np.polyfit(np.arange(n) / n, llr, 1)[0] if n > 2 else 0.0
    head = llr[:min(30, n)].mean()
    rest = llr[30:].mean() if n > 30 else head
    return [llr.mean(), llr.sum() / np.sqrt(n), llr.std(), (llr > 0).mean(), *thirds,
            moving_avg.max(), moving_avg.min(), moving_avg.std(), slope, rest - head]


# ============================================================================
# STEP 2b - Level 1: out-of-fold "opinion scores"
# ============================================================================
def naive_bayes_weights(X, y, alpha=1.0):
    """How one-sided each phrase is: log(share in machine docs / share in human docs)."""
    present = (X > 0).astype(np.float32)
    p = alpha + np.asarray(present[y == 1].sum(0)).ravel()
    q = alpha + np.asarray(present[y == 0].sum(0)).ravel()
    return np.log((p / p.sum()) / (q / q.sum())).astype(np.float32)


def phrase_model_scores(X_fit, S_fit, y_fit, X_new, S_new, docs_fit, docs_new):
    """Train level-1 models on one fold and score new documents.
    Returns 3 phrase scores + 13 bigram scores per document."""
    # 1) logistic regression on TF-IDF + statistics
    A = sp.hstack([X_fit, sp.csr_matrix(S_fit * 0.3)]).tocsr()
    B = sp.hstack([X_new, sp.csr_matrix(S_new * 0.3)]).tocsr()
    lr = LogisticRegression(C=10, solver="liblinear").fit(A, y_fit).decision_function(B)

    # 2) Naive-Bayes-weighted logistic regression
    w = naive_bayes_weights(X_fit, y_fit)
    nblr = (LogisticRegression(C=4, solver="liblinear")
            .fit(X_fit.multiply(w).tocsr(), y_fit)
            .decision_function(X_new.multiply(w).tocsr()))

    # 3) logistic-activation MLP on the 20,000 phrases with the highest chi-squared score
    chi, _ = chi2(X_fit, y_fit)
    top = np.argsort(-np.nan_to_num(chi))[:20000]
    mlp = MLPClassifier((128,), activation="logistic", alpha=1e-4, max_iter=40,
                        early_stopping=True, n_iter_no_change=5, random_state=0).fit(X_fit[:, top], y_fit)
    p = np.clip(mlp.predict_proba(X_new[:, top])[:, 1], 1e-6, 1 - 1e-6)
    mlp_score = np.log(p / (1 - p))

    # 4) bigram language models: one for human text, one for machine text
    lm_h = BigramLM().fit([d for d, t in zip(docs_fit, y_fit) if t == 0])
    lm_m = BigramLM().fit([d for d, t in zip(docs_fit, y_fit) if t == 1])
    lm = np.array([bigram_scores(lm_h, lm_m, d) for d in docs_new])

    return np.column_stack([lr, nblr, mlp_score, lm])


def svm_scores(X_fit, S_fit, y_fit, X_new, S_new):
    """Three linear SVMs: TF-IDF, NB-weighted presence, plain presence."""
    A = sp.hstack([X_fit, sp.csr_matrix(S_fit * 0.3)]).tocsr()
    B = sp.hstack([X_new, sp.csr_matrix(S_new * 0.3)]).tocsr()
    svm_tfidf = LinearSVC(C=0.5, max_iter=5000).fit(A, y_fit).decision_function(B)

    w = naive_bayes_weights(X_fit, y_fit)
    P_fit = (X_fit > 0).astype(np.float32)
    P_new = (X_new > 0).astype(np.float32)
    svm_nb = (LinearSVC(C=0.05, max_iter=5000)
              .fit(P_fit.multiply(w).tocsr(), y_fit)
              .decision_function(P_new.multiply(w).tocsr()))
    svm_presence = LinearSVC(C=0.02, max_iter=5000).fit(P_fit, y_fit).decision_function(P_new)
    return np.column_stack([svm_tfidf, svm_nb, svm_presence])


def level1(X_tr, S_tr, y, docs_tr, X_te, S_te, docs_te):
    """31 out-of-fold scores for training docs; test scores = average over the 5 fold models."""
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(S_tr, y))
    blocks_tr, blocks_te = [], []

    # (a) phrase models + bigram LMs: 16 scores
    oof, test = np.zeros((len(y), 16)), np.zeros((len(docs_te), 16))
    for k, (fit, val) in enumerate(folds):
        scores = phrase_model_scores(
            X_tr[fit], S_tr[fit], y[fit],
            sp.vstack([X_tr[val], X_te]).tocsr(), np.vstack([S_tr[val], S_te]),
            [docs_tr[i] for i in fit], [docs_tr[i] for i in val] + docs_te)
        oof[val] = scores[:len(val)]
        test += scores[len(val):] / N_FOLDS
        log(f"level 1 phrase models + bigram LMs, fold {k + 1}/{N_FOLDS}")
    blocks_tr.append(oof); blocks_te.append(test)

    # (b) trigram LMs with position features: 12 scores
    oof, test = np.zeros((len(y), 12)), np.zeros((len(docs_te), 12))
    for k, (fit, val) in enumerate(folds):
        lm_h = TrigramLM().fit([docs_tr[i] for i in fit if y[i] == 0])
        lm_m = TrigramLM().fit([docs_tr[i] for i in fit if y[i] == 1])
        oof[val] = [trigram_scores(lm_h, lm_m, docs_tr[i]) for i in val]
        test += np.array([trigram_scores(lm_h, lm_m, d) for d in docs_te]) / N_FOLDS
    log("level 1 trigram LMs done")
    blocks_tr.append(oof); blocks_te.append(test)

    # (c) linear SVMs: 3 scores
    oof, test = np.zeros((len(y), 3)), np.zeros((len(docs_te), 3))
    for fit, val in folds:
        scores = svm_scores(X_tr[fit], S_tr[fit], y[fit],
                            sp.vstack([X_tr[val], X_te]).tocsr(), np.vstack([S_tr[val], S_te]))
        oof[val] = scores[:len(val)]
        test += scores[len(val):] / N_FOLDS
    log("level 1 linear SVMs done")
    blocks_tr.append(oof); blocks_te.append(test)

    return np.hstack(blocks_tr), np.hstack(blocks_te)


# ============================================================================
# STEP 3 - Level 2: three different models on the 61 features
# ============================================================================
class LogisticMLP:
    """Neural network with sigmoid (logistic) hidden layers of 64 and 32 units,
    averaged over 5 random starts."""

    def __init__(self, n_seeds=5):
        self.n_seeds = n_seeds

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.nets = [MLPClassifier((64, 32), activation="logistic", solver="adam", alpha=1e-3,
                                   learning_rate_init=1e-3, batch_size=128, max_iter=300,
                                   early_stopping=True, validation_fraction=0.1,
                                   n_iter_no_change=15, random_state=seed).fit(Z, y)
                     for seed in range(self.n_seeds)]
        return self

    def predict_proba(self, X):
        Z = self.scaler.transform(X)
        return np.mean([net.predict_proba(Z)[:, 1] for net in self.nets], axis=0)


class ProbabilityOfMachine:
    """Wrap a scikit-learn classifier so predict_proba returns only P(machine)."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


LEVEL2_MODELS = {
    "logistic MLP": lambda: LogisticMLP(),
    "RBF SVM": lambda: ProbabilityOfMachine(make_pipeline(
        StandardScaler(), SVC(C=1.0, gamma="scale", probability=True, random_state=0))),
    "gradient boosting": lambda: ProbabilityOfMachine(HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=600, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=0)),
}


def level2(Z_tr, y, Z_te):
    """Out-of-fold P(machine) for training docs, and P(machine) for test docs,
    from each level-2 model."""
    oof = np.zeros((len(y), len(LEVEL2_MODELS)))
    test = np.zeros((len(Z_te), len(LEVEL2_MODELS)))
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=100).split(Z_tr, y))
    for j, (name, make_model) in enumerate(LEVEL2_MODELS.items()):
        for fit, val in folds:
            oof[val, j] = make_model().fit(Z_tr[fit], y[fit]).predict_proba(Z_tr[val])
        test[:, j] = make_model().fit(Z_tr, y).predict_proba(Z_te)
        log(f"level 2 {name:18s} cross-validated accuracy {((oof[:, j] > 0.5) == y).mean():.4f}")
    return oof, test


# ============================================================================
# STEP 4 - Level 3: combine the three level-2 probabilities
# ============================================================================
def combine_by_stacking(oof, y, test):
    """Best model: logistic regression on the three level-2 logits."""
    judge = LogisticRegression(C=1.0).fit(to_logit(oof), y)
    log(f"level 3 stacking weights {judge.coef_[0].round(3)} (MLP, SVM, boosting)")
    return judge.predict_proba(to_logit(test))[:, 1]


def combine_by_blending(oof, y, test):
    """Second model: weighted average; weights chosen to minimise log-loss."""
    def loss(w):
        w = np.abs(w) / np.abs(w).sum()
        return log_loss(y, (oof @ w).clip(1e-6, 1 - 1e-6))
    w = minimize(loss, np.ones(oof.shape[1]) / oof.shape[1], method="Nelder-Mead",
                 options=dict(maxiter=2000)).x
    w = np.abs(w) / np.abs(w).sum()
    log(f"level 3 blend weights {w.round(3)} (MLP, SVM, boosting)")
    return test @ w


# ============================================================================
# Main
# ============================================================================
def main(data_folder, output_file, use_blend):
    train = pd.read_json(data_folder + "train.json", lines=True)
    test = pd.read_json(data_folder + "test.json", lines=True)
    y = (train.label == "B").astype(int).values            # 1 = machine, 0 = human
    docs_tr = [np.asarray(d) for d in train.text]
    docs_te = [np.asarray(d) for d in test.text]

    # Step 1: features
    tfidf = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 3), min_df=2, max_df=0.95,
                            sublinear_tf=True, dtype=np.float32)
    X_tr = tfidf.fit_transform(train.text.map(document_to_text)).tocsr()
    X_te = tfidf.transform(test.text.map(document_to_text)).tocsr()

    stats_tr, stats_te = statistics_table(train.text.tolist(), test.text.tolist())
    to_normal = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0)
    S_tr = to_normal.fit_transform(stats_tr).astype(np.float32)
    S_te = to_normal.transform(stats_te).astype(np.float32)
    log(f"features ready: {X_tr.shape[1]} TF-IDF columns, {S_tr.shape[1]} statistics")

    # Step 2: level 1 -> 31 scores, then add the 30 statistics = 61 features
    L1_tr, L1_te = level1(X_tr, S_tr, y, docs_tr, X_te, S_te, docs_te)
    Z_tr = np.hstack([L1_tr, S_tr])
    Z_te = np.hstack([L1_te, S_te])
    log(f"level-2 input: {Z_tr.shape[1]} features per document")

    # Step 3: level 2
    oof, test_probs = level2(Z_tr, y, Z_te)

    # Step 4: level 3
    combine = combine_by_blending if use_blend else combine_by_stacking
    p_machine = combine(oof, y, test_probs)

    submission = pd.DataFrame({"id": test.id, "label": np.where(p_machine > 0.5, "B", "A")})
    submission.to_csv(output_file, index=False)
    log(f"wrote {len(submission)} predictions to {output_file}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    data_folder = args[0] if args else "./"
    if not data_folder.endswith("/"):
        data_folder += "/"
    output_file = args[1] if len(args) > 1 else "submission.csv"
    main(data_folder, output_file, use_blend="--blend" in sys.argv)
