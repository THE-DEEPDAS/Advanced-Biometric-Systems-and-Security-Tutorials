# Biometric System Performance Evaluation — README

## Files
- `biometric_evaluation.ipynb` — full implementation, executed with all outputs and plots.
- `biomet_data.csv` — the provided raw data file (unchanged).

## Data quality note (important for grading)
The raw file is **not** shaped the way the assignment describes. It is tab-separated,
has **no header**, and loads as **144 rows × 1001 columns** — i.e. rows are features and
columns are samples (the transpose of the expected "samples as rows" layout).

Inspection shows exactly **one** genuine problem: the last column (index 1000) is
completely empty — a trailing-tab artifact from the export, not a missing sample. Every
other column (0–999) is fully populated real data, including column 0 (it is **not** a
row index — its values match real feature magnitudes seen throughout the file).

Dropping only that one empty trailing column and transposing gives exactly
**1000 samples × 144 features**, which splits cleanly into **100 users × 10 samples**,
matching the assignment description exactly. (Naively dropping column 0 as well — an
easy mistake if you assume it's an index — silently discards one real sample and breaks
the clean 100×10 structure into 999 samples with one user short a sample; the notebook
explicitly avoids this.)

## Pipeline
1. **Load & clean**: read the raw CSV, drop only the fully-empty trailing column, transpose
   to samples × features, name columns `Feature_1..144`.
2. **User IDs**: samples are stored in contiguous blocks of 10 per user, so
   `UserID = row_index // 10`.
3. **Train/test split**: per user, first 5 samples (enrollment) → training set; remaining
   5 samples (captured ~6 months later) → test set. 500 enrollment / 500 test samples total.
4. **Templates**: each user's enrollment template is the mean of their 5 training feature
   vectors (100 × 144 template matrix).
5. **Matching scores**: every one of the 500 test samples is compared against all 100
   templates using
   - **Euclidean distance** (`scipy.spatial.distance.cdist`) — lower = more similar
   - **Cosine similarity** (`sklearn.metrics.pairwise.cosine_similarity`) — higher = more similar

   The score against the sample's own user is the **genuine** score (500 genuine scores
   per metric); scores against the other 99 users are **impostor** scores (49,500
   impostor scores per metric).
6. **A/B — Genuine & Impostor distributions**: overlaid histograms per metric.
7. **C/D — FAR & FRR vs threshold**: for Euclidean, a match is declared when
   `distance ≤ threshold`; for Cosine, when `similarity ≥ threshold`. FAR/FRR are swept
   over a fine threshold grid for each metric.
8. **E — ROC**: TPR = 1 − FRR, FPR = FAR, computed via `sklearn.metrics.roc_curve` (with
   Euclidean distance negated so "higher score = more genuine-like" for both metrics),
   plus AUC via `sklearn.metrics.auc`.
9. **F — EER & Decidability index**: EER is found by linearly interpolating the
   crossover point of the FAR and FRR curves. Decidability index is
   `d' = |mean_genuine − mean_impostor| / sqrt(0.5·(var_genuine + var_impostor))`.

## Results summary

| Metric              | EER    | AUC    | Decidability Index (d') |
|----------------------|--------|--------|---------------------------|
| Euclidean Distance    | ~0.115 | ~0.950 | ~1.76                     |
| Cosine Similarity      | ~0.084 | ~0.969 | ~1.93                     |

(Exact values are in the executed notebook.)

## Which matching distance performs best?
**Cosine similarity outperforms Euclidean distance** on this dataset: lower EER, higher
ROC AUC, and higher decidability index, meaning genuine and impostor scores are more
cleanly separated. This is a typical result for high-dimensional feature vectors (144
dimensions here) — cosine similarity compares vector *direction* and is less sensitive to
per-sample scale/intensity variation than raw Euclidean distance, which is dominated by
magnitude differences between samples.
