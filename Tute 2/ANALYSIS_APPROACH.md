# Age-Gap Face Recognition Analysis: Full Approach

## Objective

This analysis evaluates whether two images belong to the same person across a large age gap. Each subject contributes one genuine pair, while random cross-subject pairs provide impostor comparisons.

## Dataset

The pipeline uses `age_gap_dataset`, whose structure is:

```text
age_gap_dataset/
├── metadata.csv
├── subject_001/
│   ├── age_00.jpg
│   └── age_54.jpg
├── subject_002/
│   ├── age_03.jpg
│   └── age_54.jpg
└── ...
```

The CSV columns are:

```text
subject_id,image_1,age_1,image_2,age_2,age_gap_years
```

Image paths are resolved relative to `age_gap_dataset`. The current dataset contains 82 subjects and 82 genuine pairs.

## Metadata and image preprocessing

`analysis.py` loads `metadata.csv` with `csv.DictReader`. Each image is:

1. Loaded with OpenCV.
2. Converted from BGR to RGB.
3. Resized to 160 x 160 pixels.
4. Converted to floating point.
5. Normalized from [0, 1] to the FaceNet range [-1, 1].

The images are already face crops, so the pipeline does not run a separate face detector.

## Face embedding model

The pipeline uses `InceptionResnetV1` from `facenet-pytorch`, initialized with pretrained VGGFace2 weights:

```python
InceptionResnetV1(pretrained="vggface2")
```

Each face is converted into a 512-dimensional identity embedding. Embeddings are L2-normalized before comparison.

The model weights are stored in:

```text
Tute 2/torch_cache/checkpoints/20180402-114759-vggface2.pt
```

FaceNet embeddings are used instead of raw pixels, global moments, color histograms, or SIFT descriptors because those methods do not separate identity reliably across large age changes.

## Similarity

Two normalized embeddings are compared with cosine similarity:

```text
similarity = embedding_1 dot embedding_2
```

Higher similarity indicates a stronger identity match.

## Genuine and impostor comparisons

For every subject, the two images in its metadata row are compared. This produces the genuine-score distribution.

For impostor trials, two distinct subject IDs are selected randomly. The first image from one subject is compared with the second image from the other. The default is 5,000 impostor trials.

## Threshold metrics

For threshold `t`:

- Accept when similarity >= `t`.
- Reject when similarity < `t`.

The metrics are:

```text
FMR  = accepted impostor pairs / total impostor pairs
FNMR = rejected genuine pairs / total genuine pairs
TAR  = accepted genuine pairs / total genuine pairs
```

Thresholds are generated from the observed score range because FaceNet scores for this dataset are not restricted to the old 0.55–0.90 range.

## ROC, AUC, and EER

The evaluation labels genuine pairs as 1 and impostor pairs as 0, then calculates:

- ROC curve: true-positive rate versus false-positive rate.
- AUC: overall ranking quality across thresholds.
- EER: the point where FMR and FNMR are approximately equal.

AUC near 0.50 is random separation; AUC near 1.00 is strong separation. Lower EER is better.

## Generated files

The analysis writes these files into `age_gap_dataset`:

```text
fmr_fnmr_vs_threshold.png
roc_curves.png
score_distributions_detailed.png
threshold_heatmap.png
performance_comparison.png
threshold_operating_points.png
comprehensive_evaluation_results.json
detailed_threshold_metrics.txt
```

The JSON file contains score statistics, AUC, EER, separability gap, and threshold operating points.

## Running the analysis

Use the `torch_env` Conda environment:

```powershell
conda activate torch_env
cd "D:\ABSS Tute\Tute 2"
python analysis.py
```

Or run the environment interpreter directly:

```powershell
C:\Users\HP\miniconda3\envs\torch_env\python.exe analysis.py
```

The script uses the project-local `torch_cache` directory for model weights.

## Validated result

With the VGGFace2 FaceNet model and the current dataset, the validated result was:

```text
AUC: 0.7473
EER: 32.34%
Genuine mean similarity: 0.2703
Impostor mean similarity: 0.1236
```

This indicates meaningful identity separation, although the large age gaps remain challenging. Better face alignment or an age-invariant ArcFace model could improve performance further.

