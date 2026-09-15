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

## Methods attempted and what we learned

Several comparison methods were tested before selecting FaceNet. The goal was
to separate genuine pairs (the same subject at two ages) from impostor pairs
(different subjects).

### 1. Raw-pixel comparison

The resized image pixels were compared directly. This failed because small
changes in alignment, lighting, pose, facial expression, and especially aging
produce large pixel differences even for the same person. Raw pixels also do
not contain an identity-invariant representation.

### 2. Global image moments

Global statistics such as image moments were extracted and normalized. These
describe the overall brightness and mass distribution of the image, but not
distinctive identity details. Faces from different people therefore produced
very similar values, causing genuine and impostor scores to overlap.

### 3. Color histograms

Per-channel color histograms were tested. Histograms ignore the spatial
location of facial features, so two different faces with similar skin tone,
background, or lighting can look alike. They also become unreliable when the
child and adult images have different lighting or image quality.

### 4. Simple thresholding

Thresholds were applied to similarity scores to classify pairs as accepted or
rejected. Thresholding is only a decision rule; it cannot fix weak features.
With the earlier descriptors, genuine and impostor score distributions were
almost identical, so every threshold either accepted most pairs or rejected
most pairs. The initial results were close to random performance (AUC about
0.52 and EER about 50%).

### 5. SIFT keypoint matching

SIFT local descriptors were used to find matching keypoints between the two
images. This can work for stable local textures, but it was not reliable here:
large age changes alter local facial texture, the images have limited
resolution, and some faces contain too few stable keypoints. Different faces
could also share incidental keypoints. SIFT produced heavy score overlap and
remained close to random performance (AUC about 0.54).

## Final improvement: FaceNet identity embeddings

The final pipeline uses the pretrained `InceptionResnetV1` FaceNet model with
VGGFace2 weights. Instead of comparing pixels or hand-designed statistics,
FaceNet maps each face to a 512-dimensional embedding learned for face
identity. The embedding is designed to retain identity information while being
less sensitive to pose, illumination, and appearance changes.

The two normalized embeddings are compared using cosine similarity. Genuine
and impostor scores are then evaluated with ROC, AUC, EER, FMR, FNMR, and TAR.

This improved the validated result to:

```text
Earlier handcrafted/SIFT methods: AUC approximately 0.52–0.55
FaceNet with VGGFace2 weights:      AUC 0.7473
FaceNet EER:                        32.34%
```

The result is a meaningful improvement, although very large age gaps remain
difficult. An age-invariant ArcFace model and stronger face alignment could
improve it further.

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
