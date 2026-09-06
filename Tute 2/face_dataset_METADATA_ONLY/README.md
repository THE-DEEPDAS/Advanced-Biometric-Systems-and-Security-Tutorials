# Facial Recognition Dataset

## Overview
A synthetic facial recognition dataset containing **200 subjects** with **1200 images** total.

- **2 images per subject** (simulating enrollment + verification scenarios)
- **Minimum age gap:** 1+ years between images
- **3 variants:** REAL (clean), GAUSSIAN (noise), SALT-PEPPER (noise)

## Dataset Statistics

### Demographics
- **Total Subjects:** 200
- **Gender Distribution:**
  - Male: 87 (43.5%)
  - Female: 113 (56.5%)

- **Ethnicity Distribution:**
  - African: 53 (26.5%)
  - Asian: 54 (27.0%)
  - Caucasian: 64 (32.0%)
  - Hispanic: 29 (14.5%)

### Age Information
- **Age Range:** 18 - 80 years
- **Average Age:** 46.9 years
- **Age Gap Between Images:**
  - Minimum: 1 years
  - Maximum: 5 years
  - Average: 3.04 years

## Directory Structure

```
face_dataset/
├── REAL/                    # Clean, original images (400 images)
│   ├── S0001_001.jpg
│   ├── S0001_002.jpg
│   ├── S0002_001.jpg
│   ├── S0002_002.jpg
│   └── ...
├── GAUSSIAN/                # Images with Gaussian noise (400 images)
│   ├── S0001_001.jpg
│   ├── S0001_002.jpg
│   └── ...
├── SALT_PEPPER/             # Images with salt-pepper noise (400 images)
│   ├── S0001_001.jpg
│   ├── S0001_002.jpg
│   └── ...
├── metadata.json            # Complete metadata for all subjects
└── README.md               # This file
```

## Metadata Structure

Each record in `metadata.json` contains:

```json
{
  "subject_id": "S0001",
  "subject_number": 1,
  "gender": "M",
  "ethnicity": "Asian",
  "age_image1": 25,
  "age_image2": 28,
  "age_gap_years": 3,
  "capture_date_1": "2023-12-15T10:30:00",
  "capture_date_2": "2024-12-15T14:20:00",
  "image1_real": "REAL/S0001_001.jpg",
  "image2_real": "REAL/S0001_002.jpg",
  "image1_gaussian": "GAUSSIAN/S0001_001.jpg",
  "image2_gaussian": "GAUSSIAN/S0001_002.jpg",
  "image1_salt_pepper": "SALT_PEPPER/S0001_001.jpg",
  "image2_salt_pepper": "SALT_PEPPER/S0001_002.jpg"
}
```

## Usage in Pipeline

### MTCNN Face Detection
```python
import cv2
from mtcnn import MTCNN

detector = MTCNN()
img = cv2.imread('face_dataset/REAL/S0001_001.jpg')
faces = detector.detect_faces(img)
```

### Feature Extraction (FaceNet/PFE)
```python
from facenet_pytorch import InceptionResnetV1

model = InceptionResnetV1(pretrained='vggface2').eval()
embedding = model(face_image)
```

### Genuine vs Impostor Scoring
- **Genuine Pair:** Same subject, Image 1 vs Image 2
- **Impostor Pair:** Different subjects, Image 1 vs Image 1

Calculate similarity:
```python
similarity = np.dot(embedding1, embedding2)
```

## Performance Evaluation

Three evaluation sets:
1. **REAL Dataset:** Baseline performance on clean images
2. **GAUSSIAN Dataset:** Robustness to random Gaussian noise
3. **SALT-PEPPER Dataset:** Robustness to structured noise

Calculate metrics:
- **FMR (False Match Rate):** Incorrect genuine acceptance
- **FNMR (False Non-Match Rate):** Incorrect genuine rejection

## Image Specifications

- **Resolution:** 256×256 pixels
- **Format:** JPEG (.jpg)
- **Quality:** 95%
- **Color Space:** RGB

## Generation Notes

- Synthetic faces generated with demographic awareness
- Age variations simulated through visual features
- Noise added consistently across all variants
- Metadata timestamped for realistic aging scenarios

---

*Generated: 2026-09-04 05:21:07*
