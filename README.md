# 🎙️ Accent & Language Identification Competition

> **Classify spoken language and accent nativeness from 10-second audio clips.**  
> Built on Mozilla Common Voice · 7 languages · 3 audio augmentation types
>
> Click on 'Leaderboard' below to submit your predictions and check your rank on the live leaderboard.
>
> 
> [![Leaderboard](https://img.shields.io/badge/🏆_Leaderboard-Streamlit-red?style=for-the-badge)](https://accent-language-identification-competition-aims2026.streamlit.app/) 

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Files](#files)
4. [Task Definition](#task-definition)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Submission Format](#submission-format)
7. [Scoring Formula](#scoring-formula)
8. [Leaderboard Rules](#leaderboard-rules)
9. [Getting Started](#getting-started)
10. [Augmentation Pipeline](#augmentation-pipeline)
11. [Baseline Models](#baseline-models)
12. [Tips & Best Practices](#tips--best-practices)
13. [FAQ](#faq)
14. [License](#license)

---

## Overview

This competition challenges participants to build a model that can:

1. **Identify the spoken language** from a 10-second audio clip (8 languages)
2. **Detect whether the speaker is a native or non-native** speaker of that language

The dataset is derived from **Mozilla Common Voice** and has been augmented ×4 using three acoustic transformation pipelines simulating real-world degraded conditions (ambient noise, pitch/tempo shifts, codec compression).

| | Details |
|---|---|
| **Task type** | Multi-output classification |
| **Input** | Audio features (pre-extracted) |
| **Languages** | French,Swahili, Hausa, Arabic, Wolof, Portuguese, German |
| **Total clips** | Original × 4 after augmentation |
| **Split** | 80% train / 20% test |
| **Primary metric** | Weighted F1-score (language) |
| **Secondary metric** | Accuracy (native/non-native) |

---

## Dataset Description

### Source

The audio clips originate from [Mozilla Common Voice](https://commonvoice.mozilla.org), a large-scale multilingual speech dataset collected from volunteer contributors worldwide.

### Languages

| Code | Language | Native Accents | Foreign Accents |
|------|----------|----------------|-----------------|
| `fr` | French | paris, lyon, marseille, québec | maghrebin, africain, anglophone |
| `sw` | Swahili | tanzanian, kenyan, ugandan | congolais, rwandais |
| `ha` | Hausa | kano, sokoto, zaria | nigerien, camerounais |
| `ar` | Arabic | egyptien, marocain, golfe, levantin | franco_arabe, anglophone |
| `wo` | Wolof | dakar, thies, saint_louis, ziguinchor | gambian, mauritanien |
| `pt` | Portuguese | lisbon, porto, bresilien | angolais, mozambicain |
| `de` | German | berlin, bavarian, swiss, austrian | turkish, eastern_european |

### Audio Specifications

| Property | Value |
|----------|-------|
| Sample rate | 16,000 Hz |
| Duration | 10 seconds (padded or truncated) |
| Format | WAV PCM 16-bit (or NPY array) |
| Channels | Mono |

### Pre-extracted Features

Each row in the CSV files contains pre-extracted audio features so participants **do not need to process raw audio** to get started:

| Feature | Description |
|---------|-------------|
| `rms_energy` | Root mean square energy (loudness) |
| `zero_crossing_rate` | Rate of sign changes in signal |
| `spectral_centroid` | Weighted mean of frequencies (Hz) |
| `spectral_bandwidth` | Spread around spectral centroid (Hz) |
| `spectral_rolloff` | Frequency below which 85% of energy lies |
| `silence_ratio` | Proportion of silent frames |
| `energy_band_1..5` | Energy in 5 frequency sub-bands |
| `mfcc_1..13_mean` | Mean of 13 MFCC coefficients *(if librosa available)* |
| `mfcc_1..13_std` | Std of 13 MFCC coefficients *(if librosa available)* |
| `delta_mfcc_1..13_mean` | First-order delta MFCCs *(if librosa available)* |
| `chroma_mean / std` | Chroma feature summary *(if librosa available)* |
| `mel_mean / std` | Mel-spectrogram summary *(if librosa available)* |
| `f0_mean / std` | Fundamental frequency stats *(if librosa available)* |
| `voiced_fraction` | Fraction of voiced frames *(if librosa available)* |
| `tempo` | Estimated speech tempo (BPM) *(if librosa available)* |

> **Note:** Features marked *if librosa available* are only present when the dataset was generated with `librosa` installed. The basic 11 features are always available.

---

## Files

```
Accent & language identification competition/
│
├── 📄 train_dataset.csv         Training set — 80% of data — WITH labels
├── 📄 test_dataset.csv          Test set — 20% of data — WITHOUT labels
├── 📄 submission_example.csv    Example submission file (random predictions)
├── 📄 dataset_info.json         Competition metadata and statistics
├── 📄 README.md                 This file
├── 📄 requirements              Python dependencies
└── 📄 requirements_streamlit.txt          Streamlit dependencies
```

### File Details

#### `train_dataset.csv`
Training data with all features and labels. Use this to train and validate your model.

**Key columns:**

| Column | Type | Description |
|--------|------|-------------|
| `clip_id` | string | Unique clip identifier (e.g., `fr_000042_aug2`) |
| `language` | string | ISO language code — **TARGET 1** |
| `is_native` | int (0/1) | Native speaker flag — **TARGET 2** |
| `accent_region` | string | Accent region label — **TARGET 3** (bonus) |
| `augmented` | bool | Whether this clip is augmented |
| `aug_type` | string | `original`, `noise_ambient`, `pitch_time_shift`, `degraded_codec` |
| `age` | string | Speaker age group |
| `gender` | string | Speaker gender |
| `audio_path` | string | Path to WAV file on disk |
| `rms_energy` | float | Audio feature |
| `...` | float | (all other features) |

#### `test_dataset.csv`
Test data **without** the label columns (`language`, `is_native`, `accent_region`). Your model must predict these.

#### `submission_example.csv`
Shows the exact format your submission file must follow. The predictions in this file are **random** — they serve only as a format reference.

#### `secret_test_labels.csv`
Contains the true labels for the test set. This file is kept by the competition organizer and used to score submissions. **Do not distribute.**

---

## Task Definition

### Task 1 — Language Identification (Primary)

Given the pre-extracted features of a 10-second audio clip, predict the spoken **language** (one of 7 ISO codes).

```
Input:  feature vector (11 to 50+ columns depending on librosa availability)
Output: language code string (fr, sw, ha, ar, wo, pt, de)
```

### Task 2 — Native Speaker Detection (Secondary)

Simultaneously predict whether the speaker is a **native** speaker of the identified language.

```
Input:  same feature vector
Output: is_native = 0 (non-native) or 1 (native)
```

### Task 3 — Accent Region (Bonus)

Predict the specific **accent region** (e.g., `dakar`, `american`, `kano`). This task is optional and earns bonus points on the leaderboard.

---

## Evaluation Metrics

### Primary Metric — Weighted F1-Score (Language)

The **weighted F1-score** accounts for class imbalance by weighting each class's F1 by its support (number of true instances).

```
F1_language = Σ (support_c / total) × F1_c    for each class c
```

Where for each class:
```
F1_c = 2 × (Precision_c × Recall_c) / (Precision_c + Recall_c)
```

**Range:** 0.0 (worst) → 1.0 (perfect)

### Secondary Metric — Accuracy (Native/Non-Native)

Standard binary accuracy:

```
Acc_native = (True Positives + True Negatives) / Total Samples
```

**Range:** 0.0 (worst) → 1.0 (perfect)

### Bonus Metric — Macro F1-Score (Accent Region)

```
F1_accent = (1/N_accents) × Σ F1_c    for each accent class c
```

---

## Submission Format

Your submission must be a **CSV file named `submission.csv`** with exactly the following columns:

```csv
clip_id,language,is_native,accent_region,confidence_language,confidence_native
fr_000042,fr,1,paris,0.9231,0.8712
en_000015,en,0,french,0.8854,0.6103
sw_000088_aug1,sw,1,tanzanian,0.7645,0.9012
...
```

### Column Specifications

| Column | Required | Type | Values | Description |
|--------|----------|------|--------|-------------|
| `clip_id` | ✅ Yes | string | From `test_dataset.csv` | Must match test set IDs exactly |
| `language` | ✅ Yes | string | `fr`, `sw`, `ha`, `ar`, `wo`, `pt`, `de` | Predicted language |
| `is_native` | ✅ Yes | int | `0` or `1` | Native speaker prediction |
| `accent_region` | ✅ Yes | string | Any accent string | Predicted accent (use `unknown` if unsure) |
| `confidence_language` | ✅ Yes | float | `[0.0, 1.0]` | Model confidence for language prediction |
| `confidence_native` | ✅ Yes | float | `[0.0, 1.0]` | Model confidence for native prediction |

### Submission Rules

- All `clip_id` values from `test_dataset.csv` must be present — **no missing rows**
- `language` must be one of the 8 supported ISO codes
- `is_native` must be exactly `0` or `1` (not a float)
- Confidence scores must be in `[0.0, 1.0]`
- File encoding must be **UTF-8**
- Maximum file size: **50 MB**

### Validation Before Submitting

```python
import pandas as pd

sub    = pd.read_csv("submission.csv")
test   = pd.read_csv("test_dataset.csv")

LANGS  = {'fr','sw','ha','ar','wo','pt','de'}
errors = []

# Check all clip IDs present
missing = set(test['clip_id']) - set(sub['clip_id'])
if missing:
    errors.append(f"Missing {len(missing)} clip_ids")

# Check language values
invalid_lang = set(sub['language']) - LANGS
if invalid_lang:
    errors.append(f"Invalid language codes: {invalid_lang}")

# Check is_native values
if not sub['is_native'].isin([0,1]).all():
    errors.append("is_native must be 0 or 1")

# Check confidence ranges
for col in ['confidence_language','confidence_native']:
    if not ((sub[col] >= 0) & (sub[col] <= 1)).all():
        errors.append(f"{col} must be in [0, 1]")

if errors:
    print("❌ Submission errors:")
    for e in errors: print(f"   - {e}")
else:
    print("✅ Submission is valid — ready to submit!")
```

---

## Scoring Formula

The **final competition score** combines both metrics:

```
Final Score = 0.6 × F1_language + 0.4 × Accuracy_native
```

**Example:**

| Model | F1 Language | Acc Native | Final Score |
|-------|-------------|------------|-------------|
| Random baseline | 0.125 | 0.500 | 0.275 |
| Language-only | 0.850 | 0.500 | 0.710 |
| Full model | 0.920 | 0.810 | 0.876 |
| **Perfect** | **1.000** | **1.000** | **1.000** |

### Computing Your Score Locally

```python
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

sub    = pd.read_csv("submission.csv")
secret = pd.read_csv("secret_test_labels.csv")  # organizer only
merged = sub.merge(secret, on='clip_id', suffixes=('_pred','_true'))

f1  = f1_score(merged['language_true'], merged['language_pred'],
               average='weighted', zero_division=0)
acc = accuracy_score(merged['is_native_true'], merged['is_native_pred'])

print(f"F1 Language  : {f1:.4f}")
print(f"Acc Native   : {acc:.4f}")
print(f"Final Score  : {0.6*f1 + 0.4*acc:.4f}")
```

Or use the built-in evaluator:

```bash
python accent_dataset_builder_v2.py \
    --evaluate submission.csv \
    --secret   secret_test_labels.csv
```

---

## Leaderboard Rules

| Rule | Details |
|------|---------|
| **Max submissions/day** | 3 |
| **Public leaderboard** | Scored on 50% of test set (randomly sampled) |
| **Private leaderboard** | Scored on 100% of test set (revealed at end) |
| **Final ranking** | Based on private leaderboard only |
| **Team size** | Max 4 participants |
| **External data** | Allowed (must be declared in model card) |
| **Pre-trained models** | Allowed (must be declared) |
| **Code submission** | Required for top-3 winners |

### Anti-Cheating

Each `clip_id` in `secret_test_labels.csv` contains a `checksum` field (MD5 hash). Any submission that scores suspiciously close to 1.0 will trigger a code review.

---

## Getting Started

### Step 1 — Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv env_accent
source env_accent/bin/activate        # Mac/Linux
env_accent\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2 — Generate real data (optional, recommended)

```bash
# Download Common Voice + augment ×4 automatically
python accent_dataset_builder_v2.py --language fr --n_samples 500

# Multiple languages
python accent_dataset_builder_v2.py --language fr,wo,sw,ha --n_samples 300

# Quick demo without downloading
python accent_dataset_builder_v2.py --demo --n_samples 200
```

### Step 3 — Train a baseline model

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
train = pd.read_csv("train_dataset.csv")
test  = pd.read_csv("test_dataset.csv")

# Feature columns (exclude metadata and labels)
EXCLUDE = ['clip_id','language','is_native','accent_region',
           'age','gender','sentence','audio_path','augmented','aug_type',
           'duration_sec','sample_rate']
feat_cols = [c for c in train.columns if c not in EXCLUDE]

X_train = train[feat_cols].fillna(0)
y_lang  = train['language']
y_nat   = train['is_native']
X_test  = test[feat_cols].fillna(0)

# Train
clf_lang = RandomForestClassifier(n_estimators=200, random_state=42)
clf_lang.fit(X_train, y_lang)

clf_nat  = RandomForestClassifier(n_estimators=200, random_state=42)
clf_nat.fit(X_train, y_nat)

# Predict
pred_lang = clf_lang.predict(X_test)
pred_nat  = clf_nat.predict(X_test)
prob_lang = clf_lang.predict_proba(X_test).max(axis=1)
prob_nat  = clf_nat.predict_proba(X_test).max(axis=1)

# Build submission
import numpy as np
submission = pd.DataFrame({
    'clip_id':             test['clip_id'],
    'language':            pred_lang,
    'is_native':           pred_nat,
    'accent_region':       'unknown',
    'confidence_language': np.round(prob_lang, 4),
    'confidence_native':   np.round(prob_nat, 4),
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv generated")
```

### Step 4 — Evaluate locally (if you have the secret labels)

```bash
python accent_dataset_builder_v2.py \
    --evaluate submission.csv \
    --secret   secret_test_labels.csv
```

---

## Augmentation Pipeline

Each original clip was augmented 3 times, producing ×4 the data volume.

### aug_type: `noise_ambient`
Simulates recording in a noisy environment (café, street, office).

- Gaussian noise: amplitude 0.003–0.025
- Random gain: ±4 dB
- Telephone bandpass filter (300–3400 Hz): applied with 30% probability
- High-pass filter (80–300 Hz): applied with 40% probability

### aug_type: `pitch_time_shift`
Simulates different speaking rates and voice pitches (useful for cross-speaker generalization).

- Time stretch: rate 0.85×–1.15× (slower or faster speech)
- Pitch shift: ±3 semitones
- Random gain: ±6 dB
- Temporal shift: ±0.3 seconds

### aug_type: `degraded_codec`
Simulates phone calls, compressed recordings, or poor-quality microphones.

- MP3 compression: 16–64 kbps
- Low-pass filter: cutoff 2000–4000 Hz
- Clipping distortion: 0–5th percentile
- Gaussian noise: amplitude 0.01–0.05

### Augmentation Distribution

| aug_type | % of dataset |
|----------|-------------|
| original | 25% |
| noise_ambient | 25% |
| pitch_time_shift | 25% |
| degraded_codec | 25% |

> **Tip:** Consider training on all 4 types jointly for better robustness. You can also use `aug_type` as an auxiliary feature or train separate models per augmentation type.

---

## Baseline Models

| Model | F1 Language | Acc Native | Final Score | Notes |
|-------|-------------|------------|-------------|-------|
| Random | ~0.125 | ~0.500 | ~0.275 | Theoretical minimum |
| Majority class | ~0.140 | ~0.650 | ~0.344 | Always predict most common language |
| TF-IDF + LR (text) | — | — | — | Not applicable (audio features) |
| Random Forest (11 features) | ~0.650 | ~0.720 | ~0.678 | Basic features only |
| Random Forest (full features) | ~0.820 | ~0.780 | ~0.804 | With librosa MFCCs |
| XGBoost (full features) | ~0.860 | ~0.810 | ~0.840 | Gradient boosting |
| wav2vec 2.0 fine-tuned | ~0.930 | ~0.870 | ~0.906 | End-to-end on raw audio |

---

## Tips & Best Practices

### Feature Engineering

```python
# Interaction features between MFCCs and energy
train['mfcc1_x_energy'] = train['mfcc_1_mean'] * train['rms_energy']

# Ratio of voiced to total duration
train['speech_ratio'] = 1 - train['silence_ratio']

# Log-transform skewed features
import numpy as np
train['log_rms'] = np.log1p(train['rms_energy'])
```

### Handling Augmented Data

```python
# Option A — Use all data (recommended)
X_train = train[feat_cols]

# Option B — Train only on originals, evaluate robustness on augmented
X_train = train[train['aug_type'] == 'original'][feat_cols]

# Option C — Use aug_type as a feature
train['aug_type_encoded'] = train['aug_type'].map({
    'original': 0, 'noise_ambient': 1,
    'pitch_time_shift': 2, 'degraded_codec': 3
})
```

### Cross-Validation Strategy

Because augmented clips are derived from originals, a naive random split leaks information. Use **group-based cross-validation** to avoid this:

```python
from sklearn.model_selection import GroupKFold

# Extract original clip ID (strip augmentation suffix)
train['original_id'] = train['clip_id'].str.replace(r'_aug\d+$', '', regex=True)

gkf = GroupKFold(n_splits=5)
for fold, (tr_idx, val_idx) in enumerate(
    gkf.split(X_train, y_train, groups=train['original_id'])
):
    # Train/validate without augmented leakage
    ...
```

### Advanced Approaches

- **wav2vec 2.0 / Whisper** — End-to-end learning directly on raw audio (requires raw WAV files)
- **Spectrogram + CNN** — Convert audio to mel-spectrogram image, use image classifier
- **Multi-task learning** — Joint language + native prediction with shared encoder
- **Ensemble** — Combine Random Forest + XGBoost + SVM predictions

---

## FAQ

**Q: Do I need to process raw audio files?**  
A: No. Pre-extracted features in the CSV files are sufficient to build a competitive model. Raw WAV files are generated by the builder script for participants who want to extract their own features.

**Q: Can I use the augmented clips during training?**  
A: Yes, and it is strongly recommended. Augmented clips improve model robustness. See the cross-validation tip above to avoid leakage.

**Q: What if `librosa` is not installed?**  
A: The dataset will contain only 11 basic features. Installing `librosa` adds 30+ richer features (MFCCs, F0, chroma, etc.) that significantly boost performance.

**Q: Can I use external datasets?**  
A: Yes, external datasets (e.g., VoxLingua107, FLEURS) are allowed but must be declared in your model card submitted with the code.

**Q: My submission file has the right format but the scorer crashes.**  
A: Check that `clip_id` values match exactly (case-sensitive, including `_aug1` suffixes). Run the validation script in the [Submission Format](#submission-format) section.

**Q: Why is Wolof included?**  
A: Wolof is a major language of Senegal and The Gambia, spoken by ~10 million people. This competition deliberately includes low-resource African languages to encourage development of inclusive AI systems.

---

## License

- **Audio data:** Mozilla Common Voice License (CC0 for audio, CC-BY for text)
- **Augmentation code:** MIT License
- **Competition framework:** Free to use and modify

---

## Citation

If you use this dataset or competition framework in academic work, please cite:

```bibtex
@dataset{accent_lang_id_2026,
  title     = {Accent \& Language Identification Challenge Dataset},
  year      = {2026},
  note      = {Built on Mozilla Common Voice v13, augmented with audiomentations},
  url       = {https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0}
}
```

---

*Competition framework built with `accent_dataset_builder_v2.py` · Mozilla Common Voice · HuggingFace Datasets By Christian Munguaganze & Fatou Bintou*
