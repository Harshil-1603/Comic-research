# Comic Emotion Classification

## 0. Problem Formalization

### 0.1 Task Definition

**Input:** Comic panel image $I$, optional dialogue text $T$

**Output:** Emotion label $y \in \{anger, sadness, joy, fear, neutral\}$

### 0.2 Hypothesis

Adding explicit color features improves multimodal emotion classification over image+text baselines.

### 0.3 Metrics

- Accuracy
- Precision (macro)
- Recall (macro)
- F1 (macro)
- Confusion matrix
- Ablation deltas

---

## 20. Final Report

### 20.1 Method

This project implements a multimodal emotion classification system for comic panels using three modalities:

1. **Image Encoder:** CLIP (openai/clip-vit-base-patch32) - extracts 512-dim visual features
2. **Text Encoder:** BERT (bert-base-uncased) - extracts 768-dim text features
3. **Color Features:** HSV histogram (16 bins per channel) - extracts 48-dim color features

Two fusion architectures were implemented:
- **MLP Fusion:** Concatenation + 2-layer MLP (256 hidden units)
- **Attention Fusion:** Cross-attention between image and text + color concatenation

### 20.2 Ablations

Four ablation configurations were tested to validate the hypothesis:

| Configuration | Use Text | Use Color | Notes |
|---------------|----------|-----------|-------|
| Image Only | No | No | Baseline |
| Image + Text | Yes | No | Text-only multimodal |
| Image + Color | No | Yes | Color-only multimodal |
| Image + Text + Color | Yes | Yes | Full model |

Run ablations with:
```bash
python experiments/run_ablations.py
```

Results are logged to `experiments/results.csv`.

### 20.3 Error Analysis

Common failure modes to investigate:
- **Ambiguous emotions:** Panels with mixed or subtle emotional cues
- **Text-image mismatch:** Dialogue contradicts visual emotion
- **Color bias:** Over-reliance on color for emotion (e.g., red=anger)
- **Character vs scene:** Model confusion between character emotion and scene mood

### 20.4 Project Structure

```
.
├── data/                   # Data processing
│   ├── pdf_to_images.py    # PDF to image conversion
│   ├── panel_extractor.py  # Panel segmentation
│   ├── dataset.py          # Torch Dataset
│   └── annotations.csv     # Annotation schema
├── models/                 # Model architectures
│   ├── image_encoder.py    # CLIP encoder
│   ├── text_encoder.py     # BERT encoder
│   └── fusion_model.py     # MLP + Attention fusion
├── features/               # Feature extraction
│   ├── color_features.py   # HSV histograms
│   └── ocr.py              # Text extraction
├── utils/                  # Utilities
│   ├── collate.py          # Batch collation
│   ├── review_ui.py        # Streamlit review UI
│   ├── annotator.py        # CLI annotation tool
│   ├── split_data.py       # Train/val/test splits
│   └── reproducibility.py  # Seed + config logging
├── experiments/            # Experiments
│   └── run_ablations.py    # Ablation study script
├── train.py                # Training pipeline
├── eval.py                 # Evaluation script
├── inference.py            # Inference script
└── README.md               # This file
```

### 20.5 Usage

**Training:**
```bash
python train.py
```

**Evaluation:**
```bash
python eval.py
```

**Inference:**
```bash
python inference.py path/to/panel.jpg --text "dialogue text"
```

**Ablation Study:**
```bash
python experiments/run_ablations.py
```

**Data Splitting:**
```bash
python utils/split_data.py
```

### 20.6 Dependencies

See `requirements.txt` for pinned dependencies including:
- torch==2.2.2
- transformers==4.40.0
- opencv-python==4.9.0.80
- scikit-learn==1.4.2
- pandas==2.2.2
