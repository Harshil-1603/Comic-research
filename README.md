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
