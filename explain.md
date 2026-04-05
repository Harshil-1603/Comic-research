# Comic Emotion Multimodal MLOps Pipeline — Architecture & File Explanations

This document provides a highly detailed explanation of the current system architecture, the purpose of every file, and how data moves across the overall machine learning lifecycle.

## Overview

The `Comics` project is a fully automated, end-to-end multimodal machine learning pipeline. It aims to extract individual comic panels from raw PDF files, automatically detect and transcribe the dialogue/speech text from those panels (via OCR), assess the contextual emotion of those interactions via state-of-the-art Natural Language Models, and use **both** Image & Text features to train an advanced `Cross-Attention Fusion Network`. 

Finally, the network is evaluated to automatically classify the primary underlying emotion on a 7-class schema: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, and `surprise`.

---

## 1. Master Orchestration
**`run_pipeline.py`**
This script acts as the master conductor. It runs exactly 8 sequential steps to take the project from zero to a trained, evaluated, and production-ready system. 
1. `convert`: Extranets pages from a comic book PDF.
2. `extract`: Detects panel boundaries and crops raw pages into distinct comic panels.
3. `auto_annotate`: Harvests text & labels from the images autonomously.
4. `split`: Distributes the processed images into a training, validation, and test set.
5. `train`: Instantiates the models and trains the weights using PyTorch.
6. `eval`: Evaluates the model accuracy, dumping a confusion matrix and classification reports.
7. `ablation`: Strips parts of the models away intentionally and re-evaluates them to prove why all features (Multimodal) combined are better together.
8. `infer`: Demos the model on a test image.

**`config.py`**
Centralized hyperparameter and settings registry. Every Python file imports `config` directly to identify hyperparameters like `LEARNING_RATE`, `BATCH_SIZE`, epochs, data splits, output dimensions (e.g. `D_IMG = 512`), and local directory paths to prevent fragile hardcoded strings throughout the project.

---

## 2. Model Architecture (`models/`)
The secret piece to the puzzle is the **Multimodal Fusion Engine**. The neural network doesn't simply view the image or read the text; it fuses them mathematically to infer emotions from context.

- **`models/image_encoder.py` (CLIP ViT backbone)**
  Fine-tunes OpenAI’s `CLIP` model (Vision Transformer base). It takes the raw PIL comic panel, pushes it through the patched transformer layers, and returns a 512-dimensional continuous feature vector embodying the visual representation of the scene.
  
- **`models/text_encoder.py` (BERT backbone)**
  Fine-tunes the `bert-base-uncased` language model. It takes the text tokens recognized by OCR and returns a 768-dimensional language map representing the semantic context of what the characters are saying.

- **`models/fusion_model.py` (AttnFusion)**
  The ultimate classifier. It absorbs the 512-dim visual features and the 768-dim language tensor, along with a 48-dim color histogram representation. It aligns these arrays and passes them through a sophisticated `MultiheadAttention` (Cross-Attention) matrix—forcing the visual representations to conceptually *look* at the text embeddings—before compressing the result down into the 7 emotional probabilities.

---

## 3. Data Flow & Labeling (`data/` and `features/`)

Since there were hundreds, possibly thousands, of comic frames that lacked actual target sentiment labels, we engineered a pseudo-automatic-annotation pipeline:

- **`features/ocr.py`**:
  Takes an image tensor, renders it grayscale, applies Gaussian Blur & Otsu threshold binarizations, and reads words out of the balloons using PyTesseract.
- **`features/sentiment.py`**:
  Feeds the found text into a localized hugging-face server running `j-hartmann/emotion-english-distilroberta-base`. Without human intervention, the Roberta engine accurately estimates what the emotion is (Fear, Anger, Joy, etc.) based purely on speech syntax.
- **`features/color_features.py`**:
  Calculates the raw Hue, Saturation, and Value (HSV) spectrum of each image into 16 bins to understand how visual tonality (e.g. highly saturated red for anger, dark tones for sadness) feeds into emotion.

These submodules are directly driven by:
- **`data/scripts/auto_annotate.py`**: Merges OCR and Sentiment to iterate over every extracted graphical panel image (`data/processed/`), effectively producing the localized `data/annotations.csv` file without any manual labeling necessary.
- **`data/dataset.py`**: PyTorch standard wrapper. Handles dynamic data-loading—converting images, text arrays, and emotional strings into structured Cuda Tensors dynamically across batches exactly when the model needs it.

---

## 4. Training Engine (`train.py` & Utilities)
Our neural network training framework features extensive dynamic allocations for aggressive optimization. Look closely at `train.py`:
- It initiates the `AttnFusion` model alongside the `CLIP` and `BERT` parameter hierarchies.
- **Memory Savers**: With `BATCH_SIZE = 2` configured, standard gradients use `accumulation_steps = 4` to virtually mimic a batch size of 8, allowing high-performance learning to fit easily beneath the 6GB VRAM bounds of the RTX 4050. The Encoder parameters are also frozen (`requires_grad = False`) early on so that compute power focuses solely on the `fusion` head optimization.
- **`utils/split_data.py`** executes *Stratified Splitting* guaranteeing that 15% testing splits exist precisely in proportion to initial training categories so underrepresented datasets (like `disgust`) are evenly scattered.
- **`utils/class_weights.py`** enforces mathematically inverse cost weights against `CrossEntropyLoss` during epochs, ensuring that overwhelming majorities (like the `neutral` class) aren't allowed to drown out the loss penalty over minority classifications.

---

## 5. Experimentation & Inference (The Proof)
- **`eval.py`**:
  Loads up the `data/annotations_split.csv` looking explicitly for panels tagged as `test`, runs them through the `.pt` serialized checkpoints spawned in the `checkpoints` directory, and plots a rigorous seaborn `confusion_matrix.png` directly inside `logs/`.
- **`experiments/run_ablations.py`**:
  Rigorously isolates modalities. It effectively loops completely distinct model classes—one loop omitting color features (`ColorlessFusion`), one utilizing language only (`TextOnlyModel`), one utilizing visuals only (`ImageOnlyModel`)—to statistically prove how the multi-model architecture dominates single-mode architectures.
- **`inference.py`**:
  Demonstrator script. Once the system finishes training entirely, you can point this python file at an absolute URL of any unseen panel graphic, and it will tokenize the image context, pull out the words automatically, render embeddings, and return a robust JSON with the inferred emotion probabilities across the 7 criteria.
