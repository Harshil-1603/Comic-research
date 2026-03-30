"""
Central configuration — Comic Emotion Classification.
Updated for 7-class schema (j-hartmann) + fine-tuning setup.
"""

# ── Reproducibility ─────────────────────────────────────────
SEED = 42

# ── Data paths ───────────────────────────────────────────────
ANNOTATIONS_CSV = "data/annotations.csv"
SPLIT_CSV       = "data/annotations_split.csv"
PROCESSED_DIR   = "data/processed"
RAW_DIR         = "data/raw"
REVIEW_DIR      = "data/review"
SOURCE_DIR      = "data/source"

# ── Data splits ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Training ─────────────────────────────────────────────────
BATCH_SIZE    = 8       # reduced for fine-tuning within 6GB VRAM
LEARNING_RATE = 1e-4   # fusion head LR
ENCODER_LR    = 1e-5   # encoder LR (10x smaller to preserve pre-trained weights)
EPOCHS        = 10
NUM_WORKERS   = 0       # 0 avoids tokenizer fork warning with fine-tuning
USE_AMP       = True    # automatic mixed precision (saves ~40% VRAM)

# ── Model architecture ───────────────────────────────────────
D_IMG  = 512   # CLIP  ViT-B/32 output dim
D_TXT  = 768   # BERT  base output dim
D_COL  = 48    # HSV   histogram dim  (3 channels × 16 bins)
D_ATTN = 512   # AttnFusion internal attention dim
N_CLS  = 7     # 7-class schema (j-hartmann)

# ── Encoders ─────────────────────────────────────────────────
CLIP_MODEL = "openai/clip-vit-base-patch32"
BERT_MODEL = "bert-base-uncased"

# ── Color features ───────────────────────────────────────────
COLOR_BINS           = 16
USE_BACKGROUND_AWARE = False

# ── Output paths ─────────────────────────────────────────────
CHECKPOINTS_DIR = "checkpoints"
LOGS_DIR        = "logs"
EXPERIMENTS_DIR = "experiments"

# ── 7-class emotion schema (matches j-hartmann model output) ─
LABEL_MAP = {
    "anger":    0,
    "disgust":  1,
    "fear":     2,
    "joy":      3,
    "neutral":  4,
    "sadness":  5,
    "surprise": 6,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
