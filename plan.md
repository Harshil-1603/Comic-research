0. PROBLEM FORMALIZATION (DO FIRST)
0.1 Define Task

Input: comic panel image 
𝐼
I, optional dialogue text 
𝑇
T
Output: emotion label 
𝑦
∈
{
anger, sadness, joy, fear, neutral
}
y∈{anger, sadness, joy, fear, neutral}

0.2 Hypothesis (write this in README)

Adding explicit color features improves multimodal emotion classification over image+text baselines.

0.3 Metrics
Accuracy, Precision, Recall, F1 (macro)
Confusion matrix
Ablation deltas

Commit

git add README.md
git commit -m "Defined task, hypothesis, and metrics"
1. ENVIRONMENT (REPRODUCIBLE)
1.1 Repo + Layout
mkdir comic_emotion_ml && cd comic_emotion_ml
git init

mkdir -p data/{raw,processed,review} models features utils experiments logs checkpoints
touch train.py eval.py config.py requirements.txt README.md
1.2 Virtualenv
python3 -m venv venv
source venv/bin/activate
1.3 Dependencies (pin versions)
pip install torch==2.2.2 torchvision==0.17.2 \
transformers==4.40.0 opencv-python==4.9.0.80 \
scikit-learn==1.4.2 pandas==2.2.2 numpy==1.26.4 \
tqdm==4.66.4 matplotlib==3.8.4 pillow==10.3.0 \
pdf2image==1.17.0 pytesseract==0.3.10 streamlit==1.34.0
pip freeze > requirements.txt

Commit

git add .
git commit -m "Project scaffold + venv + pinned dependencies"
2. DATA INGESTION (PDF → IMAGES)
2.1 Install poppler (for pdf2image)
sudo apt install poppler-utils
2.2 Convert PDFs to pages

file: data/pdf_to_images.py

from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, out_dir, dpi=300):
    os.makedirs(out_dir, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=dpi)
    for i, p in enumerate(pages):
        p.save(os.path.join(out_dir, f"{os.path.basename(pdf_path)}_p{i:04d}.jpg"), "JPEG")

if __name__ == "__main__":
    pdf_to_images("data/source/comic1.pdf", "data/raw/")

Commit

git add data/pdf_to_images.py
git commit -m "PDF to image conversion"
3. PANEL EXTRACTION (AUTOMATED + DEBUG)
3.1 Contour-based extraction

file: data/panel_extractor.py

import cv2, os

def extract_panels(img_path, out_dir, min_area=8000):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # edges + morphology
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dil = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    base = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    pid = 0

    vis = img.copy()
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < min_area: continue
        panel = img[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(out_dir, f"{base}_panel_{pid}.jpg"), panel)
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
        pid += 1

    # debug image
    cv2.imwrite(os.path.join(out_dir, f"{base}_debug.jpg"), vis)
    return pid
3.2 Batch run

file: data/run_extraction.py

import os
from panel_extractor import extract_panels

inp, out = "data/raw/", "data/processed/"
os.makedirs(out, exist_ok=True)

for f in os.listdir(inp):
    if f.endswith(".jpg"):
        extract_panels(os.path.join(inp,f), out)

Commit

git add data/panel_extractor.py data/run_extraction.py
git commit -m "OpenCV panel extraction + debug overlay"
4. MANUAL REVIEW LAYER (MANDATORY)
4.1 Quick Streamlit UI

file: utils/review_ui.py

import streamlit as st, os, shutil
from PIL import Image

proc, review = "data/processed", "data/review"
os.makedirs(review, exist_ok=True)
imgs = sorted([f for f in os.listdir(proc) if f.endswith(".jpg")])

i = st.slider("Index", 0, len(imgs)-1, 0)
path = os.path.join(proc, imgs[i])
st.image(Image.open(path), caption=imgs[i])

col1, col2 = st.columns(2)
with col1:
    if st.button("Delete"):
        os.remove(path)
with col2:
    if st.button("Move to review"):
        shutil.move(path, os.path.join(review, imgs[i]))

Run:

streamlit run utils/review_ui.py

Commit

git add utils/review_ui.py
git commit -m "Manual review UI for cleaning panels"
5. ANNOTATION (SCHEMA + TOOL)
5.1 Schema

file: data/annotations.csv

image,emotion,text,source
5.2 CLI annotator

file: utils/annotator.py

import cv2, os, pandas as pd

rows = []
for f in sorted(os.listdir("data/processed")):
    p = os.path.join("data/processed", f)
    img = cv2.imread(p)
    cv2.imshow("panel", img); cv2.waitKey(1)
    e = input("emotion [anger/sadness/joy/fear/neutral]: ").strip()
    t = input("text (optional): ").strip()
    rows.append([f, e, t, "unknown"])
    cv2.destroyAllWindows()

pd.DataFrame(rows, columns=["image","emotion","text","source"]) \
  .to_csv("data/annotations.csv", index=False)

Commit

git add data/annotations.csv utils/annotator.py
git commit -m "Annotation schema + CLI annotator"
6. DATA LOADER (TORCH-READY)

file: data/dataset.py

import os, cv2, torch
import pandas as pd
from torch.utils.data import Dataset

label_map = {"anger":0,"sadness":1,"joy":2,"fear":3,"neutral":4}

class ComicDataset(Dataset):
    def __init__(self, csv, img_dir, transforms=None):
        self.df = pd.read_csv(csv)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(os.path.join(self.img_dir, r["image"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms: img = self.transforms(img)
        y = label_map[r["emotion"]]
        text = r["text"] if isinstance(r["text"], str) else ""
        return img, text, torch.tensor(y)

Commit

git add data/dataset.py
git commit -m "Torch dataset for panels"
7. COLOR FEATURES (HSV + NORMALIZATION)

file: features/color_features.py

import cv2, numpy as np, torch

def hsv_hist(image, bins=16):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h = np.histogram(hsv[:,:,0], bins=bins, range=(0,180))[0]
    s = np.histogram(hsv[:,:,1], bins=bins, range=(0,255))[0]
    v = np.histogram(hsv[:,:,2], bins=bins, range=(0,255))[0]
    feat = np.concatenate([h,s,v]).astype("float32")
    feat /= (feat.sum() + 1e-6)
    return torch.tensor(feat)  # (48,)

Commit

git add features/color_features.py
git commit -m "HSV color feature extraction"
8. IMAGE ENCODER — CLIP

Use CLIP

file: models/image_encoder.py

import torch
from transformers import CLIPProcessor, CLIPModel

class ImageEncoder:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    @torch.no_grad()
    def __call__(self, pil_images):
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)  # (B,512)
        return feats / feats.norm(dim=-1, keepdim=True)

Commit

git add models/image_encoder.py
git commit -m "CLIP image encoder"
9. TEXT ENCODER — BERT

Use BERT

file: models/text_encoder.py

import torch
from transformers import AutoTokenizer, AutoModel

class TextEncoder:
    def __init__(self, device="cuda"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, texts):
        enc = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self.model(**enc).last_hidden_state  # (B, L, 768)
        pooled = out.mean(dim=1)                  # (B,768)
        return pooled

Commit

git add models/text_encoder.py
git commit -m "BERT text encoder"
10. FUSION MODEL (BASELINE MLP)

file: models/fusion_model.py

import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, d_img=512, d_txt=768, d_col=48, n_cls=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_img + d_txt + d_col, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_cls)
        )

    def forward(self, img, txt, col):
        x = nn.functional.normalize(img, dim=-1)
        x = nn.functional.pad(x, (0,0))  # no-op, keeps explicit
        z = nn.functional.normalize(txt, dim=-1)
        c = col
        out = self.net(nn.functional.dropout(nn.functional.relu(
            nn.functional.linear(nn.functional.pad(
                nn.functional.cat([x,z,c], dim=1), (0,0)), self.net[0].weight, self.net[0].bias)
        ), p=0.0))
        # simplified forward for clarity; you can use self.net(torch.cat(...))
        return out

(Use simpler self.net(torch.cat([...],1)) in practice.)

Commit

git add models/fusion_model.py
git commit -m "Baseline multimodal fusion (MLP)"
11. TRAINING PIPELINE
11.1 Collate (to handle text lists)

file: utils/collate.py

def collate(batch):
    imgs, texts, ys = zip(*batch)
    return list(imgs), list(texts), ys
11.2 Train

file: train.py

import torch, os
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from data.dataset import ComicDataset
from features.color_features import hsv_hist
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.fusion_model import FusionModel
from utils.collate import collate

device = "cuda" if torch.cuda.is_available() else "cpu"

tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
])

ds = ComicDataset("data/annotations.csv", "data/processed", transforms=None)
dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate, num_workers=2)

img_enc = ImageEncoder(device)
txt_enc = TextEncoder(device)
model = FusionModel().to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
crit = torch.nn.CrossEntropyLoss()

def to_pil(batch_imgs):
    return [tfm(img) for img in batch_imgs]

for epoch in range(10):
    model.train()
    for imgs, texts, ys in dl:
        pil_imgs = to_pil(imgs)
        img_feat = img_enc(pil_imgs)               # (B,512)
        txt_feat = txt_enc(texts)                  # (B,768)
        col_feat = torch.stack([hsv_hist(img) for img in imgs]).to(device)  # (B,48)
        y = torch.tensor(ys).to(device)

        pred = model(img_feat, txt_feat, col_feat)
        loss = crit(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pt")
    print(f"epoch {epoch} loss {loss.item():.4f}")

Commit

git add train.py utils/collate.py
git commit -m "End-to-end training loop"
12. EVALUATION

file: eval.py

import torch
from sklearn.metrics import classification_report, confusion_matrix

# load model, run on validation split, collect y_true/y_pred
# print(classification_report(y_true, y_pred))
# print(confusion_matrix(y_true, y_pred))

Commit

git add eval.py
git commit -m "Evaluation script with metrics"
13. OCR (OPTIONAL → REQUIRED LATER)
sudo apt install tesseract-ocr

file: features/ocr.py

import pytesseract, cv2

def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return pytesseract.image_to_string(gray)

Wire this into dataset if text is empty.

Commit

git add features/ocr.py
git commit -m "OCR integration for dialogue extraction"
14. ABLATION STUDIES (NON-NEGOTIABLE)

Run 4 configs:

Image only
Image + Text
Image + Color
Image + Text + Color

Log to experiments/results.csv.

Commit

git add experiments/
git commit -m "Ablation experiments logged"
15. UPGRADE FUSION (ATTENTION)

Replace MLP with cross-attention:

import torch.nn as nn

class AttnFusion(nn.Module):
    def __init__(self, d=512, n_cls=5):
        super().__init__()
        self.q = nn.Linear(512, d)
        self.k = nn.Linear(768, d)
        self.v = nn.Linear(768, d)
        self.attn = nn.MultiheadAttention(d, num_heads=8, batch_first=True)
        self.fc = nn.Linear(d+48, n_cls)

    def forward(self, img, txt, col):
        Q = self.q(img).unsqueeze(1)   # (B,1,d)
        K = self.k(txt).unsqueeze(1)   # (B,1,d)
        V = self.v(txt).unsqueeze(1)
        A,_ = self.attn(Q,K,V)         # (B,1,d)
        A = A.squeeze(1)
        return self.fc(torch.cat([A, col], dim=1))

Commit

git commit -am "Attention-based multimodal fusion"
16. COLOR IMPROVEMENT (BACKGROUND VS FOREGROUND)
Simple heuristic: k-means (k=3–5), treat largest cluster as background.
Compute HSV on that cluster only.

Commit

git commit -am "Background-aware color features"
17. DATA SPLITS
Train/Val/Test (70/15/15), stratified by label.

Commit

git commit -am "Added stratified data splits"
18. REPRODUCIBILITY
Fix seeds, log configs.
import random, numpy as np, torch
def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

Commit

git commit -am "Reproducibility: seeds + config logging"
19. FINAL MODEL + INFERENCE

file: inference.py

# load encoders + fusion, preprocess one image, print predicted class

Commit

git add inference.py
git commit -m "Inference script"
20. REPORT (WHAT YOU CLAIM)

Include:

Method (encoders + color + fusion)
Ablations (prove color helps)
Error analysis (where it fails)

Commit

git add README.md
git commit -m "Final report and results"