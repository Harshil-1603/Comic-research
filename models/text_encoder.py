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
        pooled = out.mean(dim=1)  # (B,768)
        return pooled
