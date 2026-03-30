"""
Fine-tunable BERT text encoder (nn.Module).

Gradients flow through the full BERT backbone.
Gradient checkpointing is enabled to save VRAM.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import config


class TextEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.tok   = AutoTokenizer.from_pretrained(config.BERT_MODEL)
        self.model = AutoModel.from_pretrained(config.BERT_MODEL).to(device)

        # Gradient checkpointing for memory efficiency
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:
            pass

    def forward(self, texts):
        """
        Args:
            texts: list of str
        Returns:
            Mean-pooled BERT features  (B, 768)
        """
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        out    = self.model(**enc).last_hidden_state   # (B, L, 768)
        pooled = out.mean(dim=1)                       # (B, 768)
        return pooled
