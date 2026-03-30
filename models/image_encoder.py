"""
Fine-tunable CLIP image encoder (nn.Module).

Gradients flow through the full ViT backbone.
Gradient checkpointing is enabled to save VRAM.
"""
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import config


class ImageEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.model     = CLIPModel.from_pretrained(config.CLIP_MODEL).to(device)
        self.processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)

        # Gradient checkpointing trades compute for memory — critical for 6GB VRAM
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:
            pass   # older transformers versions may not support this

    def forward(self, pil_images):
        """
        Args:
            pil_images: list of PIL.Image
        Returns:
            L2-normalised features  (B, 512)
        """
        inputs = self.processor(
            images=pil_images, return_tensors="pt"
        ).to(self.device)
        feats = self.model.get_image_features(**inputs)          # (B, 512)
        return feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
