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
