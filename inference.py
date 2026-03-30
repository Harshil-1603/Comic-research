"""
Inference script for comic emotion classification
Load encoders + fusion model, preprocess one image, print predicted class
"""
import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import sys

from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.fusion_model import FusionModel
from features.color_features import hsv_hist

# Label mapping (reverse of label_map in dataset.py)
IDX_TO_LABEL = {0: "anger", 1: "sadness", 2: "joy", 3: "fear", 4: "neutral"}


def load_model(checkpoint_path, device="cuda"):
    """Load trained fusion model"""
    model = FusionModel().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image_path):
    """Load and preprocess image for inference"""
    # Load with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL for CLIP processor
    pil_img = Image.fromarray(img_rgb)
    
    return img_rgb, pil_img


def predict(image_path, text="", checkpoint_path="checkpoints/epoch_9.pt", device="cuda"):
    """
    Run inference on a single comic panel
    
    Args:
        image_path: Path to the comic panel image
        text: Optional dialogue text
        checkpoint_path: Path to trained model checkpoint
        device: 'cuda' or 'cpu'
    
    Returns:
        Predicted emotion label and confidence scores
    """
    # Load encoders and model
    img_enc = ImageEncoder(device)
    txt_enc = TextEncoder(device)
    model = load_model(checkpoint_path, device)
    
    # Preprocess image
    img_rgb, pil_img = preprocess_image(image_path)
    
    # Extract features
    with torch.no_grad():
        img_feat = img_enc([pil_img])
        txt_feat = txt_enc([text])
        col_feat = hsv_hist(img_rgb).unsqueeze(0).to(device)
        
        # Get prediction
        outputs = model(img_feat, txt_feat, col_feat)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    
    predicted_label = IDX_TO_LABEL[pred_idx]
    
    # Get all class probabilities
    class_probs = {IDX_TO_LABEL[i]: probs[0][i].item() for i in range(5)}
    
    return predicted_label, confidence, class_probs


def main():
    parser = argparse.ArgumentParser(description="Comic Emotion Classification Inference")
    parser.add_argument("image", type=str, help="Path to comic panel image")
    parser.add_argument("--text", type=str, default="", help="Optional dialogue text")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/epoch_9.pt", 
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"\nRunning inference on: {args.image}")
    print(f"Text: '{args.text}'")
    print("-" * 50)
    
    try:
        label, conf, probs = predict(
            args.image, 
            args.text, 
            args.checkpoint, 
            args.device
        )
        
        print(f"\nPredicted Emotion: {label.upper()}")
        print(f"Confidence: {conf:.4f}")
        print("\nClass Probabilities:")
        for emotion, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 20)
            print(f"  {emotion:10s}: {prob:.4f} {bar}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
