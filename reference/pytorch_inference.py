"""
PyTorch reference inference script for RT-DETR
This script demonstrates how to load and run the RT-DETR model
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import sys
import os

# Add RT-DETR to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RT-DETR/rtdetr_pytorch'))

from src.core import YAMLConfig


class RTDETRReference(nn.Module):
    """Wrapper for RT-DETR model in inference mode"""
    
    def __init__(self, config_path, checkpoint_path, device='cpu'):
        super().__init__()
        self.device = device
        
        # Load config and checkpoint
        cfg = YAMLConfig(config_path, resume=checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load state dict
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        
        cfg.model.load_state_dict(state)
        
        # Deploy model
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.model.to(device)
        
    def forward(self, images, orig_sizes):
        """
        Args:
            images: Tensor [B, 3, H, W]
            orig_sizes: Tensor [B, 2] - original image sizes (width, height)
        Returns:
            labels, boxes, scores
        """
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_sizes)
        return outputs


def load_image(image_path, size=(640, 640)):
    """Load and preprocess image"""
    im = Image.open(image_path).convert('RGB')
    orig_size = torch.tensor([im.size[0], im.size[1]])  # width, height
    
    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
    ])
    
    im_tensor = transforms(im).unsqueeze(0)  # Add batch dimension
    return im_tensor, orig_size.unsqueeze(0), im


def main():
    # Paths (to be updated when we download weights)
    config_path = 'RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml'
    checkpoint_path = 'path/to/checkpoint.pth'  # TODO: Download weights
    image_path = 'path/to/test/image.jpg'  # TODO: Add test image
    
    # Initialize model
    model = RTDETRReference(config_path, checkpoint_path, device='cpu')
    model.eval()
    
    # Load image
    image_tensor, orig_size, orig_image = load_image(image_path)
    
    # Run inference
    with torch.no_grad():
        labels, boxes, scores = model(image_tensor, orig_size)
    
    print(f"Detected {len(labels[0])} objects")
    print(f"Labels shape: {labels[0].shape}")
    print(f"Boxes shape: {boxes[0].shape}")
    print(f"Scores shape: {scores[0].shape}")
    
    return labels, boxes, scores


if __name__ == '__main__':
    main()