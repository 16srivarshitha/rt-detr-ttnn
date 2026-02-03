"""
Generate PyTorch model graph for RT-DETR
Similar to YOLOv4 reference: https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov4/reference/yolov4_summary.py
"""
import torch
import sys
import os
from torchviz import make_dot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RT-DETR/rtdetr_pytorch'))

from src.core import YAMLConfig


def generate_model_graph(config_path, output_path='model_graph'):
    """
    Generate visual graph of RT-DETR model architecture
    
    Args:
        config_path: Path to model config YAML
        output_path: Output filename (without extension)
    """
    print("Loading model configuration...")
    cfg = YAMLConfig(config_path, resume=None)
    
    # Create dummy input
    batch_size = 1
    img_size = 640
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    
    print(f"Model architecture loaded")
    print(f"Input shape: {dummy_input.shape}")
    
    # Get model in eval mode
    model = cfg.model
    model.eval()
    
    print("Generating forward pass...")
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output type: {type(output)}")
    if isinstance(output, dict):
        print(f"Output keys: {output.keys()}")
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
    
    # Generate graph
    print(f"Generating computation graph...")
    dot = make_dot(output, params=dict(model.named_parameters()))
    
    # Save graph
    dot.format = 'pdf'
    dot.render(output_path, cleanup=True)
    print(f"Graph saved to {output_path}.pdf")
    
    return model, output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate RT-DETR model graph')
    parser.add_argument(
        '--config',
        type=str,
        default='RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml',
        help='Path to model config'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='docs/rtdetr_model_graph',
        help='Output path for graph PDF'
    )
    
    args = parser.parse_args()
    
    model, output = generate_model_graph(args.config, args.output)