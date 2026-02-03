"""
Generate detailed model summary for RT-DETR
Extract layer info, parameters, and operations for TTNN conversion planning

Reference: https://github.com/tenstorrent/tt-metal/blob/main/models/demos/yolov4/reference/yolov4_summary.py
"""
import torch
import torch.nn as nn
import sys
import os
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RT-DETR/rtdetr_pytorch'))

from src.core import YAMLConfig


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_layer_info(module, module_name=""):
    """Recursively extract layer information"""
    layers_info = []
    
    for name, layer in module.named_children():
        full_name = f"{module_name}.{name}" if module_name else name
        
        layer_dict = {
            'name': full_name,
            'type': type(layer).__name__,
            'params': sum(p.numel() for p in layer.parameters()),
        }
        
        # Add specific info for common layer types
        if isinstance(layer, nn.Conv2d):
            layer_dict.update({
                'in_channels': layer.in_channels,
                'out_channels': layer.out_channels,
                'kernel_size': layer.kernel_size,
                'stride': layer.stride,
                'padding': layer.padding,
                'groups': layer.groups,
                'bias': layer.bias is not None,
            })
        elif isinstance(layer, nn.Linear):
            layer_dict.update({
                'in_features': layer.in_features,
                'out_features': layer.out_features,
                'bias': layer.bias is not None,
            })
        elif isinstance(layer, (nn.BatchNorm2d, nn.LayerNorm)):
            layer_dict.update({
                'num_features': layer.num_features if hasattr(layer, 'num_features') else 'N/A',
                'eps': layer.eps,
                'momentum': getattr(layer, 'momentum', 'N/A'),
            })
        elif isinstance(layer, nn.MultiheadAttention):
            layer_dict.update({
                'embed_dim': layer.embed_dim,
                'num_heads': layer.num_heads,
                'dropout': layer.dropout,
            })
        
        layers_info.append(layer_dict)
        
        # Recursively process child modules
        if len(list(layer.children())) > 0:
            layers_info.extend(get_layer_info(layer, full_name))
    
    return layers_info


def analyze_model_structure(model):
    """Analyze overall model structure"""
    structure = {
        'total_layers': 0,
        'conv_layers': 0,
        'linear_layers': 0,
        'attention_layers': 0,
        'norm_layers': 0,
        'activation_layers': 0,
    }
    
    for module in model.modules():
        structure['total_layers'] += 1
        if isinstance(module, nn.Conv2d):
            structure['conv_layers'] += 1
        elif isinstance(module, nn.Linear):
            structure['linear_layers'] += 1
        elif isinstance(module, nn.MultiheadAttention):
            structure['attention_layers'] += 1
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            structure['norm_layers'] += 1
        elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU)):
            structure['activation_layers'] += 1
    
    return structure


def generate_model_summary(config_path, output_file='docs/model_summary.txt'):
    """Generate comprehensive model summary"""
    print("Loading model configuration...")
    cfg = YAMLConfig(config_path, resume=None)
    
    model = cfg.model
    model.eval()
    
    print("Analyzing model structure...")
    
    # Get parameter counts
    total_params, trainable_params = count_parameters(model)
    
    # Get layer information
    layers_info = get_layer_info(model)
    
    # Get structure analysis
    structure = analyze_model_structure(model)
    
    # Write summary to file
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RT-DETR Model Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # Parameter counts
        f.write("Parameter Counts:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n\n")
        
        # Structure summary
        f.write("Model Structure:\n")
        f.write("-" * 40 + "\n")
        for key, value in structure.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Detailed layer information
        f.write("Detailed Layer Information:\n")
        f.write("=" * 80 + "\n\n")
        
        for i, layer in enumerate(layers_info):
            f.write(f"Layer {i}: {layer['name']}\n")
            f.write(f"  Type: {layer['type']}\n")
            f.write(f"  Parameters: {layer['params']:,}\n")
            
            # Write layer-specific details
            for key, value in layer.items():
                if key not in ['name', 'type', 'params']:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"Model summary saved to {output_file}")
    print(f"\nQuick Stats:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total layers: {structure['total_layers']}")
    print(f"  Conv layers: {structure['conv_layers']}")
    print(f"  Linear layers: {structure['linear_layers']}")
    print(f"  Attention layers: {structure['attention_layers']}")
    
    return layers_info, structure


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate RT-DETR model summary')
    parser.add_argument(
        '--config',
        type=str,
        default='RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml',
        help='Path to model config'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='docs/model_summary.txt',
        help='Output path for summary file'
    )
    
    args = parser.parse_args()
    
    layers_info, structure = generate_model_summary(args.config, args.output)
