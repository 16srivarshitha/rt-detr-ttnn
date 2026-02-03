"""Minimal model loader that bypasses dataset imports"""
import torch
import torch.nn as nn
import yaml
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RT-DETR/rtdetr_pytorch'))

# Import only the model components we need
from src.nn.backbone.presnet import PResNet
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from src.zoo.rtdetr.rtdetr import RTDETR


def load_config(config_path):
    """Load YAML config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model_from_config(config_path):
    """Build RT-DETR model from config"""
    config = load_config(config_path)
    
    # Build backbone
    backbone_cfg = config.get('PResNet', {})
    backbone = PResNet(
        depth=backbone_cfg.get('depth', 50),
        variant=backbone_cfg.get('variant', 'd'),
        freeze_at=backbone_cfg.get('freeze_at', 0),
        return_idx=backbone_cfg.get('return_idx', [1, 2, 3]),
        num_stages=backbone_cfg.get('num_stages', 4),
        freeze_norm=backbone_cfg.get('freeze_norm', True)
    )
    
    # Build encoder
    encoder_cfg = config.get('HybridEncoder', {})
    encoder = HybridEncoder(
        in_channels=encoder_cfg.get('in_channels', [512, 1024, 2048]),
        feat_strides=encoder_cfg.get('feat_strides', [8, 16, 32]),
        hidden_dim=encoder_cfg.get('hidden_dim', 256),
        use_encoder_idx=encoder_cfg.get('use_encoder_idx', [2]),
        num_encoder_layers=encoder_cfg.get('num_encoder_layers', 1),
        nhead=encoder_cfg.get('nhead', 8),
        dim_feedforward=encoder_cfg.get('dim_feedforward', 1024),
        dropout=encoder_cfg.get('dropout', 0.0),
        enc_act=encoder_cfg.get('enc_act', 'gelu'),
        expansion=encoder_cfg.get('expansion', 1.0),
        depth_mult=encoder_cfg.get('depth_mult', 1),
        act=encoder_cfg.get('act', 'silu'),
    )
    
    # Build decoder
    decoder_cfg = config.get('RTDETRTransformer', {})
    decoder = RTDETRTransformer(
        num_classes=80,  # COCO classes
        hidden_dim=decoder_cfg.get('hidden_dim', 256),
        num_queries=decoder_cfg.get('num_queries', 300),
        feat_channels=decoder_cfg.get('feat_channels', [256, 256, 256]),
        feat_strides=decoder_cfg.get('feat_strides', [8, 16, 32]),
        num_levels=decoder_cfg.get('num_levels', 3),
        num_decoder_layers=decoder_cfg.get('num_decoder_layers', 6),
    )
    
    # Build full model
    model = RTDETR(backbone, encoder, decoder)
    
    return model


if __name__ == '__main__':
    config_path = 'RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml'
    
    print("Building model from config...")
    model = build_model_from_config(config_path)
    
    print(f"Model type: {type(model)}")
    print(f"Model: {model}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 640, 640)
    print(f"\nTesting forward pass with input shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output type: {type(output)}")
    if isinstance(output, dict):
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
