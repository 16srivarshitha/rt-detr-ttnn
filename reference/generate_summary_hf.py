"""Generate model summary from HuggingFace RT-DETR"""
from transformers import RTDetrForObjectDetection
import torch

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def analyze_model(model_name="PekingU/rtdetr_r50vd_coco_o365"):
    print(f"Loading {model_name}...")
    model = RTDetrForObjectDetection.from_pretrained(model_name)
    
    total, trainable = count_parameters(model)
    
    print(f"\n{'='*80}")
    print(f"RT-DETR Model Summary")
    print(f"{'='*80}\n")
    print(f"Model: {model_name}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}\n")
    
    print(f"Architecture Components:")
    print(f"  - Backbone: ResNet-50")
    print(f"  - Encoder layers: {model.config.encoder_layers}")
    print(f"  - Decoder layers: {model.config.decoder_layers}")
    print(f"  - Num queries: {model.config.num_queries}")
    print(f"  - Hidden dim: {model.config.d_model}")
    print(f"  - Num classes: {len(model.config.id2label)}")
    
    # Save to file
    with open('docs/model_summary_hf.txt', 'w') as f:
        f.write(f"RT-DETR Model Summary (HuggingFace)\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Total parameters: {total:,}\n")
        f.write(f"Trainable parameters: {trainable:,}\n\n")
        f.write(f"Config:\n{model.config}\n")
    
    print(f"\nSummary saved to docs/model_summary_hf.txt")

if __name__ == '__main__':
    analyze_model()
