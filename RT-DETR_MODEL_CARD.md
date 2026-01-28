# RT-DETR Model Card

## Model Overview
**RT-DETR** (Real-Time DEtection TRansformer) is an end-to-end transformer-based object detector designed for real-time inference.

## Key Information
- **Task**: Object Detection
- **Architecture**: Hybrid CNN-Transformer
- **Target Dataset**: COCO 2017
- **Target Hardware**: Tenstorrent Wormhole/Blackhole (N150/N300)
- **Input Resolution**: 640×640 RGB images
- **Output**: Bounding boxes + class labels (80 COCO classes)

## Architecture Components
1. **Backbone**: ResNet-50/101 (CNN feature extractor)
2. **Encoder**: Hybrid CNN-Transformer with multi-scale fusion
3. **Decoder**: Transformer decoder with dynamic query selection
4. **Detection Heads**: Classification + bbox regression

## Key Features
- Eliminates Non-Maximum Suppression (NMS)
- End-to-end differentiable
- Multi-scale feature processing
- Dynamic query-based detection

## Reference Implementation
- **Official Repository**: https://github.com/lyuwenyu/RT-DETR
- **Paper**: RT-DETR: DETRs Beat YOLOs on Real-time Object Detection (arXiv:2304.08069)
- **Pretrained Models**: Available on HuggingFace
- **Framework**: PyTorch/PaddlePaddle

## Target Performance (Stage 1)
- Runs on N150/N300 without errors
- PCC > 0.99 vs PyTorch reference
- Valid detections on COCO validation set
- mAP comparable to PyTorch baseline

## Notes
- Start with RT-DETR-R50 variant
- Batch size 1 for initial bring-up
- Follow YOLOv4 demo structure as reference