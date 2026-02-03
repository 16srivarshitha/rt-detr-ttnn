"""
RT-DETR reference using HuggingFace Transformers
This is more stable and compatible with modern PyTorch
"""
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from PIL import Image
import requests


def load_model(model_name="PekingU/rtdetr_r50vd_coco_o365"):
    """Load RT-DETR model from HuggingFace"""
    print(f"Loading model: {model_name}")
    
    image_processor = RTDetrImageProcessor.from_pretrained(model_name)
    model = RTDetrForObjectDetection.from_pretrained(model_name)
    model.eval()
    
    return model, image_processor


def run_inference(image_path, model, image_processor, threshold=0.5):
    """Run inference on an image"""
    # Load image
    if image_path.startswith('http'):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)
    
    # Preprocess
    inputs = image_processor(images=image, return_tensors="pt")
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs, 
        threshold=threshold, 
        target_sizes=target_sizes
    )[0]
    
    return results, image


def print_detections(results):
    """Print detection results"""
    scores = results["scores"].tolist()
    labels = results["labels"].tolist()
    boxes = results["boxes"].tolist()
    
    print(f"\nDetected {len(scores)} objects:")
    for score, label, box in zip(scores, labels, boxes):
        print(f"  Label: {label}, Score: {score:.3f}, Box: {box}")


if __name__ == '__main__':
    # Load model
    model, processor = load_model()
    
    print(f"\nModel architecture:")
    print(f"  Type: {type(model)}")
    print(f"  Config: {model.config}")
    
    # Test with a sample image
    test_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    print(f"\nRunning inference on test image: {test_url}")
    
    results, image = run_inference(test_url, model, processor)
    print_detections(results)
    
    print("\nModel loaded successfully!")
