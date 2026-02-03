"""Test if we can load the model without dataset dependencies"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RT-DETR/rtdetr_pytorch'))

# Patch the import before loading
import torchvision
if not hasattr(torchvision, 'datapoints'):
    import torchvision.transforms.v2 as transforms_v2
    if hasattr(transforms_v2, 'datapoints'):
        torchvision.datapoints = transforms_v2.datapoints
    elif hasattr(transforms_v2, 'tv_tensors'):
        torchvision.datapoints = transforms_v2.tv_tensors
    else:
        # Create a dummy module
        class DummyDatapoints:
            pass
        torchvision.datapoints = DummyDatapoints()

from src.core import YAMLConfig

config_path = 'RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml'
print("Loading config...")
cfg = YAMLConfig(config_path, resume=None)
print("Config loaded successfully!")
print(f"Model type: {type(cfg.model)}")
