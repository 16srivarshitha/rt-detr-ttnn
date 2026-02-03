# Setup script for RT-DETR PyTorch reference implementation

echo "Setting up RT-DETR reference implementation..."

# Clone RT-DETR if not already present
if [ ! -d "RT-DETR" ]; then
    echo "Cloning RT-DETR repository..."
    git clone https://github.com/lyuwenyu/RT-DETR.git
    echo "RT-DETR cloned successfully"
else
    echo "RT-DETR already exists, skipping clone"
fi

# Install RT-DETR dependencies
echo "Installing RT-DETR dependencies..."
pip install -r RT-DETR/rtdetr_pytorch/requirements.txt

echo "Setup complete!"
echo "Next steps:"
echo "1. Download pretrained weights"
echo "2. Add test images to ../data/"
echo "3. Run inference with pytorch_inference.py"