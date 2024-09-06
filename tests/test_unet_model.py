import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.model import UNet
from src.dataset import GolfBallDataset

def test_unet_model():
    model = UNet(encoder_name='resnet18', in_channels=3, out_channels=4, output_stride=32)
    
    print(model)
    
    model.eval()
    
    sample_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"Model output shape with random input: {output.shape}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = GolfBallDataset(root_dir=os.path.join(project_root, 'data'), purpose='detection', transform=transform)
    
    image, bbox = dataset[0]
    
    print(f"Sample image shape: {image.shape}")
    print(f"Sample bbox: {bbox}")
    
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
    
    print(f"Model output shape with sample image: {output.shape}")
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    axs[0, 0].imshow(image.squeeze().permute(1, 2, 0))
    axs[0, 0].set_title("Input Image")
    axs[0, 0].axis('off')
    
    # Model output channels
    for i in range(4):
        row = (i + 1) // 3
        col = (i + 1) % 3
        axs[row, col].imshow(output.squeeze().cpu().numpy()[i], cmap='hot')
        axs[row, col].set_title(f"Channel {i}")
        axs[row, col].axis('off')
    
    fig.delaxes(axs[1, 2])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_unet_model()