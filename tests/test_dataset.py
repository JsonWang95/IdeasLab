import os
import sys
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.dataset import GolfBallDataset

def visualize_sample(image, bbox):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Denormalize bbox coordinates
    height, width = image.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    xmin, xmax = xmin * width, xmax * width
    ymin, ymax = ymin * height, ymax * height
    
    # Create a Rectangle patch
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                         linewidth=2, edgecolor='r', facecolor='none')
    
    ax.add_patch(rect)
    plt.show()

def test_dataset_loading():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = GolfBallDataset(root_dir=os.path.join(project_root, 'data'), purpose='detection', split='train', transform=transform)

    print(f"Dataset size: {len(dataset)}")

    # Test loading a few samples
    for i in range(5): 
        image, bbox = dataset[i]
        print(f"Sample {i}:")
        print(f"Image shape: {image.shape}")
        print(f"Bounding box: {bbox}")

        image_np = image.permute(1, 2, 0).numpy()
        
        visualize_sample(image_np, bbox)

if __name__ == "__main__":
    test_dataset_loading()