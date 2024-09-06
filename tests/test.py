import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import GolfBallDataset
from src.model import UNet
from src.utils import visualize_heatmap_prediction
import matplotlib.pyplot as plt
import numpy as np

def test_model(model, test_loader, device, save_dir):
    model.eval()
    total_iou = 0
    num_samples = 0

    with torch.no_grad():
        for i, (images, heatmaps) in enumerate(test_loader):
            images, heatmaps = images.to(device), heatmaps.to(device)
            outputs = model(images)
            
            # Calculate IoU
            pred_heatmaps = torch.sigmoid(outputs)
            iou = calculate_iou(pred_heatmaps, heatmaps)
            total_iou += iou.sum().item()
            num_samples += images.size(0)
            
            if i < 5:
                for j in range(min(images.size(0), 5)):
                    image = images[j].cpu()
                    heatmap = heatmaps[j].cpu()
                    prediction = pred_heatmaps[j].cpu()
                    
                    visualize_heatmap_prediction(image, heatmap, prediction)
                    plt.savefig(os.path.join(save_dir, f'tracking_prediction_{i+1}_{j+1}.png'))
                    plt.close()

    avg_iou = total_iou / num_samples
    print(f"Average IoU on tracking data: {avg_iou:.4f}")
    
    return avg_iou

def calculate_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return intersection / (union + 1e-6)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(encoder_name='resnet18', in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load('training_results/best_model.pth', map_location=device))
    
    test_dataset = GolfBallDataset(root_dir='data', purpose='tracking', split='test', image_size=256)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    save_dir = 'tracking_results'
    os.makedirs(save_dir, exist_ok=True)
    
    avg_iou = test_model(model, test_loader, device, save_dir)
    
    with open(os.path.join(save_dir, 'tracking_results.txt'), 'w') as f:
        f.write(f"Average IoU on tracking data: {avg_iou:.4f}\n")

if __name__ == "__main__":
    main()