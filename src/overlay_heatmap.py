import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np

from dataset import GolfBallDataset
from timm_unet import Unet
from utils import overlay_heatmap

# Define the normalization constants
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def visualize_tracking_results(model, tracking_loader, device, save_dir):
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, heatmaps, _) in enumerate(tqdm(tracking_loader, desc="Visualizing tracking results")):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            
            outputs = model(images)
            pred_heatmaps = torch.sigmoid(outputs)
            
            for j, (image, heatmap, pred_heatmap) in enumerate(zip(images, heatmaps, pred_heatmaps)):
                # Denormalize and convert image to numpy array
                image_np = image.cpu().permute(1, 2, 0).numpy()
                image_np = std * image_np + mean
                image_np = np.clip(image_np, 0, 1)

                # Overlay ground truth heatmap
                heatmap_np = heatmap.cpu().squeeze().numpy()
                overlaid_gt = overlay_heatmap(image_np, heatmap_np)

                # Overlay predicted heatmap
                pred_heatmap_np = pred_heatmap.cpu().squeeze().numpy()
                overlaid_pred = overlay_heatmap(image_np, pred_heatmap_np)

                # Create a figure with three subplots
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

                ax1.imshow(image_np)
                ax1.set_title('Input Image')
                ax1.axis('off')

                ax2.imshow(overlaid_gt)
                ax2.set_title('Ground Truth Heatmap Overlay')
                ax2.axis('off')

                ax3.imshow(overlaid_pred)
                ax3.set_title('Predicted Heatmap Overlay')
                ax3.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'tracking_overlay_batch_{i}_sample_{j}.png'))
                plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model
    model = Unet(
        backbone='mobilenetv2_100',
        in_chans=3,
        num_classes=1,
        decoder_channels=(256, 128, 64, 32, 16),
        center=True
    ).to(device)

    # Adjust the path to your best model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "timm_unet_training_results_fixed_data")
    best_model_path = os.path.join(model_dir, "best_model.pth")
    
    if not os.path.exists(best_model_path):
        print(f"Model not found at {best_model_path}")
        print("Available files in the directory:")
        print(os.listdir(model_dir))
        return

    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Prepare the tracking dataset
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])
    
    tracking_dataset = GolfBallDataset(root_dir='data', purpose='tracking', split='test', transform=test_transform)
    tracking_loader = DataLoader(tracking_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Create a directory for saving results
    save_dir = 'tracking_visualization_results'
    os.makedirs(save_dir, exist_ok=True)

    # Visualize tracking results
    visualize_tracking_results(model, tracking_loader, device, save_dir)
    
    print(f"Visualization completed. Results saved in {save_dir}")

if __name__ == "__main__":
    main()