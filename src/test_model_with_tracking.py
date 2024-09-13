import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

from dataset import GolfBallDataset
from timm_unet import Unet
from utils import visualize_heatmap_overlay, heatmap_iou

def calculate_center_distance(pred_heatmap, true_heatmap):
    pred_center = np.unravel_index(pred_heatmap.argmax(), pred_heatmap.shape)
    true_center = np.unravel_index(true_heatmap.argmax(), true_heatmap.shape)
    return np.sqrt((pred_center[0] - true_center[0])**2 + (pred_center[1] - true_center[1])**2)

def test_model(model, test_loader, device, save_dir, dataset_name, iou_threshold=0.5, confidence_threshold=0.5):
    model.eval()
    total_iou = 0
    num_samples = 0
    all_distances = []
    all_confidences = []
    all_true_labels = []

    with torch.no_grad():
        for i, (images, heatmaps, _) in enumerate(tqdm(test_loader, desc=f"Testing on {dataset_name}")):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            
            outputs = model(images)
            pred_heatmaps = torch.sigmoid(outputs)
            
            # Calculate IoU
            batch_iou = heatmap_iou(pred_heatmaps, heatmaps, threshold=iou_threshold)
            total_iou += batch_iou.sum().item()
            num_samples += images.size(0)

            # Calculate center distance
            for pred, true in zip(pred_heatmaps.cpu().numpy(), heatmaps.cpu().numpy()):
                distance = calculate_center_distance(pred[0], true[0])
                all_distances.append(distance)

            # Store confidences and true labels for precision-recall curve
            all_confidences.extend(pred_heatmaps.max(dim=2)[0].max(dim=2)[0].cpu().numpy().flatten())
            all_true_labels.extend((heatmaps.max(dim=2)[0].max(dim=2)[0] > iou_threshold).cpu().numpy().flatten())

            # Visualize predictions
            if i % 5 == 0:  # Visualize every 5th batch
                for j in range(min(3, images.size(0))):  # Visualize up to 3 samples per batch
                    visualize_heatmap_overlay(
                        images[j].cpu(),
                        heatmaps[j].cpu(),
                        outputs[j].cpu(),
                        save_path=os.path.join(save_dir, f'{dataset_name}_prediction_overlay_batch_{i}_sample_{j}.png')
                    )

    avg_iou = total_iou / num_samples
    avg_distance = np.mean(all_distances)
    detection_rate = np.mean(np.array(all_confidences) > confidence_threshold)

    # Calculate precision-recall curve and average precision
    precision, recall, _ = precision_recall_curve(all_true_labels, all_confidences)
    avg_precision = average_precision_score(all_true_labels, all_confidences)

    # Plot precision-recall curve
    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_precision_recall_curve.png'))
    plt.close()

    # Plot confidence distribution
    plt.figure()
    plt.hist(np.array(all_confidences)[np.array(all_true_labels) == 1], bins=50, alpha=0.5, label='True Positives')
    plt.hist(np.array(all_confidences)[np.array(all_true_labels) == 0], bins=50, alpha=0.5, label='False Positives')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title(f'Confidence Distribution - {dataset_name}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_confidence_distribution.png'))
    plt.close()

    return avg_iou, avg_distance, detection_rate, avg_precision

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

    # Prepare the datasets
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    detection_test_dataset = GolfBallDataset(root_dir='data', purpose='detection', split='test', transform=test_transform)
    detection_test_loader = DataLoader(detection_test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    tracking_dataset = GolfBallDataset(root_dir='data', purpose='tracking', split='test', transform=test_transform)
    tracking_loader = DataLoader(tracking_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Create a directory for saving results
    save_dir = 'comprehensive_test_results'
    os.makedirs(save_dir, exist_ok=True)

    # Test on detection test set
    detection_avg_iou, detection_avg_distance, detection_rate, detection_avg_precision = test_model(
        model, detection_test_loader, device, save_dir, 'detection_test', iou_threshold=0.5, confidence_threshold=0.5
    )
    
    # Test on tracking data
    tracking_avg_iou, tracking_avg_distance, tracking_rate, tracking_avg_precision = test_model(
        model, tracking_loader, device, save_dir, 'tracking', iou_threshold=0.5, confidence_threshold=0.5
    )
    
    print(f"Detection Test Set Results:")
    print(f"  Average IoU: {detection_avg_iou:.4f}")
    print(f"  Average Center Distance: {detection_avg_distance:.4f}")
    print(f"  Detection Rate: {detection_rate:.4f}")
    print(f"  Average Precision: {detection_avg_precision:.4f}")
    
    print(f"\nTracking Dataset Results:")
    print(f"  Average IoU: {tracking_avg_iou:.4f}")
    print(f"  Average Center Distance: {tracking_avg_distance:.4f}")
    print(f"  Detection Rate: {tracking_rate:.4f}")
    print(f"  Average Precision: {tracking_avg_precision:.4f}")

    # Save the results to a file
    with open(os.path.join(save_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Detection Test Set Results:\n")
        f.write(f"  Average IoU: {detection_avg_iou:.4f}\n")
        f.write(f"  Average Center Distance: {detection_avg_distance:.4f}\n")
        f.write(f"  Detection Rate: {detection_rate:.4f}\n")
        f.write(f"  Average Precision: {detection_avg_precision:.4f}\n\n")
        f.write(f"Tracking Dataset Results:\n")
        f.write(f"  Average IoU: {tracking_avg_iou:.4f}\n")
        f.write(f"  Average Center Distance: {tracking_avg_distance:.4f}\n")
        f.write(f"  Detection Rate: {tracking_rate:.4f}\n")
        f.write(f"  Average Precision: {tracking_avg_precision:.4f}\n")

if __name__ == "__main__":
    main()