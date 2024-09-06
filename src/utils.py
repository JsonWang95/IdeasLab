import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_heatmap_prediction(image, target, prediction):
    """
    Visualize the original image with target and predicted heatmaps.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax1.imshow(image.permute(1, 2, 0))
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Target heatmap
    ax2.imshow(image.permute(1, 2, 0))
    ax2.imshow(target.squeeze(), cmap='hot', alpha=0.6)
    ax2.set_title('Target Heatmap')
    ax2.axis('off')
    
    # Predicted heatmap
    ax3.imshow(image.permute(1, 2, 0))
    ax3.imshow(torch.sigmoid(prediction).squeeze(), cmap='hot', alpha=0.6)
    ax3.set_title('Predicted Heatmap')
    ax3.axis('off')
    
    plt.tight_layout()

def visualize_bbox_prediction(image, target, prediction):
    """
    Visualize the original image, ground truth, and prediction with bounding boxes.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax1.imshow(image.permute(1, 2, 0))
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Ground truth
    ax2.imshow(image.permute(1, 2, 0))
    ax2.add_patch(plt.Rectangle((target[0], target[1]), target[2]-target[0], target[3]-target[1],
                                fill=False, edgecolor='r', linewidth=2))
    ax2.set_title('Ground Truth')
    ax2.axis('off')
    
    # Prediction
    ax3.imshow(image.permute(1, 2, 0))
    ax3.add_patch(plt.Rectangle((prediction[0], prediction[1]), prediction[2]-prediction[0], prediction[3]-prediction[1],
                                fill=False, edgecolor='g', linewidth=2))
    ax3.set_title('Prediction')
    ax3.axis('off')
    
    plt.tight_layout()

def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection / float(box1_area + box2_area - intersection)
    return iou

def evaluate_model(model, test_loader, device):
    model.eval()
    total_iou = 0
    num_samples = 0
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
            for pred, target in zip(outputs, targets):
                total_iou += iou(pred.cpu().numpy(), target.cpu().numpy())
                num_samples += 1
    
    return total_iou / num_samples