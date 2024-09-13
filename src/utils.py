import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import cv2

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def visualize_heatmap_prediction(image, heatmap, prediction, save_path=None, threshold=0.5):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    
    # Denormalize and convert image to numpy array
    image = image.permute(1, 2, 0).numpy()
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    ax1.imshow(image)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    ax2.imshow(heatmap.squeeze(), cmap='hot', interpolation='nearest')
    ax2.set_title('Ground Truth Heatmap')
    ax2.axis('off')
    
    pred_raw = torch.sigmoid(prediction).squeeze().cpu().numpy()
    ax3.imshow(pred_raw, cmap='hot', interpolation='nearest')
    ax3.set_title('Raw Predicted Heatmap')
    ax3.axis('off')
    
    pred_thresholded = (pred_raw > threshold).astype(float)
    ax4.imshow(pred_thresholded, cmap='hot', interpolation='nearest')
    ax4.set_title(f'Thresholded Prediction (>{threshold})')
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_random_samples(model, dataloader, device, num_samples=5, save_dir='random_samples'):
    model.eval()
    all_images = []
    all_heatmaps = []
    all_outputs = []
    
    with torch.no_grad():
        for images, heatmaps, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            all_images.extend(images.cpu())
            all_heatmaps.extend(heatmaps)
            all_outputs.extend(outputs.cpu())
    
    indices = np.random.choice(len(all_images), num_samples, replace=False)
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for i, idx in enumerate(indices):
        visualize_heatmap_prediction(
            all_images[idx],
            all_heatmaps[idx],
            all_outputs[idx],
            save_path=os.path.join(save_dir, f'random_sample_prediction_{i}.png')
        )

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

def save_checkpoint(model, optimizer, epoch, val_loss, filename):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename, model, optimizer=None):
    """
    Load model checkpoint
    """
    if not os.path.isfile(filename):
        raise ValueError(f"File doesn't exist {filename}")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    print(f"Loaded checkpoint from epoch {epoch} with validation loss {val_loss}")
    return model, optimizer, epoch, val_loss

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

def heatmap_iou(pred, target, threshold=0.5):
    """
    Calculate IoU between predicted and target heatmaps.
    
    Args:
    pred (torch.Tensor): Predicted heatmap
    target (torch.Tensor): Target heatmap
    threshold (float): Threshold to consider a pixel as positive
    
    Returns:
    torch.Tensor: IoU score
    """
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)  # Adding small epsilon to avoid division by zero
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

import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def overlay_heatmap(image, heatmap, alpha=0.7, colormap=cv2.COLORMAP_JET):
    # Ensure image is in the correct format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Convert heatmap to uint8 and apply colormap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
    
    # Convert both to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Overlay heatmap on image
    overlaid = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Convert back to RGB for matplotlib
    return cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)

def visualize_heatmap_overlay(image, heatmap, prediction, save_path=None, threshold=0.5):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Denormalize and convert image to numpy array
    image = image.permute(1, 2, 0).numpy()
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    ax1.imshow(image)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # Overlay ground truth heatmap
    heatmap_np = heatmap.squeeze().numpy()
    overlaid_gt = overlay_heatmap(image, heatmap_np)
    ax2.imshow(overlaid_gt)
    ax2.set_title('Ground Truth Heatmap Overlay')
    ax2.axis('off')
    
    # Overlay predicted heatmap
    pred_np = torch.sigmoid(prediction).squeeze().cpu().numpy()
    overlaid_pred = overlay_heatmap(image, pred_np)
    ax3.imshow(overlaid_pred)
    ax3.set_title('Predicted Heatmap Overlay')
    ax3.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()