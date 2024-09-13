import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
from dataset import GolfBallDataset, JointRandomHorizontalFlip, JointRandomRotation, JointCompose
from timm_unet import Unet 
from utils import visualize_heatmap_prediction, save_checkpoint, load_checkpoint, visualize_random_samples

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, heatmaps, _ in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, heatmaps)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, heatmaps, _ in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def freeze_encoder(model):
    for param in model.encoder.parameters():
        param.requires_grad = False

def unfreeze_encoder_layer(model, layers_to_unfreeze):
    for i, param in enumerate(reversed(list(model.encoder.parameters()))):
        if i < layers_to_unfreeze:
            param.requires_grad = True
        else:
            break

def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Joint transforms for both image and heatmap
    joint_transform = JointCompose([
        JointRandomHorizontalFlip(),
        JointRandomRotation(10),
    ])

    # Transforms for image only
    image_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset and DataLoader
    train_dataset = GolfBallDataset(root_dir='data', purpose='detection', split='train', 
                                    transform=image_transform, joint_transform=joint_transform)
    val_dataset = GolfBallDataset(root_dir='data', purpose='detection', split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model, loss, and optimizer
    model = Unet(
        backbone='mobilenetv2_100',
        in_chans=3,
        num_classes=1,
        decoder_channels=(256, 128, 64, 32, 16),
        center=True
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # Create a directory for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'training_results_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    freeze_encoder(model)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(save_dir, 'best_model.pth'))
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:  # Visualize every 5 epochs and on the last epoch
            visualize_random_samples(model, train_loader, device, save_dir=os.path.join(save_dir, f'train_samples_epoch_{epoch+1}'))
            visualize_random_samples(model, val_loader, device, save_dir=os.path.join(save_dir, f'val_samples_epoch_{epoch+1}'))
        
        # Visualize predictions
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                images, heatmaps, _ = next(iter(val_loader))
                images = images.to(device)
                outputs = model(images)
                
                for i in range(min(3, len(images))):  # Visualize up to 3 samples
                    visualize_heatmap_prediction(
                        images[i].cpu(),
                        heatmaps[i].cpu(),
                        outputs[i].cpu(),
                        save_path=os.path.join(save_dir, f'prediction_epoch_{epoch+1}_sample_{i+1}.png')
                    )
        
        if epoch == 5:  # Start unfreezing after 5 epochs
            unfreeze_encoder_layer(model, 10)  # Unfreeze last 10 layers
        elif epoch == 10:
            unfreeze_encoder_layer(model, 20)  # Unfreeze last 20 layers
        elif epoch == 15:
            unfreeze_encoder_layer(model, 30)  # Unfreeze last 30 layers
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    
    print("Training completed!")

if __name__ == "__main__":
    main()