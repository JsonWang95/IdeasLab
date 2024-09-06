import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from dataset import GolfBallDataset
from model import UNet
from utils import visualize_heatmap_prediction

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, save_dir):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, heatmaps) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, heatmaps = images.to(device), heatmaps.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        
        if (epoch + 1) % 5 == 0 or epoch == 0:  # Visualize at the start and every 5 epochs
            visualize_sample_prediction(model, val_loader, device, epoch, save_dir)
    
    plot_training_progress(train_losses, val_losses, save_dir)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, heatmaps in data_loader:
            images, heatmaps = images.to(device), heatmaps.to(device)
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            running_loss += loss.item() * images.size(0)
    
    return running_loss / len(data_loader.dataset)

def visualize_sample_prediction(model, data_loader, device, epoch, save_dir):
    model.eval()
    images, heatmaps = next(iter(data_loader))
    images, heatmaps = images.to(device), heatmaps.to(device)
    
    with torch.no_grad():
        outputs = model(images)
    
    image = images[0].cpu()
    heatmap = heatmaps[0].cpu()
    prediction = outputs[0].cpu()
    
    visualize_heatmap_prediction(image, heatmap, prediction)
    plt.savefig(os.path.join(save_dir, f'prediction_epoch_{epoch+1}.png'))
    plt.close()

def plot_training_progress(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_progress.png'))
    plt.close()

def main():
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #print(f"CUDA available: {torch.cuda.is_available()}")
    #print(f"Current device: {torch.cuda.current_device()}")
    #print(f"Device name: {torch.cuda.get_device_name(0)}")

    train_dataset = GolfBallDataset(root_dir='data', purpose='detection', split='train', transform=train_transform)
    val_dataset = GolfBallDataset(root_dir='data', purpose='detection', split='val')
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Check the first batch
    images, heatmaps = next(iter(train_loader))
    print(f"Batch shape - Images: {images.shape}, Heatmaps: {heatmaps.shape}")

    model = UNet(encoder_name='resnet18', in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    save_dir = 'training_results'
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, save_dir)

    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))

if __name__ == "__main__":
    main()