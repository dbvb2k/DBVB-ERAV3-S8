from __future__ import print_function
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from tqdm import tqdm
import logging
from Models import get_cifar10_model, get_cifar10_model2
from transforms import get_train_transforms, get_test_transforms, AlbumentationsDataset
from torchsummary import summary
import os
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize lists for tracking metrics
train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(f'Loss={loss.item():0.4f} Accuracy={100*correct/processed:0.2f}%')
    
    epoch_accuracy = 100 * correct / processed
    train_acc.append(epoch_accuracy)
    logger.info(f'Train Epoch: {epoch} Accuracy: {epoch_accuracy:.2f}%')
    return epoch_accuracy

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    accuracy = 100. * correct / len(test_loader.dataset)
    test_acc.append(accuracy)
    
    logger.info(f'Test Epoch: {epoch} Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy

def plot_metrics():
    fig, axs = plt.subplots(2,2,figsize=(15, 10))
    axs[0, 0].plot([t.item() for t in train_losses])
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()

if __name__ == '__main__':
    # CIFAR10 mean and std
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD = (0.2470, 0.2435, 0.2616)

    # Download and load the data
    logger.info("Downloading/Loading CIFAR10 dataset...")
    
    # Simple data directory setup that works in both environments
    DATA_DIR = './data'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    train_data = datasets.CIFAR10(DATA_DIR, train=True, download=True)
    test_data = datasets.CIFAR10(DATA_DIR, train=False, download=True)
    logger.info(f"Dataset loaded: {len(train_data)} training samples, {len(test_data)} test samples")

    # Apply transforms
    logger.info("Applying data transformations...")
    train_transform = get_train_transforms(CIFAR_MEAN, CIFAR_STD)
    test_transform = get_test_transforms(CIFAR_MEAN, CIFAR_STD)

    train_dataset = AlbumentationsDataset(train_data, train_transform)
    test_dataset = AlbumentationsDataset(test_data, test_transform)

    # Dataloaders
    EPOCHS = 60
    BATCH_SIZE = 128
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                             shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                            shuffle=False, num_workers=2)
    logger.info(f"Dataloaders created with batch size: {BATCH_SIZE}")

    # Model, Optimizer and Scheduler setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # model = get_cifar10_model().to(device)
    model = get_cifar10_model2().to(device)
    logger.info("Model architecture:")
    summary(model, (3, 32, 32))

    # Calculate total steps correctly
    total_steps = EPOCHS * len(train_loader)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.1,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
        div_factor=20,
        final_div_factor=100,
        anneal_strategy='cos'
    )
    logger.info("Optimizer and scheduler initialized")

    # Training loop
    logger.info("Starting training...")

    best_accuracy = 0

    # Create headers for our metrics table
    metrics_headers = ["Epoch", "Train Acc", "Test Acc", "Acc Diff", "Train Loss", "Test Loss"]
    epoch_metrics = []

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{EPOCHS}")
        train_accuracy = train(model, device, train_loader, optimizer, epoch)
        test_accuracy = test(model, device, test_loader, epoch)
        
        # Calculate average training loss for this epoch
        train_loss = sum(train_losses[-len(train_loader):]) / len(train_loader)
        test_loss = test_losses[-1]  # Latest test loss
        
        # Store metrics for this epoch
        epoch_metrics.append([
            epoch,
            f"{train_accuracy:.2f}%",
            f"{test_accuracy:.2f}%",
            f"{(train_accuracy - test_accuracy):.2f}%",
            f"{train_loss:.4f}",
            f"{test_loss:.4f}"
        ])
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            logger.info(f"New best accuracy: {best_accuracy:.2f}%")

    logger.info(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    
    # Display metrics table
    logger.info("\nEpoch-wise Training Summary:")
    print("\n" + tabulate(epoch_metrics, headers=metrics_headers, tablefmt="grid"))
    
    # Plot final metrics
    plot_metrics()





