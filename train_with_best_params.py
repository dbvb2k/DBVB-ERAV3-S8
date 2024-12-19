import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from tqdm import tqdm
import logging
from Models import get_cifar10_model, get_cifar10_model2
from transforms import get_train_transforms, get_test_transforms, AlbumentationsDataset
import os
import json
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingSession:
    def __init__(self, params_file="best_parameters.txt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = self.load_parameters(params_file)
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        
    def load_parameters(self, params_file):
        """Load best parameters from the file"""
        params = {}
        try:
            with open(params_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("Best test accuracy"):
                        continue
                    if ':' in line:
                        try:
                            key, value = [part.strip() for part in line.split(':', 1)]
                            # Convert string values to appropriate types
                            try:
                                value = eval(value)
                            except:
                                value = value
                            params[key] = value
                        except Exception as e:
                            logger.warning(f"Skipping line '{line}': {str(e)}")
                            
            if not params:
                raise ValueError("No valid parameters found in file")
                
            logger.info("Loaded parameters:")
            for key, value in params.items():
                logger.info(f"  {key}: {value}")
                
            return params
        except FileNotFoundError:
            logger.error(f"Parameters file '{params_file}' not found!")
            logger.info("Using default parameters...")
            # Provide default parameters
            return {
                "batch_size": 128,
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "max_lr": 0.1,
                "pct_start": 0.3,
                "div_factor": 10,
                "final_div_factor": 100,
                "epochs": 100
            }
    
    def get_dataloaders(self):
        """Create and return dataloaders"""
        DATA_DIR = './data'
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR_STD = (0.2470, 0.2435, 0.2616)

        train_data = datasets.CIFAR10(DATA_DIR, train=True, download=True)
        test_data = datasets.CIFAR10(DATA_DIR, train=False, download=True)

        train_transform = get_train_transforms(CIFAR_MEAN, CIFAR_STD)
        test_transform = get_test_transforms(CIFAR_MEAN, CIFAR_STD)

        train_dataset = AlbumentationsDataset(train_data, train_transform)
        test_dataset = AlbumentationsDataset(test_data, test_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=True,
            num_workers=4
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=False,
            num_workers=4
        )
        return train_loader, test_loader
    
    def train_epoch(self, model, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        model.train()
        pbar = tqdm(train_loader, desc="Training")
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss/(batch_idx+1),
                'accuracy': 100.*correct/total
            })
            
        return train_loss/len(train_loader), correct/total
    
    def test_epoch(self, model, test_loader):
        """Evaluate the model"""
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        return test_loss/len(test_loader), correct/total
    
    def train(self, epochs=None):
        """Full training loop"""
        if epochs is None:
            epochs = self.params.get('epochs', 30)  # Use 30 as default if not specified
            
        train_loader, test_loader = self.get_dataloaders()
        model = get_cifar10_model().to(self.device)
        # model = get_cifar10_model2().to(self.device)
        
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.params['lr'],
            momentum=self.params['momentum'],
            weight_decay=self.params['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.params['max_lr'],
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=self.params['pct_start'],
            div_factor=self.params['div_factor'],
            final_div_factor=self.params['final_div_factor']
        )
        
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            logger.info(f'\nEpoch: {epoch+1}/{epochs}')
            
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, scheduler)
            test_loss, test_acc = self.test_epoch(model, test_loader)
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
            logger.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%')
            
            # Save best model
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(model.state_dict(), 'best_model.pth')
                logger.info(f'Best model saved with accuracy: {best_accuracy*100:.2f}%')
        
        self.plot_training_results()
        return best_accuracy
    
    def plot_training_results(self):
        """Plot training and testing metrics"""
        plt.figure(figsize=(12, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Training and Testing Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.test_accuracies, label='Test Accuracy')
        plt.title('Training and Testing Accuracies')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()

if __name__ == "__main__":
    trainer = TrainingSession()
    # Train for more epochs than during optimization to achieve better accuracy
    best_accuracy = trainer.train(epochs=50)
    logger.info(f"Training completed. Best accuracy: {best_accuracy*100:.2f}%") 