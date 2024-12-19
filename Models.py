from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

# ======================================================================================================================

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        
        # C1 Block
        # RF start: 1
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),    # RF: 1 + (3-1) = 3
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),  # RF: 3 + (3-1)*1 = 5
            nn.BatchNorm2d(16),                                       # Jump = 2 after this layer
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # C2 Block - Dilated
        # RF: 5, jump = 2
        self.c2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, padding=4, dilation=4, bias=False),  # RF: 5 + (3-1)*2*4 = 21
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1, bias=False),    # RF: 21 + (3-1)*2 = 25
            nn.BatchNorm2d(32),                                       # Jump = 4 after this layer
            nn.ReLU(),
            nn.Dropout(0.1),            
        )
        
        # C3 Block - Depthwise Separable
        # RF: 25, jump = 4
        self.c3 = nn.Sequential(
            # Depthwise
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),   # RF: 25 + (3-1)*4 = 33
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Pointwise
            nn.Conv2d(32, 48, kernel_size=1, bias=False),    # RF: 33 (1x1 conv doesn't change RF)
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False),    # RF: 33 + (3-1)*4 = 41
            nn.BatchNorm2d(48),                              # Jump = 8 after this layer
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # C4 Block
        # RF: 41, jump = 8
        self.c4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1, bias=False),  # RF: 41 + (3-1)*8 = 57
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),    # RF: 57 + (3-1)*8 = 73
            # nn.BatchNorm2d(64),                              # Final RF = 73 > 44 (requirement met)
            # nn.ReLU(),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def get_cifar10_model():
    return CIFAR10Net()

# ======================================================================================================================
# Common function to save the model

def save_model(model, accuracy):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cifar10_model_{accuracy:.2f}_{timestamp}.pth"
    torch.save(model.state_dict(), f"models/{filename}")
    return filename

# ======================================================================================================================

class CIFAR10Net2(nn.Module):
    def __init__(self):
        super(CIFAR10Net2, self).__init__()
        
        # C1 Block
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1, bias=False),    # Reduced from 16 to 12
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(12, 24, kernel_size=3, padding=2, dilation=2, bias=False),  # Reduced from 32 to 24
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # C2 Block - Dilated
        self.c2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=4, dilation=4, bias=False),  # Reduced from 48 to 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),          
            nn.Conv2d(32, 48, kernel_size=3, padding=8, dilation=8, bias=False),  # Reduced from 64 to 48
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # C3 Block - Depthwise Separable with Dilation
        self.c3 = nn.Sequential(
            # Depthwise with dilation
            nn.Conv2d(48, 48, kernel_size=3, padding=6, dilation=6, groups=48, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),            
            # Pointwise
            nn.Conv2d(48, 64, kernel_size=1, bias=False),    # Reduced from 96 to 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4, bias=False),  # Reduced from 96 to 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # C4 Block with Dilation
        self.c4 = nn.Sequential(
            nn.Conv2d(64, 84, kernel_size=3, padding=2, dilation=2, bias=False),  # Reduced from 128 to 96
            nn.BatchNorm2d(84),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(84, 84, kernel_size=3, padding=2, dilation=2, bias=False),  # Reduced from 128 to 96
            # nn.BatchNorm2d(96),
            # nn.ReLU(),
            # nn.Dropout(0.1),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(84, 10)  # Changed from 128 to 96

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.gap(x)
        x = x.view(-1, 84)  # Changed from 128 to 96
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def get_cifar10_model2():
    return CIFAR10Net2()

# ======================================================================================================================

class CIFAR10Net3(nn.Module):
    def __init__(self):
        super(CIFAR10Net3, self).__init__()
        
        # C1 Block
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1, bias=False),    # Reduced from 16 to 12
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(12, 24, kernel_size=3, padding=2, dilation=2, bias=False),  # Reduced from 32 to 24
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # C2 Block - Dilated
        self.c2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=4, dilation=4, bias=False),  # Reduced from 48 to 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),          
            nn.Conv2d(32, 48, kernel_size=3, padding=8, dilation=8, bias=False),  # Reduced from 64 to 48
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # C3 Block - Depthwise Separable with Dilation
        self.c3 = nn.Sequential(
            # Depthwise with dilation
            nn.Conv2d(48, 48, kernel_size=3, padding=6, dilation=6, groups=48, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),            
            # Pointwise
            nn.Conv2d(48, 64, kernel_size=1, bias=False),    # Reduced from 96 to 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4, bias=False),  # Reduced from 96 to 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # C4 Block with Dilation
        self.c4 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=2, dilation=2, bias=False),  # Reduced from 128 to 96
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(96, 96, kernel_size=3, padding=2, dilation=2, bias=False),  # Reduced from 128 to 96
            # nn.BatchNorm2d(96),
            # nn.ReLU(),
            # nn.Dropout(0.1),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, 10)  # Changed from 128 to 96

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.gap(x)
        x = x.view(-1, 96)  # Changed from 128 to 96
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def get_cifar10_model3():
    return CIFAR10Net3()

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# You can test it like this:
if __name__ == "__main__":
    model = get_cifar10_model()
    model2 = get_cifar10_model2()
    model3 = get_cifar10_model3()
    print(f"Number of parameters: Model 1: {count_parameters(model):,}")
    print(f"Number of parameters: Model 2: {count_parameters(model2):,}")
    print(f"Number of parameters: Model 3: {count_parameters(model3):,}")



