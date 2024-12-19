from __future__ import print_function
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
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from ray import train
import pandas as pd
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_cifar(config, checkpoint_dir=None):
    # Data loading
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
        batch_size=int(config["batch_size"]), 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=int(config["batch_size"]), 
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    model = get_cifar10_model2().to(device)

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )

    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["max_lr"],
        epochs=config["epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=config["pct_start"],
        div_factor=config["div_factor"],
        final_div_factor=config["final_div_factor"]
    )

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Calculate average training loss
        train_loss = train_loss / len(train_loader)

        # Test accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        
        # Report both metrics
        train.report({
            "accuracy": accuracy,
            "train_loss": train_loss,
            "epoch": epoch
        })

def tune_cifar():
    # Add custom trial name creator
    def trial_name_creator(trial):
        return f"trial_{trial.trial_id}"

    # Create absolute path for storage
    storage_path = os.path.abspath("ray_results")
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "momentum": tune.uniform(0.8, 0.99),
        "batch_size": tune.choice([64, 128, 256]),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "epochs": 50,
        "max_lr": tune.loguniform(1e-3, 1),
        "pct_start": tune.uniform(0.1, 0.4),
        "div_factor": tune.choice([10, 20, 25]),
        "final_div_factor": tune.choice([50, 100, 1000])
    }

    scheduler = ASHAScheduler(
        max_t=20,
        grace_period=1,
        reduction_factor=2,
        metric="accuracy",
        mode="max"
    )

    reporter = CLIReporter(
        parameter_columns=["lr", "momentum", "batch_size", "weight_decay", "max_lr"],
        metric_columns=["accuracy", "training_iteration"]
    )

    result = tune.run(
        train_cifar,
        config=config,
        num_samples=20,
        scheduler=scheduler,
        progress_reporter=reporter,
        resources_per_trial={"cpu": 2, "gpu": 0.5},
        storage_path=storage_path,  # Using absolute path
        trial_dirname_creator=trial_name_creator,
        name="cifar_tune"
    )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

def analyze_results(experiment_path=None):
    """
    Analyze and visualize results from Ray Tune experiments.
    Args:
        experiment_path (str, optional): Path to specific experiment. 
                                       If None, uses most recent experiment.
    """
    # If no specific path provided, get the most recent experiment
    if experiment_path is None:
        base_dir = os.path.abspath("ray_results")
        if not os.path.exists(base_dir):
            print("No results found! Run tune_cifar() first.")
            return
            
        experiments = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
        if not experiments:
            print("No experiments found in ray_results directory!")
            return
            
        experiment_path = max(experiments, key=os.path.getmtime)
    
    # Load the experiment analysis
    analysis = ExperimentAnalysis(experiment_path)
    
    # Get all trials dataframe
    df = analysis.dataframe()
    
    # Print best configuration and results
    best_trial = analysis.get_best_trial("accuracy", "max", "last")
    print("\n=== Best Trial Results ===")
    print(f"Best trial last accuracy: {best_trial.last_result['accuracy']:.4f}")
    print("\nBest trial config:")
    for param, value in best_trial.config.items():
        print(f"{param}: {value}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Accuracy distribution
    plt.subplot(2, 2, 1)
    df['accuracy'].hist(bins=20)
    plt.title('Distribution of Final Accuracies')
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    
    # Plot 2: Learning rate vs Accuracy
    plt.subplot(2, 2, 2)
    plt.scatter(df['config/lr'], df['accuracy'])
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Learning Rate vs Accuracy')
    
    # Plot 3: Batch Size vs Accuracy
    plt.subplot(2, 2, 3)
    batch_sizes = df['config/batch_size'].unique()
    accuracies = [df[df['config/batch_size'] == bs]['accuracy'].mean() 
                 for bs in batch_sizes]
    plt.bar([str(bs) for bs in batch_sizes], accuracies)
    plt.xlabel('Batch Size')
    plt.ylabel('Average Accuracy')
    plt.title('Batch Size vs Average Accuracy')
    
    # Plot 4: Training Progress of Best Trial
    plt.subplot(2, 2, 4)
    # Get the data for the best trial
    best_trial_df = df[df['trial_id'] == best_trial.trial_id]
    plt.plot(best_trial_df['training_iteration'], best_trial_df['accuracy'])
    plt.xlabel('Training Iteration')
    plt.ylabel('Accuracy')
    plt.title('Best Trial Training Progress')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Number of trials: {len(df)}")
    print(f"Average accuracy: {df['accuracy'].mean():.4f}")
    print(f"Std deviation: {df['accuracy'].std():.4f}")
    print(f"Min accuracy: {df['accuracy'].min():.4f}")
    print(f"Max accuracy: {df['accuracy'].max():.4f}")

if __name__ == "__main__":
    logger.info("Starting hyperparameter tuning...")
    tune_cifar()
    logger.info("Analyzing results...")
    analyze_results()





