from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from tqdm import tqdm
import logging
from Models import get_cifar10_model, get_cifar10_model2, get_cifar10_model3
from transforms import get_train_transforms, get_test_transforms, AlbumentationsDataset
import os
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
import numpy as np
import argparse
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dataloaders(batch_size):
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
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

def objective(trial):
    """Optuna objective function to maximize test accuracy"""
    
        # Add trial number logging at the start of each trial
    logger.info(f"\nTrial {trial.number} started...")
    # Generate the hyperparameters with adjusted ranges

    config = {
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),  # Removed 64 as it's too small
        "lr": trial.suggest_float("lr", 1e-3, 5e-2, log=True),  # Narrowed range
        "momentum": trial.suggest_float("momentum", 0.85, 0.95),  # Narrowed to better range
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True),  # Adjusted for better regularization
        "max_lr": trial.suggest_float("max_lr", 0.1, 0.5, log=True),  # Adjusted range
        "pct_start": trial.suggest_float("pct_start", 0.2, 0.3),  # Narrowed to better range
        "div_factor": trial.suggest_categorical("div_factor", [10, 15, 20]),  # Adjusted values
        "final_div_factor": trial.suggest_categorical("final_div_factor", [100, 200]),  # Adjusted values
        "epochs": 30  # Increased epochs
    }
    
    logger.info(f"Trial {trial.number} using parameters: {config}")

    # Get dataloaders
    train_loader, test_loader = get_dataloaders(config["batch_size"])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = get_cifar10_model3().to(device)
    
    # Setup optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )
    
    # Setup scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["max_lr"],
        epochs=config["epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=config["pct_start"],
        div_factor=config["div_factor"],
        final_div_factor=config["final_div_factor"]
    )
    
    # Track best accuracy and early stopping counter
    best_accuracy = 0.0
    patience = 5  # Number of epochs to wait for improvement
    patience_counter = 0
    min_accuracy_threshold = 0.70  # Minimum accuracy threshold after 10 epochs
    
    # Training loop
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
        train_accuracy = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
            
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = F.nll_loss(outputs, target)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_accuracy = test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)

        # Add more detailed logging per epoch
        logger.info(
            f"Trial {trial.number} Epoch {epoch}: "
            f"Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}, "
            f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
        )
        
        # Update best accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping conditions
        if epoch >= 10 and test_accuracy < min_accuracy_threshold:
            # If accuracy is too low after 10 epochs, stop this trial
            raise optuna.TrialPruned(f"Accuracy {test_accuracy:.4f} below threshold {min_accuracy_threshold}")
            
        if patience_counter >= patience:
            # If no improvement for 'patience' epochs, stop this trial
            break
        
        # Report intermediate value for pruning
        trial.report(test_accuracy, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_accuracy

def plot_optimization_history(study):
    """Plot the optimization history"""
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Optimization History")
    plt.show()

def plot_param_importances(study):
    """Plot parameter importances"""
    # Check if we have enough trials for parameter importance
    completed_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    if len(completed_trials) < 2:
        logger.warning("Need at least 2 completed trials to plot parameter importances")
        return
    
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title("Parameter Importances")
    plt.show()

def plot_parallel_coordinate(study):
    """Plot parallel coordinate"""
    # Check if we have any completed trials
    completed_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    if len(completed_trials) < 1:
        logger.warning("Need at least 1 completed trial to plot parallel coordinates")
        return
        
    plt.figure(figsize=(20, 10))
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.title("Parallel Coordinate Plot")
    plt.show()

def print_plot_trialresults(study):
    """Print and plot trial results"""
    completed_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    if len(completed_trials) > 0:
        logger.info("\nBest trial:")
        trial = study.best_trial
        
        logger.info(f"  Best test accuracy: {trial.value:.4f}")
        logger.info("  Best parameters:")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")
        
        # Plot results
        plot_optimization_history(study)
        
        # Only plot parameter importances if we have enough trials
        if len(completed_trials) >= 2:
            plot_param_importances(study)
            
        plot_parallel_coordinate(study)
    else:
        logger.warning("No completed trials found. The study may not have converged.")

def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='Optuna hyperparameter optimization for CIFAR10')
    parser.add_argument('--output', 
                       type=str, 
                       default='best_parameters.txt',
                       help='Output file name for best parameters (default: best_parameters.txt)')
    args = parser.parse_args()

    # Set logging level to DEBUG for more verbose output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting Optuna hyperparameter optimization...")

    # Add callback for additional trial information
    def print_callback(study, trial):
        logger.info(f"\nTrial {trial.number} finished:")
        if trial.value is not None:
            logger.info(f"  Value: {trial.value:.4f}")
            logger.info(f"  Params: {trial.params}")
            try:
                if study.best_trial == trial:
                    logger.info("  Best trial so far!")
            except ValueError:
                # This happens when there's no successful trial yet
                logger.info("  First successful trial!")
        else:
            logger.info("  Trial failed or pruned")
    
    # Create a new study with more specific configuration
    study = optuna.create_study(
        direction="maximize",  # We want to maximize test accuracy
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1,
            n_min_trials=5
        ),
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=5,  # Number of random trials before TPE starts
            seed=42  # For reproducibility
        ),
        study_name="cifar10_optimization-19thDec-4",
        storage="sqlite:///optuna_study.db"             # Save to SQLite database
    )
    
    # Optimize with more trials for better exploration
    study.optimize(
        objective, 
        n_trials=30,  # Increased number of trials
        timeout=None,
        show_progress_bar=True,
        callbacks=[print_callback]  # Add callback here        
    )
    
    # Print results
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    logger.info("\nStudy statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")
    
    # Check if there are any complete trials
    if len(complete_trials) <= 0 or len(pruned_trials) <= 0:
        logger.warning("No complete trials found. The study may not have converged.")
        return
    
    print_plot_trialresults(study)
    trial = study.best_trial
    
    # Save best parameters to the specified file (if not specified, default is best_parameters.txt)
    best_params_path = args.output
    logger.info(f"\nSaving best parameters to {best_params_path}")
    with open(best_params_path, 'w') as f:
        f.write(f"Best test accuracy: {trial.value:.4f}\n")
        f.write("Parameters:\n")  # Clear section header
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")  # Consistent format

    logger.info(f"\nStudy saved to optuna_study.db")
    logger.info("To load this study later, use:")
    logger.info('study = optuna.load_study(study_name="cifar10_optimization-19thDec-4", storage="sqlite:///optuna_study.db")')

if __name__ == "__main__":
    # To run the hyper parameter finding trials for a given model
    main()

    # Code required to view the results of a particular trial - study_name to be changed accordingly
    # study = optuna.load_study(study_name="cifar10_optimization-19thDec-4", storage="sqlite:///optuna_study.db")
    # print_plot_trialresults(study)




