import optuna
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_optimization_history(study):
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Optimization History")
    plt.show()

def plot_param_importances(study):
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title("Parameter Importances")
    plt.show()

def plot_parallel_coordinate(study):
    plt.figure(figsize=(20, 10))
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.title("Parallel Coordinate Plot")
    plt.show()

def main():
    # Load the study
    study = optuna.load_study(
        study_name="cifar10_optimization",
        storage="sqlite:///optuna_study.db"
    )
    
    logger.info("\nStudy statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(study.get_trials(states=[TrialState.PRUNED]))}")
    logger.info(f"  Number of complete trials: {len(study.get_trials(states=[TrialState.COMPLETE]))}")
    
    logger.info("\nBest trial:")
    trial = study.best_trial
    logger.info(f"  Best test accuracy: {trial.value:.4f}")
    logger.info("  Best parameters:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    # Plot results
    plot_optimization_history(study)
    plot_param_importances(study)
    plot_parallel_coordinate(study)

if __name__ == "__main__":
    main() 