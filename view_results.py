import os
import argparse
from tensorboard import program
import webbrowser
from ray.tune import ExperimentAnalysis

def launch_tensorboard(logdir, port=6006):
    """
    Launch TensorBoard server and open it in the default web browser
    
    Args:
        logdir (str): Directory containing the logs
        port (int): Port to run TensorBoard on
    """
    # Start TensorBoard
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--port', str(port)])
    url = tb.launch()
    print(f"TensorBoard started at {url}")
    
    # Open in browser
    webbrowser.open(url)
    
    try:
        input("Press Enter to stop TensorBoard...")
    except KeyboardInterrupt:
        print("\nStopping TensorBoard...")

def get_latest_experiment():
    """Get the path to the most recent experiment"""
    base_dir = os.path.abspath("ray_results")
    if not os.path.exists(base_dir):
        raise FileNotFoundError("No ray_results directory found!")
        
    experiments = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
    if not experiments:
        raise FileNotFoundError("No experiments found in ray_results directory!")
        
    return max(experiments, key=os.path.getmtime)

def print_experiment_summary(experiment_path):
    """Print summary of the experiment results"""
    analysis = ExperimentAnalysis(experiment_path)
    best_trial = analysis.get_best_trial("accuracy", "max", "last")
    
    print("\n=== Experiment Summary ===")
    print(f"Experiment path: {experiment_path}")
    print(f"\nBest Trial Results:")
    print(f"Trial ID: {best_trial.trial_id}")
    print(f"Accuracy: {best_trial.last_result['accuracy']:.4f}")
    print("\nBest Configuration:")
    for param, value in best_trial.config.items():
        print(f"{param}: {value}")

def main():
    parser = argparse.ArgumentParser(description='View Ray Tune results in TensorBoard')
    parser.add_argument('--logdir', type=str,  help='Path to experiment directory')
    parser.add_argument('--port', type=int, default=6006, help='Port for TensorBoard')
    args = parser.parse_args()
    
    try:
        # Use provided path or get latest experiment
        experiment_path = args.logdir if args.logdir else get_latest_experiment()
        
        # Print experiment summary
        print_experiment_summary(experiment_path)
        
        # Launch TensorBoard
        launch_tensorboard(experiment_path, args.port)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 