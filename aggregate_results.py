import json
import glob
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import wandb
import os
import sys

# Function to load results from JSON files
def load_results(directory, job_id):
    result_files = glob.glob(f'{directory}/result_{job_id}_*.json')
    all_results = []

    for file in result_files:
        with open(file, 'r') as f:
            results = json.load(f)
            all_results.append(results)
    
    return all_results

# Function to compute mean and confidence intervals
def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data, axis=0)
    sem = stats.sem(data, axis=0)
    interval = stats.t.interval(confidence, n-1, loc=mean, scale=sem)
    return mean, interval

# Function to aggregate results
def aggregate_results(results):
    aggregated = {}
    metrics = ['training loss', 'training accuracy', 'test loss', 'test accuracy', 'pseudogradient l2 norm mean', 'pseudogradient l2 norm standard deviation']
    
    for metric in metrics:
        data = [res[metric] for res in results]
        mean, interval = compute_confidence_interval(np.array(data))
        aggregated[metric] = {
            'mean': mean,
            'interval': interval
        }
    
    return aggregated

# Function to plot and log results to W&B
def plot_and_log_results(aggregated_results, job_id):
    wandb.init(project='Confidence Interval', name=f'Aggregated_Results_Job_{job_id}')

    for metric, values in aggregated_results.items():
        mean = values['mean']
        lower = values['interval'][0]
        upper = values['interval'][1]
        
        # Plotting
        plt.figure()
        plt.fill_between(range(len(mean)), lower, upper, alpha=0.2)
        plt.plot(range(len(mean)), mean, label=metric)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.title(f'{metric} with Confidence Interval Job {job_id}')
        
        # Log the plot to W&B
        wandb.log({f"{metric}_confidence_interval": wandb.Image(plt)})
        plt.clf()  # Clear the plot for the next metric

# Function to save aggregated results to JSON and text files
def save_aggregated_results(aggregated_results, job_id):
    results_directory = '/net/scratch/sulee/fedada2_all/results_aggregated'
    os.makedirs(results_directory, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_compatible_results = {}
    for metric, values in aggregated_results.items():
        json_compatible_results[metric] = {
            'mean': values['mean'].tolist(),
            'interval': [values['interval'][0].tolist(), values['interval'][1].tolist()]
        }

    json_file = os.path.join(results_directory, f'aggregated_results_{job_id}.json')
    text_file = os.path.join(results_directory, f'aggregated_results_{job_id}.txt')
    
    with open(json_file, 'w') as f:
        json.dump(json_compatible_results, f, indent=4)
    
    with open(text_file, 'w') as f:
        for metric, values in json_compatible_results.items():
            f.write(f"{metric}:\n")
            f.write(f"  Mean: {values['mean']}\n")
            f.write(f"  Confidence Interval: {values['interval']}\n\n")


# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: python aggregate_results.py <job_id>")
        sys.exit(1)

    job_id = sys.argv[1]
    results_directory = '/net/scratch/sulee/fedada2_all/results'

    results = load_results(results_directory, job_id)
    aggregated_results = aggregate_results(results)
    plot_and_log_results(aggregated_results, job_id)
    save_aggregated_results(aggregated_results, job_id)

if __name__ == "__main__":
    main()
