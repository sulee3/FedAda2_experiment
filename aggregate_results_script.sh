#!/bin/bash
#SBATCH --job-name=conf_interval
#SBATCH --output=/home/sulee/FedAda2_experiment/slurm_out/fedavg/%j.out
#SBATCH --error=/home/sulee/FedAda2_experiment/slurm_out/fedavg/%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=tianlab

# Activate the virtual environment
source /home/sulee/FedAda2_experiment/myenv/bin/activate

# Run the aggregation script
python aggregate_results.py $1

# How to use: On terminal, submit the two lines below, one after another.
# Navigate to appropriate directory first, for me it is: /home/sulee/FedAda2_experiment/$ 
# job_id=$(sbatch --parsable slurm_confidence.sh)
# sbatch --dependency=afterok:$job_id aggregate_results_script.sh $job_id
