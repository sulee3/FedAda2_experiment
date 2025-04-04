#!/bin/bash
#SBATCH --job-name=ada_run
#SBATCH --output=/net/scratch/sulee/fedada2_all/logs/%j.out
#SBATCH --error=/net/scratch/sulee/fedada2_all/logs/%j.err
#SBATCH --time 4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 3
#SBATCH --array=0-0
#SBATCH --mem=32GB
#SBATCH --partition=general
#SBATCH --exclude=l001

# Note: you can only use one of mem or mem-per-cpu. (used to be 3 or 10 CPUs) 
# mem=32GB OR mem-per-cpu=32GB

# Activate the virtual environment
source /home/sulee/FedAda2_experiment/myenv/bin/activate
# source /net/scratch/sulee/venvs/adap_ensemble_venv_fixed/bin/activate


######################### New Hyperparameter Tuning Grid #########################

# client_lr=(1e-3 2e-3 3e-3 4e-3 5e-3 6e-3 7e-3 8e-3 9e-3 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1)
# server_lr=(1e-5)
# client_eps=(1e-7)
# server_eps=(1e-7)


clips=(0.1)
noise_mults=(1)
loc_warms=(10)

# Calculate lengths of arrays
len_server_lr=${#server_lr[@]}
len_client_lr=${#client_lr[@]}
len_server_eps=${#server_eps[@]}
len_client_eps=${#client_eps[@]}
len_ada_mode=${#ada_mode[@]}
len_server_preconditioner=${#server_preconditioner[@]}
len_client_opt=${#client_opt[@]}
len_clips=${#clips[@]}
len_noise_mults=${#noise_mults[@]}
len_loc_warms=${#loc_warms[@]}

let i=$SLURM_ARRAY_TASK_ID%$len_client_lr
let j=($SLURM_ARRAY_TASK_ID/$len_client_lr)%$len_server_lr
let k=($SLURM_ARRAY_TASK_ID/$((len_client_lr*len_server_lr)))%$len_client_eps
let l=($SLURM_ARRAY_TASK_ID/$((len_client_lr*len_server_lr*len_client_eps)))%$len_server_eps
let m=($SLURM_ARRAY_TASK_ID/$((len_client_lr*len_server_lr*len_client_eps*len_server_eps)))%$len_clips
let n=($SLURM_ARRAY_TASK_ID/$((len_client_lr*len_server_lr*len_client_eps*len_server_eps*len_clips)))%$len_noise_mults
let o=($SLURM_ARRAY_TASK_ID/$((len_client_lr*len_server_lr*len_client_eps*len_server_eps*len_clips*len_noise_mults)))%$len_loc_warms

python main_run.py  --ada_mode 0  --server_preconditioner 0  --client_lr ${client_lr[$i]}  --server_lr ${server_lr[$j]}  \
--client_eps ${client_eps[$k]}  --server_eps ${server_eps[$l]} \
--client_opt SM3_adam_noBC \
--server_opt Adam_noBC \
--model vit --dataset cifar100 \
--eval_every_kepochs 5 \
--epochs 601 \
--num_users 1000 \
--frac 0.01 \
--local_ep 1 \
--local_bs 32 \
--sigma ${noise_mults[$n]} \
--clip ${clips[$m]} \
--diff_private 0 \
--update_delay 4 \
--server_beta1 0.9 \
--server_beta2 0.999 \
--server_beta1_adagrad 0 \
--fine_tune_mode 0 \
--localwarmup 1 \
--globalwarmup 0 \
--cold_lr_client 0.0 \
--cold_lr_server 0.0 \
--local_warm_backprops ${loc_warms[$o]} \
--global_warmup_epochs_function_of_global_epoch_step 30 \
--reload_data 0 \
--seed 2 \
--base_project_name  \
