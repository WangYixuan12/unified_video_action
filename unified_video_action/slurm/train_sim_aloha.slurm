#!/bin/bash
#SBATCH --job-name="uva"
#SBATCH --output="/projects/bcyd/ywang41/unified_video_action/configurations/slurm_scripts/slurm_outputs/%x/%j.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=32   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcyd-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 48:00:00

export WANDB_API_KEY=65a3cb43423c6ede326b59f04363f333b5ef5f87
export WANDB_DIR='/projects/bcyd/ywang41/unified_video_action/wandb'
export WANDB_DATA_DIR='/projects/bcyd/ywang41/unified_video_action/wandb/share'
export HYDRA_FULL_ERROR=1

source /u/ywang41/miniforge3/etc/profile.d/conda.sh
conda activate diffusion-forcing
cd /projects/bcyd/ywang41/unified_video_action
accelerate launch --num_processes=2 --main_process_port=29501 train.py \
	--config-dir=. \
	--config-name=uva_sim_aloha_dataset \
	model.policy.action_model_params.predict_action=False \
	model.policy.selected_training_mode=video_model \
	model.policy.optimizer.learning_rate=1e-4 \
	task.dataset.dataset_dir='/scratch/bcyd/ywang41/diffusion-forcing/data/sim_aloha/pusht_0407' \
	logging.project=uva \
	hydra.run.dir=checkpoints/uva_sim_aloha_data_video_model \
	dataloader.batch_size=16 \
	val_dataloader.batch_size=16 \
	training.checkpoint_every=1
