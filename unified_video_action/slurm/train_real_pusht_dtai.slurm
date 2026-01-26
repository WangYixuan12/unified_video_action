#!/bin/bash
#SBATCH --job-name="uva_real_pusht_dtai"
#SBATCH --output="/projects/bcyd/ywang41/unified_video_action/slurm/outputs/%x/%j.out"
#SBATCH --mem=120g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ghx4
#SBATCH --time=48:00:00
#SBATCH --account=bfqx-dtai-gh
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

export WANDB_API_KEY=65a3cb43423c6ede326b59f04363f333b5ef5f87
export WANDB_DIR='/projects/bcyd/ywang41/diffusion-forcing/wandb'
export WANDB_DATA_DIR='/projects/bcyd/ywang41/diffusion-forcing/wandb/share'
export HYDRA_FULL_ERROR=1

module reset
module load python/miniforge3_pytorch/2.7.0
module load cudnn

source /sw/user/python/miniforge3-pytorch-24.11.3/etc/profile.d/conda.sh
conda activate /u/ywang41/.conda/envs/df_il_2
cd /projects/bcyd/ywang41/unified_video_action

export TMPDIR=/work/hdd/bcyd/ywang41/tmp
mkdir -p $TMPDIR
export WANDB_CACHE_DIR=$TMPDIR
export WANDB_DATA_DIR=$TMPDIR
export WANDB_ARTIFACTS_DIR=$TMPDIR
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

accelerate launch --num_processes=8 train.py \
	--config-dir=. \
	--config-name=uva_real_aloha_dataset_cam_1 \
	model.policy.action_model_params.predict_action=False \
	model.policy.selected_training_mode=video_model \
	model.policy.optimizer.learning_rate=1e-4 \
	task.dataset.dataset_dir='/work/hdd/bcyd/ywang41/diffusion-forcing/data/real_aloha/pusht_1000_1101' \
	task.shape_meta.action.shape=[4] \
	logging.project=uva \
	hydra.run.dir=checkpoints/uva_real_aloha_pusht \
	dataloader.batch_size=32 \
	val_dataloader.batch_size=32 \
	training.checkpoint_every=1
