#!/bin/bash
#SBATCH -c 4
#SBATCH --job-name=generate_50
#SBATCH --mem=25GB
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu
#SBATCH --partition=cbmm

#SBATCH -D ./log/

cd /om/user/shobhita/src/chexpert/src/6.819FinalProjectRAMP

hostname
export CUDA_VISIBLE_DEVICES=0
/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/shobhita/src/chexpert/Thorax_GAN/ProGAN/generate_data.py --idx=100
